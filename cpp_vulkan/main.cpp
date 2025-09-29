#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/layer.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstring>

static constexpr int ART_SIZE = 32;
static constexpr int NUM_CHANNELS = 3;
static constexpr int STEPS = 500;
static constexpr int ACTUAL_STEPS = 490;
// Simple no-op custom layer to handle leftover framework ops like Tensor.to
struct NoopLayer : public ncnn::Layer {
	NoopLayer() { one_blob_only = true; support_inplace = false; }
	virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& /*opt*/) const {
		top_blob = bottom_blob.clone();
		return 0;
	}
};

static ncnn::Layer* NoopLayer_creator() { return new NoopLayer(); }


// Match Python beta schedule
static inline float get_beta_scalar(int t)
{
	const float L = 0.001f;
	const float H = 0.018f;
	return ((H - L) * (float)t / (float)STEPS + L);
}

static std::vector<float> compute_alpha_t()
{
	std::vector<float> alpha_t(STEPS + 1);
	float prod = 1.0f;
	for (int i = 0; i <= STEPS; ++i)
	{
		float beta = get_beta_scalar(i);
		prod *= (1.0f - beta);
		alpha_t[i] = prod;
	}
	return alpha_t;
}

// Run one UNet forward using NCNN. Inputs are single-sample tensors.
static ncnn::Mat run_unet(ncnn::Net& net, const ncnn::Mat& x, const ncnn::Mat& t)
{
	ncnn::Extractor ex = net.create_extractor();
	ex.input("x", x);
	ex.input("t", t);

	ncnn::Mat out;
	ex.extract("out", out);
	return out;
}

static ncnn::Mat sample_step(ncnn::Net& net,
			 const ncnn::Mat& im,
			 int t,
			 const std::vector<float>& alpha_t,
			 float noise_mul)
{
	// Prepare scalar timestamp tensor (length 1)
	ncnn::Mat ts(1);
	ts[0] = (float)t;

	// Run UNet to get noise prediction (same shape as input im)
	ncnn::Mat noise = run_unet(net, im, ts);

	float beta_t = get_beta_scalar(t);
	float alpha = 1.0f - beta_t;
	float denom = std::sqrt(1.0f - alpha_t[t]);
	float inv_sqrt_alpha = 1.0f / std::sqrt(alpha);
	float coef = (1.0f - alpha) / denom;

	// CPU elementwise math: new_mean = inv_sqrt_alpha * (im - coef * noise) + beta_t * z * noise_mul
	// Generate z ~ N(0,1)
	std::mt19937 rng(12345 + t);
	std::normal_distribution<float> dist(0.f, 1.f);

	ncnn::Mat out(im.w, im.h, im.c);
	for (int c = 0; c < im.c; ++c)
	{
		const float* pim = im.channel(c);
		const float* pno = noise.channel(c);
		float* pout = out.channel(c);
		for (int i = 0; i < im.w * im.h; ++i)
		{
			float z = (t == 1) ? 0.f : dist(rng);
			float new_mean = inv_sqrt_alpha * (pim[i] - coef * pno[i]);
			pout[i] = new_mean + beta_t * z * noise_mul;
		}
	}
	return out;
}

static ncnn::Mat sample_from(ncnn::Net& net,
		  ncnn::Mat im,
		  int t,
		  const std::vector<float>& alpha_t,
		  float noise_mul)
{
	for (int i = t; i >= 1; --i)
	{
		im = sample_step(net, im, i, alpha_t, noise_mul);
	}
	return im;
}

int main(int argc, char** argv)
{
	if (argc < 5)
	{
		std::cerr << "Usage: pixart_vulkan <unet.param> <unet.bin> <num_samples> <out_base>" << std::endl;
		return 1;
	}
	const char* param_path = argv[1];
	const char* bin_path = argv[2];
	const int num_samples = std::atoi(argv[3]);
	const std::string out_base = argv[4];

	if (ncnn::get_gpu_count() == 0)
	{
		std::cerr << "No Vulkan-capable GPU found." << std::endl;
		return 2;
	}

	ncnn::create_gpu_instance();

	ncnn::Net net;
	net.opt.use_vulkan_compute = 1;
	net.opt.use_fp16_storage = 1;
	net.opt.use_fp16_arithmetic = 0; // stability

	// Register potential leftover ops as no-ops before parsing params
	net.register_custom_layer("Tensor.to", NoopLayer_creator);
	net.register_custom_layer("Tensor.to_11", NoopLayer_creator);

	if (net.load_param(param_path)) { std::cerr << "Failed to load param" << std::endl; return -1; }
	if (net.load_model(bin_path)) { std::cerr << "Failed to load bin" << std::endl; return -1; }

	std::vector<float> alpha_t = compute_alpha_t();

	auto total_start = std::chrono::high_resolution_clock::now();

	for (int s = 0; s < num_samples; ++s)
	{
		// Initialize with random normal
		ncnn::Mat h(ART_SIZE, ART_SIZE, NUM_CHANNELS);
		{
			std::mt19937 rng(42 + s);
			std::normal_distribution<float> dist(0.f, 1.f);
			for (size_t i = 0; i < h.total(); ++i) ((float*)h.data)[i] = dist(rng);
		}

		auto t0 = std::chrono::high_resolution_clock::now();
		h = sample_from(net, h, ACTUAL_STEPS, alpha_t, /*noise_mul=*/7.5f);
		auto t1 = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
		std::cout << "Sample " << (s+1) << "/" << num_samples << ": " << ms << " ms" << std::endl;

		if (s == 0)
		{
			// Clamp and save first sample to PPM
			for (size_t i = 0; i < h.total(); ++i)
			{
				float v = ((float*)h.data)[i];
				v = std::max(0.f, std::min(1.f, v));
				((float*)h.data)[i] = v;
			}

			std::string ppm_path = out_base + ".ppm";
			std::ofstream ofs(ppm_path, std::ios::binary);
			ofs << "P6\n" << ART_SIZE << " " << ART_SIZE << "\n255\n";
			std::vector<unsigned char> rgb(ART_SIZE * ART_SIZE * 3);
			for (int y = 0; y < ART_SIZE; ++y)
			{
				for (int x = 0; x < ART_SIZE; ++x)
				{
					for (int c = 0; c < 3; ++c)
					{
						float v = h.channel(c).row(y)[x];
						int u8 = (int)std::round(v * 255.f);
						rgb[(y * ART_SIZE + x) * 3 + c] = (unsigned char)std::max(0, std::min(255, u8));
					}
				}
			}
			ofs.write((const char*)rgb.data(), rgb.size());
			ofs.close();
			std::cout << "Saved " << ppm_path << std::endl;
		}
	}

	auto total_end = std::chrono::high_resolution_clock::now();
	auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
	std::cout << "Total: " << total_ms << " ms (" << (total_ms / (double)num_samples) << " ms per sample)" << std::endl;

	ncnn::destroy_gpu_instance();
	return 0;
}


