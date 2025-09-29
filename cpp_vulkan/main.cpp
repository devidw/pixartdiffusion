#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>

static constexpr int ART_SIZE = 32;
static constexpr int NUM_CHANNELS = 3;
static constexpr int STEPS = 500;
static constexpr int ACTUAL_STEPS = 490;

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

// Run one UNet forward using NCNN. Inputs:
// x: NCHW float32, t: N length float32 timestamps
static ncnn::Mat run_unet(ncnn::Net& net, const ncnn::Mat& x, const ncnn::Mat& t, ncnn::VkCompute* cmd)
{
	ncnn::Extractor ex = net.create_extractor();
	ex.set_vulkan_compute(true);
	if (cmd) ex.set_vkcompute(cmd);

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
				 float noise_mul,
				 ncnn::VkCompute* cmd)
{
	const int N = im.c; // NCNN uses dims: w,h,c for 3D; for 4D use Mat with dims=4

	// For simplicity, use CPU to generate noise tensor, then upload
	std::vector<float> z_data(im.total());
	std::mt19937 rng(12345);
	std::normal_distribution<float> dist(0.f, 1.f);
	for (size_t i = 0; i < z_data.size(); ++i) z_data[i] = dist(rng);

	ncnn::Mat z = ncnn::Mat(ART_SIZE, ART_SIZE, NUM_CHANNELS * N, (void*)z_data.data());
	z = z.clone();

	if (t == 1)
	{
		z.fill(0.f);
	}

	// Prepare timestamp vector
	ncnn::Mat ts(N);
	for (int i = 0; i < N; ++i) ts[i] = (float)t;

	// Run UNet to get noise prediction
	ncnn::Mat noise = run_unet(net, im, ts, cmd);

	float beta_t = get_beta_scalar(t);
	float alpha = 1.0f - beta_t;
	float denom = std::sqrt(1.0f - alpha_t[t]);

	// new_mean = alpha^(-0.5) * (im - (1 - alpha)/denom * noise)
	ncnn::Mat new_mean = im.clone();
	new_mean.substract_mul(noise, (1.f - alpha) / denom);
	new_mean *= 1.0f / std::sqrt(alpha);

	// add_noise = beta_t * z * noise_mul
	z *= beta_t * noise_mul;
	new_mean += z;
	return new_mean;
}

static ncnn::Mat sample_from(ncnn::Net& net,
			  ncnn::Mat im,
			  int t,
			  const std::vector<float>& alpha_t,
			  float noise_mul,
			  ncnn::VkCompute* cmd)
{
	for (int i = t; i >= 1; --i)
	{
		im = sample_step(net, im, i, alpha_t, noise_mul, cmd);
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

	if (net.load_param(param_path)) { std::cerr << "Failed to load param" << std::endl; return -1; }
	if (net.load_model(bin_path)) { std::cerr << "Failed to load bin" << std::endl; return -1; }

	std::vector<float> alpha_t = compute_alpha_t();

	// NCNN Mat layout for 4D is shaped as w,h,c,elempack with dims=3; for batch we can stack channels
	// We will pack batch into channel dimension: channels = N*C
	const int packed_channels = num_samples * NUM_CHANNELS;
	ncnn::Mat h(ART_SIZE, ART_SIZE, packed_channels);

	// Initialize with random normal
	{
		std::mt19937 rng(42);
		std::normal_distribution<float> dist(0.f, 1.f);
		for (size_t i = 0; i < h.total(); ++i) ((float*)h.data)[i] = dist(rng);
	}

	// Use a command queue
	ncnn::VkCompute cmd(ncnn::get_gpu_device());

	auto t0 = std::chrono::high_resolution_clock::now();
	h = sample_from(net, h, ACTUAL_STEPS, alpha_t, /*noise_mul=*/7.5f, &cmd);
	cmd.submit_and_wait();
	auto t1 = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	std::cout << "Sampling time: " << ms << " ms total (" << (ms / (double)num_samples) << " ms per sample)" << std::endl;

	// Save first sample to PPM
	try {
		// Extract first sample (first 3 channels)
		ncnn::Mat im0(ART_SIZE, ART_SIZE, NUM_CHANNELS);
		for (int c = 0; c < NUM_CHANNELS; ++c)
		{
			const float* src = h.channel(c);
			float* dst = im0.channel(c);
			std::memcpy(dst, src, ART_SIZE * ART_SIZE * sizeof(float));
		}

		// Clamp to [0,1]
		for (size_t i = 0; i < im0.total(); ++i)
		{
			float v = ((float*)im0.data)[i];
			v = std::max(0.f, std::min(1.f, v));
			((float*)im0.data)[i] = v;
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
					float v = im0.channel(c).row(y)[x];
					int u8 = (int)std::round(v * 255.f);
					rgb[(y * ART_SIZE + x) * 3 + c] = (unsigned char)std::max(0, std::min(255, u8));
				}
			}
		}
		ofs.write((const char*)rgb.data(), rgb.size());
		ofs.close();
		std::cout << "Saved " << ppm_path << std::endl;
	}
	catch (...) {
		std::cerr << "Warning: failed to write PPM" << std::endl;
	}

	ncnn::destroy_gpu_instance();
	return 0;
}


