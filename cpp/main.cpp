#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>

// Diffusion params (must match Python)
static constexpr int ART_SIZE = 32;
static constexpr int NUM_CHANNELS = 3;
static constexpr int STEPS = 500;
static constexpr int ACTUAL_STEPS = 490;

// Linear beta schedule used in noise.py
torch::Tensor get_beta(const torch::Tensor& t) {
    auto L = torch::tensor(0.001f);
    auto H = torch::tensor(0.018f);
    return ((H - L) * t.to(torch::kFloat32) / STEPS + L).to(torch::kFloat32);
}

// Precompute alpha_t array (size STEPS+1)
torch::Tensor compute_alpha_t() {
    auto alpha_t = torch::zeros({STEPS + 1}, torch::kFloat32);
    float prod = 1.0f;
    for (int i = 0; i <= STEPS; ++i) {
        auto beta = get_beta(torch::tensor(i));
        prod *= (1.0f - beta.item<float>());
        alpha_t[i] = prod;
    }
    return alpha_t;
}

// One reverse step
torch::Tensor sample_step(torch::jit::script::Module& unet,
                          const torch::Tensor& im,
                          int t,
                          const torch::Tensor& alpha_t,
                          float noise_mul) {
    auto N = im.sizes()[0];

    auto z = torch::randn_like(im);
    if (t == 1) {
        z = z * 0.0f;
    }

    auto ts = torch::full({(long)N}, t, torch::kLong);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(im);
    inputs.emplace_back(ts);
    auto noise = unet.forward(inputs).toTensor();

    auto beta_t = get_beta(torch::tensor(t));
    auto alpha = 1.0f - beta_t;
    float denom = std::sqrt(1.0f - alpha_t.index({t}).item<float>());
    auto new_mean = torch::pow(alpha, -0.5f) * (im - (1 - alpha) / denom * noise);

    // Match Python: add_noise = getβ(t) * z * noise_mul
    auto add_noise = get_beta(torch::tensor(t)) * z * noise_mul;
    return new_mean + add_noise;
}

torch::Tensor sample_from(torch::jit::script::Module& unet,
                          torch::Tensor im,
                          int t,
                          const torch::Tensor& alpha_t,
                          float noise_mul) {
    for (int i = t; i >= 1; --i) {
        im = sample_step(unet, im, i, alpha_t, noise_mul);
    }
    return (im + 1.0f) / 2.0f;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: pixart_cpu <unet_scripted.pt> <num_samples> [output.pt]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    int num_samples = std::atoi(argv[2]);
    std::string out_path = argc >= 4 ? argv[3] : "outputs_cpu.pt";

    torch::NoGradGuard no_grad;
    torch::jit::script::Module unet;
    try {
        unet = torch::jit::load(model_path, torch::kCPU);
        unet.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    auto alpha_t = compute_alpha_t();

    auto size = std::vector<int64_t>{num_samples, NUM_CHANNELS, ART_SIZE, ART_SIZE};
    auto h = torch::randn(size, torch::kFloat32);

    auto t0 = std::chrono::high_resolution_clock::now();
    h = sample_from(unet, h, ACTUAL_STEPS, alpha_t, /*noise_mul=*/7.5f);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Sampling time: " << ms << " ms total (" << (ms / (double)num_samples) << " ms per sample)" << std::endl;

    // Save raw tensor for inspection
    torch::save(h, out_path);
    std::cout << "Saved " << out_path << std::endl;

    // Also save first sample as a simple PPM image for quick viewing
    try {
        auto im0 = h.index({0}); // (C,H,W)
        im0 = im0.detach().cpu().clamp(0.0f, 1.0f);
        if (im0.sizes()[0] == 1) {
            im0 = im0.repeat({3,1,1});
        }
        auto H = (int)im0.sizes()[1];
        auto W = (int)im0.sizes()[2];
        auto im_u8 = (im0 * 255.0f).to(torch::kU8).contiguous();
        std::string ppm_path = std::string(out_path);
        auto pos = ppm_path.find_last_of('.');
        if (pos != std::string::npos) ppm_path = ppm_path.substr(0, pos);
        ppm_path += ".ppm";
        std::ofstream ofs(ppm_path, std::ios::binary);
        ofs << "P6\n" << W << " " << H << "\n255\n";
        auto data = im_u8.permute({1,2,0}).contiguous(); // (H,W,C)
        ofs.write(reinterpret_cast<const char*>(data.data_ptr()), W*H*3);
        ofs.close();
        std::cout << "Saved " << ppm_path << std::endl;
    } catch (...) {
        std::cerr << "Warning: failed to write PPM preview" << std::endl;
    }
    return 0;
}


