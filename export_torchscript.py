import argparse
import torch
from pixartdiffusion import model as diffusion_model
from pixartdiffusion import util


def main():
    parser = argparse.ArgumentParser(description="Export UNet to TorchScript for C++ libtorch")
    parser.add_argument('--model', '-m', type=str, default='models/AOS_AOF.pt', help='Path to UNet checkpoint (.pt)')
    parser.add_argument('--out', '-o', type=str, default='models/unet_scripted.pt', help='Output TorchScript path')
    args = parser.parse_args()

    device = torch.device('cpu')
    net = diffusion_model.UNet().to(device).eval()
    util.load_model(net, args.model)

    # Example inputs: (N, C, H, W), (N)
    example_x = torch.zeros(1, 3, 32, 32, dtype=torch.float, device=device)
    example_t = torch.ones(1, dtype=torch.long, device=device)

    # Trace or script
    scripted = torch.jit.trace(net, (example_x, example_t))
    scripted.save(args.out)
    print(f"Saved TorchScript to {args.out}")


if __name__ == '__main__':
    main()


