import argparse
import os
import torch
from pixartdiffusion import model as diffusion_model
from pixartdiffusion import util


def main():
    parser = argparse.ArgumentParser(description="Export UNet to ONNX for Vulkan/NCNN")
    parser.add_argument('--model', '-m', type=str, default='models/AOS_AOF.pt', help='Path to UNet checkpoint (.pt)')
    parser.add_argument('--out', '-o', type=str, default='models/unet.onnx', help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--batch', type=int, default=1, help='Export batch size')
    args = parser.parse_args()

    device = torch.device('cpu')
    net = diffusion_model.UNet().to(device).eval()
    util.load_model(net, args.model)

    # Example inputs: (N, C, H, W), (N)
    N, C, H, W = args.batch, 3, 32, 32
    example_x = torch.zeros(N, C, H, W, dtype=torch.float, device=device)
    # Use float timestamps to avoid integer cast ops in ONNX for broader runtime support
    example_t = torch.ones(N, dtype=torch.float, device=device)

    # For mobile runtimes like NCNN, prefer fixed batch shapes
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    torch.onnx.export(
        net,
        (example_x, example_t),
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["x", "t"],
        output_names=["out"],
        dynamic_axes=None,
    )
    print(f"Saved ONNX to {args.out}")


if __name__ == '__main__':
    main()


