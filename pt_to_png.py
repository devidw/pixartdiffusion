import argparse
import torch
import numpy as np
from pixartdiffusion import util
from matplotlib import image as mpl_image


def main():
    parser = argparse.ArgumentParser(description="Convert saved .pt tensor (B,C,H,W, 0..1) to PNG spritesheet")
    parser.add_argument("input_pt", type=str, help="Path to .pt tensor saved by C++ app")
    parser.add_argument("output_png", type=str, help="Output PNG path")
    parser.add_argument("--width", type=int, default=None, help="Spritesheet width (columns)")
    args = parser.parse_args()

    x = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    if isinstance(x, dict) and "state_dict" in x:
        raise SystemExit("Input .pt looks like a checkpoint, not a tensor.")

    assert x.ndim == 4, f"Expected tensor of shape (B,C,H,W), got {tuple(x.shape)}"

    sheet = util.to_drawable(x, fix_width=args.width)
    mpl_image.imsave(args.output_png, sheet)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()


