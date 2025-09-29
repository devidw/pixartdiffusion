import argparse
import os
from typing import Optional
import time

import torch

from pixartdiffusion import model as diffusion_model
from pixartdiffusion import util, sample, parameters


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        preferred = preferred.lower()
        if preferred == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        if preferred == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        if preferred == 'cpu':
            return torch.device('cpu')

    # Auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync_device(dev: torch.device) -> None:
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elif dev.type == 'mps':
        # Ensure pending GPU ops are finished for accurate timing
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate pixel art with PixArtDiffusion")

    parser.add_argument('--model', '-m', type=str, default='models/AOS_AOF.pt', help='Path to UNet checkpoint (.pt)')
    parser.add_argument('--out', '-o', type=str, default='outputs/output.png', help='Output spritesheet path (.png)')
    parser.add_argument('--width', type=int, default=1, help='Spritesheet width (columns)')
    parser.add_argument('--height', type=int, default=1, help='Spritesheet height (rows)')
    parser.add_argument('--noise-mul', type=float, default=7.5, help='Sampling noise multiplier (higher = more chaotic)')
    parser.add_argument('--display-count', type=int, default=0, help='Times to display intermediate result (0 to disable)')
    parser.add_argument('--device', type=str, default=None, help='Force device: cuda|mps|cpu (default: auto)')

    # CLIP selection
    parser.add_argument('--select', action='store_true', help='Enable CLIP re-ranking selection over extra candidates')
    parser.add_argument('--select-prompt', type=str, default='cool colourful pixel art character #pixelart', help='Prompt used for CLIP selection')
    parser.add_argument('--gen-mult', type=int, default=1, help='Candidates per output used for CLIP selection (1 disables)')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32', help='CLIP model to load (via openai-clip)')

    # CLIP guidance (slow)
    parser.add_argument('--guidance', action='store_true', help='Enable CLIP guidance during sampling (very slow)')
    parser.add_argument('--guidance-prompt', type=str, default='a green octopus #pixelart', help='Prompt for CLIP guidance')
    parser.add_argument('--guidance-shiftn', type=int, default=8, help='Number of spatial shifts for CLIP guidance')
    parser.add_argument('--guidance-mul', type=float, default=100.0, help='Strength multiplier for CLIP guidance')
    parser.add_argument('--guidance-dropoff', type=float, default=0.25, help='0..1 dropoff factor across timesteps')

    args = parser.parse_args()

    device = resolve_device(args.device)

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Build and load model
    net = diffusion_model.UNet().to(device).eval()
    epoch = util.load_model(net, args.model)
    print(f"Loaded model from epoch {epoch}")
    print(f"Using device: {device}")

    # Optionally load CLIP
    clip_model = None
    tokenized_text_select = None
    classifier_func = None

    if args.select or args.guidance:
        try:
            import clip  # openai-clip
        except Exception as e:
            raise RuntimeError("CLIP features requested but 'clip' package not installed. pip install git+https://github.com/openai/CLIP.git") from e

        print("Loading CLIP model...")
        t_clip_load_start = time.perf_counter()
        clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
        sync_device(device)
        print(f"CLIP loaded in {time.perf_counter() - t_clip_load_start:.3f}s")

        # Validate CLIP resolution divisibility
        assert clip_model.visual.input_resolution % parameters.ART_SIZE == 0, "CLIP input resolution must be divisible by ART_SIZE"

        if args.select:
            tokenized_text_select = clip.tokenize([args.select_prompt]).to(device)

        if args.guidance:
            tokenized_text_guidance = clip.tokenize([args.guidance_prompt]).to(device)

            def get_sigma_classifier(t: torch.Tensor) -> torch.Tensor:
                L = torch.tensor(args.guidance_dropoff, dtype=torch.float)
                H = torch.tensor(1.0, dtype=torch.float)
                return args.guidance_mul * ((H - L) * t/parameters.STEPS + L).float()

            classifier_func = sample.clip_grad_func(
                clip_model,
                tokenized_text_guidance,
                num_shifts=args.guidance_shiftn,
                shift_range=(clip_model.visual.input_resolution // parameters.ART_SIZE) * 2,
            )
        else:
            get_sigma_classifier = None
    else:
        get_sigma_classifier = None

    # Determine number of total samples
    top_n = args.width * args.height
    total = top_n if args.gen_mult <= 1 else top_n * args.gen_mult
    print(f"Generating {args.width} * {args.height} * {max(1, args.gen_mult)} = {total} samples...")

    t_all_start = time.perf_counter()
    with torch.no_grad():
        sync_device(device)
        t_sample_start = time.perf_counter()
        xs = sample.sample(
            net,
            total,
            display_count=args.display_count,
            noise_mul=args.noise_mul,
            classifier_func=classifier_func,
            classifier_mul_func=get_sigma_classifier,
        )
        sync_device(device)
        sample_time = time.perf_counter() - t_sample_start
        per_candidate = sample_time / max(1, total)
        print(f"Sampling time: {sample_time:.3f}s (per candidate: {per_candidate:.3f}s)")

        # CLIP re-ranking selection
        if args.select and args.gen_mult > 1:
            t_rerank_start = time.perf_counter()
            xs = sample.CLIP_rerank(clip_model, xs, tokenized_text_select)[:top_n]
            sync_device(device)
            print(f"CLIP re-ranking time: {time.perf_counter() - t_rerank_start:.3f}s for {total} candidates")
        else:
            xs = xs[:top_n]

        t_sheet_start = time.perf_counter()
        sheet = util.to_drawable(xs, fix_width=args.width)
        from matplotlib import image as mpl_image
        mpl_image.imsave(args.out, sheet)
        io_time = time.perf_counter() - t_sheet_start
        total_time = time.perf_counter() - t_all_start
        print(f"Saved {args.out}")
        print(f"Sheet+save time: {io_time:.3f}s  |  End-to-end: {total_time:.3f}s")


if __name__ == '__main__':
    main()


