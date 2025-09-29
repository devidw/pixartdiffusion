#!/usr/bin/env bash
set -euo pipefail

# Minimal inference runner (assumes C++ binary and TorchScript model already exist)
# Usage: ./infer.sh [num_samples] [out_base]

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NUM_SAMPLES="${1:-1}"
OUT_BASE="${2:-outputs/output_cpu}"

BIN="$ROOT_DIR/cpp/build/pixart_cpu"
MODEL_TS="$ROOT_DIR/models/unet_scripted.pt"

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_TS" ]]; then
  echo "Error: TorchScript model not found: $MODEL_TS" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_BASE")"

CMD="\"$BIN\" \"$MODEL_TS\" $NUM_SAMPLES \"$OUT_BASE.pt\""

echo "Running $NUM_SAMPLES sample(s)..."
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -p bash -lc "$CMD"
else
  time bash -lc "$CMD"
fi

# Convert preview to PNG if possible
if [[ -f "$OUT_BASE.ppm" ]] && command -v sips >/dev/null 2>&1; then
  sips -s format png "$OUT_BASE.ppm" --out "$OUT_BASE.png" >/dev/null
  echo "Saved $OUT_BASE.png"
fi

echo "Done. Outputs: $OUT_BASE.pt, ${OUT_BASE}.ppm (and .png if sips available)"
