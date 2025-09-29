#!/usr/bin/env bash
set -euo pipefail

# Minimal Vulkan inference runner using NCNN binary
# Usage: ./infer_vulkan.sh [num_samples] [out_base]

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NUM_SAMPLES="${1:-1}"
OUT_BASE="${2:-outputs/output_vulkan}"

BIN="$ROOT_DIR/cpp_vulkan/build/pixart_vulkan"
PARAM="$ROOT_DIR/models/unet.param"
BINW="$ROOT_DIR/models/unet.bin"

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found: $BIN" >&2
  exit 1
fi
if [[ ! -f "$PARAM" || ! -f "$BINW" ]]; then
  echo "Error: NCNN model not found: $PARAM $BINW" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_BASE")"

CMD="\"$BIN\" \"$PARAM\" \"$BINW\" $NUM_SAMPLES \"$OUT_BASE\""

echo "Running $NUM_SAMPLES sample(s) on Vulkan..."
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -p bash -lc "$CMD"
else
  time bash -lc "$CMD"
fi

if [[ -f "$OUT_BASE.ppm" ]] && command -v sips >/dev/null 2>&1; then
  sips -s format png "$OUT_BASE.ppm" --out "$OUT_BASE.png" >/dev/null
  echo "Saved $OUT_BASE.png"
fi

echo "Done. Outputs: ${OUT_BASE}.ppm (and .png if sips available)"


