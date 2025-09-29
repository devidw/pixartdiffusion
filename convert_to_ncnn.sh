#!/usr/bin/env bash
set -euo pipefail

# Convert ONNX to NCNN param/bin using ncnn tools (onnx2ncnn + ncnnoptimize)
# Usage: ./convert_to_ncnn.sh models/unet.onnx models/unet.param models/unet.bin

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <in.onnx> <out.param> <out.bin>" >&2
  exit 1
fi

IN_ONNX="$1"
OUT_PARAM="$2"
OUT_BIN="$3"

if ! command -v onnx2ncnn >/dev/null 2>&1; then
  echo "Error: onnx2ncnn not found in PATH. Install ncnn or add tools to PATH." >&2
  exit 1
fi
if ! command -v ncnnoptimize >/dev/null 2>&1; then
  echo "Error: ncnnoptimize not found in PATH. Install ncnn or add tools to PATH." >&2
  exit 1
fi

TMP_PARAM="${OUT_PARAM%.param}.raw.param"
TMP_BIN="${OUT_BIN%.bin}.raw.bin"

onnx2ncnn "$IN_ONNX" "$TMP_PARAM" "$TMP_BIN"

# Optimize for Vulkan GPU; fp16 storage helps bandwidth, fp16 arithmetic off for stability
ncnnoptimize "$TMP_PARAM" "$TMP_BIN" "$OUT_PARAM" "$OUT_BIN" 0

rm -f "$TMP_PARAM" "$TMP_BIN"
echo "NCNN wrote: $OUT_PARAM $OUT_BIN"


