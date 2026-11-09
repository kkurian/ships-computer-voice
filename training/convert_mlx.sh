#!/bin/bash
# Convert a fine-tuned checkpoint to the 6-bit MLX model say.py runs, on the Mac:
#
#   bash training/convert_mlx.sh ft-output/output/checkpoint-epoch-0 my-voice-mlx-6bit
#
# Uses mlx-audio from the .qwen venv (requirements-qwen.txt). 6-bit weights,
# quantisation groups of 64. The 1.8 GB bf16 checkpoint becomes about 1.8 GB on
# disk including the unquantised speech tokenizer (0.68 GB), and about 1.9 GiB
# resident when loaded.
set -euo pipefail
SRC=$1; DST=$2
PY=${SHIPS_COMPUTER_QWEN_PYTHON:-$(dirname "$0")/../.qwen/bin/python}
"$PY" -m mlx_audio.convert --hf-path "$SRC" --mlx-path "$DST" --quantize --q-bits 6 --q-group-size 64
echo "converted: $DST"
