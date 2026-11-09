#!/bin/bash
# The whole fine-tune on one Linux GPU host, unattended. Run it with nohup on the pod:
#
#   nohup bash /workspace/ft/run_pod.sh &
#
# Expects /workspace/ft/ to hold this folder's files and data/ (see
# prepare_clips.py: data/clips/, data/metadata.csv, data/ref.wav). Steps:
# start guard.sh, install pinned dependencies, download the pinned base model
# and tokenizer, prepare the clips, extract audio codes, train, render a sanity
# check. Each epoch writes output/checkpoint-epoch-N/ and then a SAVED marker,
# so copyback.sh can pull it while training goes on. At the end it writes DONE;
# on any error, FAILED. Progress: STATUS (one line per step) and run.log.
#
# Settings (environment): LR (5e-6), EPOCHS (3), BS (batch size, 2; gradient
# accumulation is 4), SPEAKER (computer), PAD (0.4 s of silence after each clip).
set -Eeuo pipefail
W=${W:-/workspace/ft}; cd "$W"
exec >> "$W/run.log" 2>&1
mark(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$W/STATUS"; }
trap 'mark "FAILED at line $LINENO: $BASH_COMMAND"; touch "$W/FAILED"' ERR
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
LR=${LR:-5e-6}; EPOCHS=${EPOCHS:-3}; BS=${BS:-2}; SPEAKER=${SPEAKER:-computer}; PAD=${PAD:-0.4}
QWEN_COMMIT=022e286b98fbec7e1e916cb940cdf532cd9f488e
BASE_REV=5d83992436eae1d760afd27aff78a71d676296fc       # Qwen/Qwen3-TTS-12Hz-0.6B-Base
TOKENIZER_REV=7dd38ad4e9bad454aae9cd937d0cd577604fe229  # Qwen/Qwen3-TTS-Tokenizer-12Hz
mark "start lr=$LR epochs=$EPOCHS bs=$BS speaker=$SPEAKER pad=$PAD"
nohup bash "$W/guard.sh" >> "$W/guard.log" 2>&1 &

# torch comes from the image; torchaudio must match it exactly.
TV=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
mark "torch $TV"
pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/cu128
printf 'torch==%s\ntorchaudio==%s\n' "$TV" "$TV" > constraints.txt
[ -d Qwen3-TTS ] || git clone -q https://github.com/QwenLM/Qwen3-TTS.git
git -C Qwen3-TTS checkout -q "$QWEN_COMMIT"
pip install -q -c constraints.txt ./Qwen3-TTS -r "$W/requirements-training.txt"
pip freeze > pip-freeze.txt
mark "dependencies installed"

python - "$BASE_REV" "$TOKENIZER_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", revision=sys.argv[1], local_dir="models/base")
snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", revision=sys.argv[2], local_dir="models/tokenizer")
PY
mark "models downloaded"

python "$W/prepare_clips.py" "$W/data" "$W/prepared" --pad "$PAD"
cd Qwen3-TTS/finetuning
python prepare_data.py --device cuda:0 --tokenizer_model_path "$W/models/tokenizer" \
  --input_jsonl "$W/prepared/train_raw.jsonl" --output_jsonl "$W/prepared/train_with_codes.jsonl"
mark "audio codes extracted: $(wc -l < "$W/prepared/train_with_codes.jsonl") clips"

cp "$W/sft_12hz_patched.py" .   # it imports dataset.py from this folder
python sft_12hz_patched.py --init_model_path "$W/models/base" --output_model_path "$W/output" \
  --train_jsonl "$W/prepared/train_with_codes.jsonl" --batch_size "$BS" --lr "$LR" \
  --num_epochs "$EPOCHS" --speaker_name "$SPEAKER"
mark "training finished"

cd "$W"
python "$W/sanity.py" "$W/output" "$SPEAKER"
mark "sanity $(tr -d '\n ' < output/sanity.json)"
touch "$W/DONE"; mark DONE
