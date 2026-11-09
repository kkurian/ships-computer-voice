# ships-computer-voice

## TL;DR

A text-to-speech voice in the style of the Star Trek: The Next Generation ship's computer, plus a script that speaks with it, streamed. The voice ("version 1") is a fine-tune of Qwen3-TTS 0.6B, run on Apple silicon with mlx-audio. On an M1 it starts speaking 0.41 s after it is called (once loaded) and makes speech faster than real time (0.73 s of work per second of speech). Whisper reads all 7 bench lines back exactly. Training took 5 minutes on one rented NVIDIA A40; the whole pod session, including a failed first run, cost about $0.42.

## Background

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is an open text-to-speech model family from Alibaba's Qwen team, Apache-2.0, code and weights. Its 0.6B "Base" model can be fine-tuned on one speaker. Version 1 was fine-tuned from it on 425 transcribed clips (21 minutes) of the TNG computer voice, chosen from 1,005, and picked by ear on 2026-11-09 from three epochs. The voice weights carry a separate notice: see [VOICE-NOTICE.md](VOICE-NOTICE.md).

This repository used to hold a Piper voice; that is release [v1.0.0](../../releases/tag/v1.0.0). Version 1 replaces it (the voice is "version 1"; it ships in release v2.0.0). On similar bench lines, Whisper read-back went from 0.79 to 1.00 and first audio from about 1.0 s to 0.41 s, at about twice the compute per second of speech.

## What's here

- `say.py`: speaks text, streamed, or renders a WAV file. Standard library only; it runs the model in a worker process.
- `pronounce.tsv`: the pronunciation table (format at the top of the file). `test_say.py`: unit tests for the text handling.
- `requirements-qwen.txt`: the voice runtime (mlx-audio and its dependencies), pinned exactly.
- `training/`: the fine-tuning pipeline: clip preparation, Qwen's fine-tuning script with fixes, an unattended pod runner, a cost guard, a copier, conversion to MLX. `requirements-training.txt` pins it.
- `tools/`: `bench.py` (speed, through say.py), `readback.py` (intelligibility, Whisper), `noise.py` (noise floor). `requirements-tools.txt` pins the last two.
- The voice itself is not in git. It is attached to release [v2.0.0](../../releases/tag/v2.0.0); `SHA256SUMS` here lists its checksums.

## Install

macOS on Apple silicon (M1 or later). Tested with Python 3.12 from Homebrew.

```sh
git clone https://github.com/kkurian/ships-computer-voice.git
cd ships-computer-voice
gh release download v2.0.0 --pattern ships-computer-v1-mlx-6bit.tar   # or download it from the release page
shasum -a 256 --ignore-missing -c SHA256SUMS
tar xf ships-computer-v1-mlx-6bit.tar && rm ships-computer-v1-mlx-6bit.tar
brew install python@3.12 ffmpeg
python3.12 -m venv .qwen && .qwen/bin/pip install -r requirements-qwen.txt
```

The archive is 1.8 GB and unpacks to `ships-computer-v1-mlx-6bit/`.

## Use

```sh
./say.py "Diagnostic complete."
echo "Working." | ./say.py
./say.py --out reply.wav "Unable to comply."
./say.py --say-as data=DAY-tuh "Accessing data."
./say.py --print-text "Project build complete."     # prints: PRAH-ject build complete.
```

- The first call starts a worker: a separate process in `.qwen/` that loads the model (2-20 s, depending on the disk cache), warms it up and then serves requests over a Unix socket, one at a time. It exits after 10 minutes without a request (`SHIPS_COMPUTER_QWEN_IDLE`, seconds). While loaded it holds about 1.9 GiB. `./say.py --stop-worker` stops it.
- Audio streams as it is generated, in half-second pieces, through `ffmpeg` to the speakers, so speech starts before the sentence is finished. Text over 400 characters is generated a few sentences at a time.
- The audio passes through moderate compression, +5 dB and a limiter. Set `SHIPS_COMPUTER_VOICE_FX` to change the `ffmpeg` filter chain, or to `""` to turn it off. On macOS the player is `ffmpeg -f audiotoolbox`, not `ffplay`, which can clip the first syllable.
- Pronunciation: whole words, case-insensitive, respelled from `pronounce.tsv`, then any `pronounce-*.tsv` beside it (your own; git ignores them), then the tables listed in `SHIPS_COMPUTER_PRONOUNCE` (colon-separated), then `--say-as WORD=RESPELLING` (repeatable). Later entries win. Capitals in a respelling mark stress; the voice does not spell them out.
- Fallback: if the model, the venv or `ffmpeg` is missing, or the worker fails before the first audio, `say.py` says why on stderr and speaks with a macOS system voice (`say`), first of Allison (Enhanced), Samantha (Enhanced), Ava (Enhanced), Allison, Samantha, Kathy that is installed. `--out` and hints still apply. On other systems it exits 1.
- `SHIPS_COMPUTER_QWEN_MODEL` and `SHIPS_COMPUTER_QWEN_PYTHON` override where the model and the venv's python are. `SHIPS_COMPUTER_QWEN_RUN` sets the folder for the worker's socket and its logs, one JSONL file per worker session (default `/tmp/ships-computer-voice-<uid>`, private to you).
- Tests: `python3 -m unittest -v test_say` (no model needed; runs on Linux too).

## How it was made

Why Qwen3-TTS
- Zero-shot cloning trial on an M1 (mlx-audio 0.6.4, 6-bit, cloned from 2-4 s reference clips): 0.6B started speaking in 0.78 s with 0.85 s of work per second of speech; 1.7B needed 1.07 s per second, too slow to stream. Both were read back by Whisper at 0.96-0.97.
- Fine-tuning the 0.6B was the next step: no reference clip at run time, a steadier voice, and half the time to first audio (0.41 s against 0.78 s).

Data
- Source: [STCOMP/majel_tts_medium_clean](https://huggingface.co/datasets/STCOMP/majel_tts_medium_clean), 1,005 transcribed clips, 2.0 h, 24 kHz. Its licence is not stated. Nothing from it is published here.
- Kept only the ship's computer. Interviews, narration, the Interactive Technical Manual readings and kiosk lines were dropped by source. The rest were scored by a classifier on speaker embeddings (ECAPA-TDNN, Resemblyzer) and pitch and spectral features, trained on on-screen computer lines against interview clips (out-of-fold AUC 1.000), and kept above a margin. Clips whose words could not be settled (15) or that stayed noisy (1) were dropped too: 580 of 1,005 in all.
- Cleaning only where it measurably helped (DNSMOS): 294 clips through Demucs, 2 also through DeepFilterNet, 129 untouched. Silence trimmed, -20 LUFS, 24 kHz mono. Re-transcribed with Whisper large-v3; 49 transcripts were corrected by hand where it and the dataset disagreed. One near-duplicate removed.
- Result: 425 clips, 21.1 min. One clean 2.8 s clip serves as the reference that sets the speaker.
- `training/prepare_clips.py` does the last step for your own clips: 24 kHz mono, 0.4 s of silence after each clip (so the voice learns to stop), and the JSONL Qwen's tools read.

Training
- Qwen's single-speaker fine-tuning script, `finetuning/sft_12hz.py` at commit `022e286b98`, with fixes, as `training/sft_12hz_patched.py`. Base model `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at revision `5d83992436`; tokenizer `Qwen/Qwen3-TTS-Tokenizer-12Hz` at `7dd38ad4e9`.
- The first run, with the script as published, babbled: every test line ran to the 48 s cap. The script shifts the labels twice (Hugging Face's loss already shifts them) and feeds the second-stage predictor the hidden state of the frame it is predicting, so it learns to copy its answer (upstream issues #371 and #337, PR #278). Both are fixed and marked `PATCH` in the file, with three more: the text projection the model applies at inference (PR #336), fp32 master weights (bf16 AdamW rounded small updates away), and an atomic checkpoint write.
- Batch 2 with gradient accumulation 4, learning rate 5e-6, 3 epochs of 213 steps; mean loss 2.13, 1.68, 1.41. Each epoch took 100 s.
- `training/run_pod.sh` runs everything unattended and marks each saved epoch and the end (`SAVED`, `DONE`, `FAILED`); `training/copyback.sh` pulls checkpoints as they land; `training/guard.sh` stops the pod if nobody is watching. `training/sanity.py` renders two lines per epoch, which is how the babbling run was caught in minutes.
- Epochs 0, 1 and 2 measured alike (first audio 0.41 s, Whisper 0.95-0.97). Epoch 0 was chosen by ear and converted to 6-bit MLX with `training/convert_mlx.sh`.

Hardware and time
- One NVIDIA A40 (48 GB), Runpod, EU-SE-1, PyTorch 2.8.0 with CUDA 12.8. Pod session 41.9 min: setup, the failed run (5 min of training), the fixed run (5 min), checks.
- Clip selection and cleaning ran on our own Linux server, the trials and conversion on an M1 Mac mini: no cost.

Cost

| Item | Use | Cost |
|---|---|---|
| A40, $0.59/h, 41.9 min | both training runs, sanity checks | $0.42 |
| **Total** | | **$0.42** |

The figure is the pod's run time at its hourly price. The pod was deleted afterwards; no storage was kept.

## Performance on an M1

Apple M1, 16 GB, macOS 26, load average 1.9. `tools/bench.py --runs 3`, median of 3 warm runs through `say.py`'s worker, the 6-bit model. Times run from the request to the worker to the first and last audio.

| Line | Speech | First audio | Total | Work per second of speech |
|---|---|---|---|---|
| "Working." | 1.20 s | 0.40 s | 1.04 s | 0.87 s |
| "Unable to comply. The signing key is not available on this machine." | 5.28 s | 0.41 s | 3.64 s | 0.69 s |
| "Warning. Test suite failed. Two examples failing in the build queue." | 5.36 s | 0.41 s | 4.16 s | 0.78 s |
| "Specify parameters. Which branch should be merged into main?" | 4.72 s | 0.41 s | 3.43 s | 0.73 s |
| "Accessing library computer data. Three files modified. All tests passing." | 6.32 s | 0.41 s | 4.65 s | 0.74 s |
| "Project build complete. Three of four features passing." | 4.08 s | 0.41 s | 3.19 s | 0.78 s |
| Four-sentence status report (26 words) | 11.36 s | 0.43 s | 7.84 s | 0.69 s |
| **All** | **38.3 s** | **0.41 s median** | **28.0 s** | **0.73 s** |

- Cold start (a new worker: load, warm-up, first request): 4.1 s to first audio with the model files in the disk cache, 8.7 s right after they were written. A cold load on an M1 laptop has taken 20 s.
- Worker memory: 1.9 GiB resident.
- In another run, with macOS's media-analysis service using most of a core, the voice slowed to 1.06 s of work per second of speech (four lines over 1.17 s). A stream that slow falls behind and stutters; the voice needs most of an idle M1.

Intelligibility: Whisper small.en (`tools/readback.py`) transcribed each line; the score is word-sequence similarity to the input, ignoring case and punctuation (1.00 = exact).

| Line | Whisper heard | Score |
|---|---|---|
| "Working." | "Working" | 1.00 |
| "Unable to comply…" | "Unable to comply, the signing key is not available on this machine." | 1.00 |
| "Warning. Test suite failed…" | "Warning, test suite failed. Two examples failing in the build queue." | 1.00 |
| "Specify parameters…" | "Specify parameters which branch should be merged into main." | 1.00 |
| "Accessing library computer data…" | "Accessing library computer data. Three files modified, all tests passing." | 1.00 |
| "Project build complete…" | "Project Build Complete Three of four features passing." | 1.00 |
| Status report | exact | 1.00 |

Mean 1.00; 7 of 7 exact. Noise floor (`tools/noise.py`, the quietest 10% of 20 ms frames, before VOICE_FX): -111 dBFS mean, against speech at -17 dBFS; the pauses are near digital silence.

Not measured: older or newer Macs, Linux, listener ratings.

## Licence

- Code and documentation: MIT, see [LICENSE](LICENSE).
- `training/sft_12hz_patched.py` is modified from Qwen's Apache-2.0 code and stays Apache-2.0: see its header and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
- Voice weights (the v2.0.0 release archives): not MIT. See [VOICE-NOTICE.md](VOICE-NOTICE.md).
- Dependencies, all permissive: mlx-audio and mlx (MIT), transformers, huggingface_hub, accelerate and qwen-tts (Apache-2.0), PyTorch, NumPy, SciPy and soundfile (BSD-3), mlx-whisper (MIT). Versions are pinned in the `requirements-*.txt` files. `say.py` imports none of them; the worker imports mlx-audio.
- No sound effects or third-party audio are included.
