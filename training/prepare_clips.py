#!/usr/bin/env python3
"""Turn your own transcribed clips into Qwen3-TTS fine-tuning input.

    python3 prepare_clips.py DATA_DIR OUT_DIR [--pad 0.4]

DATA_DIR holds:
  clips/<id>.wav    one line of speech each (any sample rate; stereo is mixed down)
  metadata.csv      one `<id>|<transcript>` per line, UTF-8
  ref.wav           one clean, typical clip of the voice (2-5 s): its speaker
                    embedding becomes the fine-tune's speaker

Writes OUT_DIR/wav/<id>.wav and OUT_DIR/ref.wav (24 kHz mono 16-bit; each clip
followed by --pad seconds of silence, so the model learns to end cleanly) and
OUT_DIR/train_raw.jsonl, the input of Qwen's finetuning/prepare_data.py.

Needs numpy, scipy, soundfile (all in requirements-training.txt).
"""
import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 24000


def load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != RATE:
        g = gcd(sr, RATE)
        a = resample_poly(a, RATE // g, sr // g).astype(np.float32)
    peak = float(np.abs(a).max()) if a.size else 0.0
    return a / peak if peak > 1.0 else a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--pad", type=float, default=0.4, help="seconds of silence after each clip")
    a = ap.parse_args()
    (a.out_dir / "wav").mkdir(parents=True, exist_ok=True)
    ref = a.out_dir / "ref.wav"
    sf.write(str(ref), load(a.data_dir / "ref.wav"), RATE, subtype="PCM_16")
    pad = np.zeros(int(a.pad * RATE), np.float32)
    n, total = 0, 0.0
    with open(a.out_dir / "train_raw.jsonl", "w", encoding="utf-8") as out:
        for line in (a.data_dir / "metadata.csv").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, sep, text = line.partition("|")
            if not sep or not text.strip():
                sys.exit(f"metadata.csv: expected <id>|<transcript>, got {line!r}")
            audio = np.concatenate([load(a.data_dir / "clips" / f"{cid}.wav"), pad])
            dest = a.out_dir / "wav" / f"{cid}.wav"
            sf.write(str(dest), audio, RATE, subtype="PCM_16")
            out.write(json.dumps({"audio": str(dest.resolve()), "text": " ".join(text.split()),
                                  "ref_audio": str(ref.resolve())}, ensure_ascii=False) + "\n")
            n += 1
            total += len(audio) / RATE
    print(f"prepared {n} clips, {total / 60:.1f} min including padding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
