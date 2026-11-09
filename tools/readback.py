#!/usr/bin/env python3
"""Intelligibility: Whisper reads back each bench render; score each against its line.

    .tools/bin/python tools/readback.py [bench-out]

Reads <dir>/line1.wav ... line7.wav from tools/bench.py. The score is
word-sequence similarity between the bench line and what Whisper heard
(1.00 = exact; punctuation and case ignored). Uses mlx-whisper with
mlx-community/whisper-small.en-mlx (downloaded on first use); audio is
resampled to 16 kHz first.
"""
import difflib
import re
import sys
from math import gcd
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import LINES  # noqa: E402

MODEL = "mlx-community/whisper-small.en-mlx"


def words(s: str) -> list:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).split()


def heard(path: Path) -> str:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return mlx_whisper.transcribe(a, path_or_hf_repo=MODEL)["text"].strip()


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "bench-out")
    scores = []
    print("| Line | Whisper heard | Score |\n|---|---|---|")
    for i, line in enumerate(LINES, 1):
        got = heard(d / f"line{i}.wav")
        s = difflib.SequenceMatcher(None, words(line), words(got)).ratio()
        scores.append(s)
        print(f"| {i} | {got} | {s:.2f} |")
    print(f"\nmean {sum(scores) / len(scores):.2f}; exact {sum(s == 1.0 for s in scores)} of {len(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
