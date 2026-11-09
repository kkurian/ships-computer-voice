#!/usr/bin/env python3
"""Rough noise floor of WAV files: quiet frames against loud frames.

    .tools/bin/python tools/noise.py bench-out/*.wav

Splits each file into 20 ms frames and takes their RMS level in dBFS. Floor is
the 10th percentile (the pauses), speech the 90th, and the difference a crude
signal-to-noise figure. It measures only the pauses: noise under speech and
its character need ears.
"""
import os
import sys

import numpy as np
import soundfile as sf


def levels(path: str) -> tuple:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    n = int(sr * 0.02)
    k = len(a) // n
    frames = a[:k * n].reshape(k, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12))
    floor, speech = np.percentile(db, 10), np.percentile(db, 90)
    return floor, speech, speech - floor


def main() -> int:
    print("file\tfloor dBFS\tspeech dBFS\tdifference dB")
    rows = []
    for f in sorted(sys.argv[1:]):
        fl, sp, d = levels(f)
        rows.append((fl, sp, d))
        print(f"{os.path.basename(f)}\t{fl:.1f}\t{sp:.1f}\t{d:.1f}")
    if rows:
        print("mean\t" + "\t".join(f"{np.mean([r[i] for r in rows]):.1f}" for i in range(3)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
