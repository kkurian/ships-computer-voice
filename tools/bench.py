#!/usr/bin/env python3
"""Measure the voice on this machine, through say.py's worker, as say.py uses it.

    python3 tools/bench.py [--runs 3] [--out-dir bench-out]

Stops any running worker first, so the first request measures a cold start
(worker start, model load, warm-up, first audio). Then each line runs --runs
times on the warm worker; the table shows medians. The first warm render of
each line is written, without VOICE_FX, to <out-dir>/lineN.wav for
tools/readback.py and tools/noise.py. Standard library only; plays nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import say  # noqa: E402

LINES = [
    "Working.",
    "Unable to comply. The signing key is not available on this machine.",
    "Warning. Test suite failed. Two examples failing in the build queue.",
    "Specify parameters. Which branch should be merged into main?",
    "Accessing library computer data. Three files modified. All tests passing.",
    "Project build complete. Three of four features passing.",
    "Status report. The build queue processed four changes. Three passed their tests and "
    "were merged. One failed a regression check and was returned to the author.",
]


def once(text: str):
    """One request: (seconds to first audio, total seconds, PCM bytes, sample rate)."""
    t0 = time.monotonic()
    rate, chunks = say.open_stream(text)
    first, buf = time.monotonic() - t0, bytearray()
    for c in chunks:
        buf += c
    return first, time.monotonic() - t0, bytes(buf), rate


def worker_rss_gib() -> float:
    _, sock, _ = say._worker_paths(say.model_dir(), say.qwen_python())
    try:
        pid = json.loads(sock.with_suffix(".json").read_text())["pid"]
        kib = int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                                 text=True).stdout.strip())
    except (OSError, ValueError, KeyError):
        return float("nan")
    return kib / 2**20


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", default="bench-out")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table, _ = say.load_tables([], os.environ.get("SHIPS_COMPUTER_PRONOUNCE"))
    spoken = [say.apply_pronunciations(t, table) for t in LINES]

    say.stop_worker()
    time.sleep(1)
    t0 = time.monotonic()
    first, total, _, _ = once(spoken[0])
    print(f"model {say.model_dir()}; load average {os.getloadavg()[0]:.1f}; {a.runs} warm runs per line")
    print(f"cold start (new worker: start, load, warm-up, first request): first audio {first:.2f} s, "
          f"done {time.monotonic() - t0:.2f} s\n")
    print("| Line | Speech | First audio | Total | Work per second of speech |")
    print("|---|---|---|---|---|")
    firsts, tot_wall, tot_audio = [], 0.0, 0.0
    for i, text in enumerate(spoken, 1):
        runs = [once(text) for _ in range(a.runs)]
        f, w = med([r[0] for r in runs]), med([r[1] for r in runs])
        pcm, rate = runs[0][2], runs[0][3]
        audio = len(pcm) / 2 / rate
        with wave.open(str(out / f"line{i}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        firsts.append(f)
        tot_wall += w
        tot_audio += audio
        print(f"| {i} | {audio:.2f} s | {f:.2f} s | {w:.2f} s | {w / audio:.2f} s |")
    print(f"| **All** | **{tot_audio:.1f} s** | **{med(firsts):.2f} s median** | **{tot_wall:.1f} s** | "
          f"**{tot_wall / tot_audio:.2f} s** |")
    print(f"\nworker resident memory: {worker_rss_gib():.2f} GiB; WAVs in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
