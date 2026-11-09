#!/usr/bin/env python3
"""Speak text in the ship's-computer voice, streamed, or render it to a WAV file.

    say.py "Diagnostic complete."              # speak (stdin works too)
    say.py --out reply.wav "Working."          # render to a WAV file instead
    say.py --say-as project=PRAH-ject "..."    # a pronunciation hint for this request
    say.py --print-text "..."                  # print the text the voice will read

The voice is a Qwen3-TTS fine-tune, run by mlx-audio on Apple silicon. It runs in
a worker: a separate process, in its own venv, that loads the model on first use,
takes requests over a Unix socket one at a time and exits after
SHIPS_COMPUTER_QWEN_IDLE seconds without one. The worker streams 16-bit mono PCM
as it is generated; this file pipes it through ffmpeg (VOICE_FX) to the speakers
or into the WAV file. This file is standard library only: it never imports the
model code, so it runs with any python3 (3.9+).

Pronunciation: pronounce.tsv next to this file, then each pronounce-*.tsv next
to it (sorted by name; local tables, not in git), then each table named in
SHIPS_COMPUTER_PRONOUNCE, then each --say-as hint; later entries win. The format
and matching rules are at the top of pronounce.tsv.

Fallback: if the model folder, its venv or ffmpeg is missing, or the worker fails
before the first audio, say.py says why once on stderr and speaks with a macOS
system voice (`say`) instead, with the same text, hints and --out. On other
systems it exits 1 with that message.

Environment:
  SHIPS_COMPUTER_QWEN_MODEL    model folder (default: ships-computer-v1-mlx-6bit/
                               next to this file)
  SHIPS_COMPUTER_QWEN_PYTHON   python of the venv holding mlx-audio (default:
                               .qwen/bin/python next to this file)
  SHIPS_COMPUTER_QWEN_RUN      folder for the worker's socket, state and JSONL logs
                               (default: $TMPDIR/ships-computer-voice-<uid>)
  SHIPS_COMPUTER_QWEN_IDLE     seconds the worker waits for a request before
                               exiting (default: 600)
  SHIPS_COMPUTER_PRONOUNCE     extra pronunciation tables, colon-separated
  SHIPS_COMPUTER_VOICE_FX      ffmpeg audio filter chain; "" disables processing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILTIN_TABLE = HERE / "pronounce.tsv"

# Moderate compression (threshold -20 dB, 4:1, 5 ms attack, 80 ms release), a
# fixed gain and a limiter. Fixed gain rather than loudnorm, which cannot measure
# a live stream and misjudges one-word lines.
DEFAULT_FX = ("acompressor=threshold=-20dB:ratio=4:attack=5:release=80:makeup=6dB,"
              "volume=5dB,alimiter=limit=0.97:level=disabled")

# macOS voices for the fallback, most preferred first; the first one installed is
# used, else the system default voice.
SYSTEM_VOICES = (
    "Allison (Enhanced)", "Samantha (Enhanced)", "Ava (Enhanced)",
    "Allison", "Samantha", "Kathy",
)

TEXT_LIMIT = 400        # characters per generation call; longer text is split at sentences
START_TIMEOUT = 180.0   # seconds to wait for a new worker to load the model


def model_dir() -> Path:
    return Path(os.environ.get("SHIPS_COMPUTER_QWEN_MODEL") or HERE / "ships-computer-v1-mlx-6bit")


def qwen_python() -> Path:
    return Path(os.environ.get("SHIPS_COMPUTER_QWEN_PYTHON") or HERE / ".qwen" / "bin" / "python")


def run_dir() -> Path:
    explicit = os.environ.get("SHIPS_COMPUTER_QWEN_RUN")
    if explicit:
        return Path(explicit)
    return Path(tempfile.gettempdir()) / f"ships-computer-voice-{os.getuid()}"


def voice_fx() -> str:
    return os.environ.get("SHIPS_COMPUTER_VOICE_FX", DEFAULT_FX)


# ---------------------------------------------------------------------------
# Pronunciation: pure functions, no I/O.

def _key(word: str) -> str:
    return " ".join(word.split()).casefold()


def parse_table(text: str, source: str = "<table>") -> tuple[dict, list]:
    """Parse a pronunciation table. Returns ({key: respelling}, [warnings]).

    One entry per line, `word<TAB>respelling`. Blank lines and lines starting
    with # are ignored. A later line for the same word replaces an earlier one.
    """
    entries, warnings = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 2 or not cols[0].strip() or not cols[1].strip():
            warnings.append(f"{source}:{n}: expected <word><TAB><respelling>; line skipped")
            continue
        entries[_key(cols[0])] = cols[1].strip()
    return entries, warnings


def parse_hint(hint: str) -> tuple[str, str]:
    """`word=RESPELLING` -> (key, respelling). Raises ValueError if malformed."""
    word, sep, spelling = hint.partition("=")
    if not sep or not word.strip() or not spelling.strip():
        raise ValueError(f"--say-as wants WORD=RESPELLING, got {hint!r}")
    return _key(word), spelling.strip()


def merge_tables(*tables: dict) -> dict:
    """Later tables win over earlier ones."""
    merged = {}
    for t in tables:
        merged.update(t)
    return merged


def apply_pronunciations(text: str, table: dict) -> str:
    """Replace each whole-word, case-insensitive match of a table key.

    Longer keys are tried first, so a two-word entry beats a one-word entry
    inside it. Replacement is one pass: a respelling is never itself matched.
    The respelling is used as written, except that its first letter is
    capitalised when the matched word's first letter is a capital.
    """
    if not table:
        return text
    keys = sorted(table, key=lambda k: (-len(k), k))
    alternatives = "|".join(r"\s+".join(re.escape(part) for part in k.split(" ")) for k in keys)
    rx = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)

    def sub(m: re.Match) -> str:
        spelled = table[_key(m.group(0))]
        if m.group(0)[0].isupper() and spelled[0].islower():
            spelled = spelled[0].upper() + spelled[1:]
        return spelled

    return rx.sub(sub, text)


def split_text(text: str, limit: int = TEXT_LIMIT) -> list:
    """Split text longer than `limit` into chunks of whole sentences.

    A single sentence longer than `limit` stays whole.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text] if text else []
    chunks, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + 1 + len(sentence) > limit:
            chunks.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        chunks.append(cur)
    return chunks


def token_cap(text: str) -> int:
    """Most codec frames (12.5 per second of speech) to generate for `text`.

    About three times the length speech needs, so a runaway generation stops
    within seconds rather than minutes.
    """
    return min(2048, 50 + 3 * len(text))


def parse_say_voices(listing: str) -> list:
    """Voice names from the output of `say -v '?'`."""
    names = []
    for line in listing.splitlines():
        m = re.match(r"^(.+?)\s+[a-z]{2,3}_[A-Za-z0-9]+\s+#", line)
        if m:
            names.append(m.group(1).strip())
    return names


def pick_system_voice(installed: list) -> str | None:
    """The first SYSTEM_VOICES entry installed, as `say -v` names it.

    macOS may list a voice with its language, "Samantha (English (US))"; that
    form counts, US English first.
    """
    for name in SYSTEM_VOICES:
        for candidate in (name, f"{name} (English (US))"):
            if candidate in installed:
                return candidate
        for candidate in installed:
            if candidate.startswith(f"{name} (English ("):
                return candidate
    return None


def load_tables(hints: list, env_paths: str | None = None,
                builtin: Path = BUILTIN_TABLE) -> tuple[dict, list]:
    """The merged table for one request, and warnings to print.

    Order: the built-in table, each pronounce-*.tsv beside it (sorted by name),
    each table in `env_paths` (colon-separated), then the hints.
    """
    tables, warnings = [], []
    for f in [builtin, *sorted(builtin.parent.glob("pronounce-*.tsv"))]:
        if f.is_file():
            t, w = parse_table(f.read_text(encoding="utf-8"), str(f))
            tables.append(t)
            warnings += w
    for p in filter(None, (env_paths or "").split(":")):
        try:
            t, w = parse_table(Path(p).read_text(encoding="utf-8"), p)
        except OSError as e:
            warnings.append(f"{p}: pronunciation table not read ({e.strerror}); skipped")
            continue
        tables.append(t)
        warnings += w
    tables.append(dict(parse_hint(h) for h in hints))
    return merge_tables(*tables), warnings


# ---------------------------------------------------------------------------
# Wire format between say.py and the worker.
# Request: one JSON line. Reply: frames of 1 type byte + 4-byte big-endian
# length + payload. H: JSON header {"sample_rate": n}, sent with the first
# audio. A: s16le mono PCM. E: JSON {"error": "..."}. Z: end, length 0.

def frame(kind: bytes, payload: bytes = b"") -> bytes:
    return kind + struct.pack(">I", len(payload)) + payload


def read_frame(f) -> tuple:
    head = f.read(5)
    if len(head) < 5:
        return None, b""
    (n,) = struct.unpack(">I", head[1:])
    payload = f.read(n)
    if len(payload) < n:
        return None, b""
    return head[:1], payload


class Unavailable(Exception):
    """The fine-tuned voice cannot speak this request; use the fallback."""


def _worker_paths(model: Path, python: Path) -> tuple:
    wid = hashlib.sha1(f"{model.resolve()}\0{python}".encode()).hexdigest()[:10]
    rd = run_dir()
    return rd, rd / f"worker-{wid}.sock", rd / f"worker-{wid}.err"


def _connect(path: Path) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(str(path))
    except OSError:
        s.close()
        raise
    return s


def _last_line(path: Path) -> str:
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    return lines[-1][:300] if lines else ""


def connect_worker(model: Path, python: Path) -> socket.socket:
    """A connection to a ready worker for this model, starting one if needed."""
    if not (model / "config.json").is_file():
        raise Unavailable(f"voice model not found at {model}")
    if not os.access(python, os.X_OK):
        raise Unavailable(f"voice venv not found ({python})")
    rd, sock, err = _worker_paths(model, python)
    rd.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        return _connect(sock)
    except OSError:
        pass
    import fcntl
    with open(rd / "spawn.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _connect(sock)
        except OSError:
            try:
                sock.unlink()
            except FileNotFoundError:
                pass
        with open(err, "wb") as errf:
            proc = subprocess.Popen(
                [str(python), str(Path(__file__).resolve()), "--worker",
                 "--worker-model", str(model), "--worker-socket", str(sock)],
                stdin=subprocess.DEVNULL, stdout=errf, stderr=errf,
                start_new_session=True, cwd=str(rd))
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            try:
                return _connect(sock)
            except OSError:
                pass
            if proc.poll() is not None:
                raise Unavailable(f"voice worker exited while loading (code {proc.returncode}): "
                                  f"{_last_line(err)} [{err}]")
            time.sleep(0.05)
        raise Unavailable(f"voice worker not ready after {START_TIMEOUT:.0f} s [{err}]")


def open_stream(text: str, model: Path | None = None, python: Path | None = None) -> tuple:
    """Ask the worker for `text`. Returns (sample_rate, iterator of PCM bytes).

    Raises Unavailable if no audio arrives. The iterator raises RuntimeError if
    the worker stops after audio has started.
    """
    model, python = model or model_dir(), python or qwen_python()
    for attempt in (1, 2):
        s = connect_worker(model, python)
        f = s.makefile("rb")
        try:
            s.sendall(json.dumps({"text": text}).encode() + b"\n")
            kind, payload = read_frame(f)
        except OSError:
            kind, payload = None, b""
        if kind is None and attempt == 1:
            s.close()   # a worker that was exiting for idleness; start a fresh one
            continue
        if kind == b"E":
            s.close()
            raise Unavailable(f"voice worker failed: {json.loads(payload)['error']}")
        if kind != b"H":
            s.close()
            raise Unavailable("voice worker closed the connection before any audio")
        rate = int(json.loads(payload)["sample_rate"])
        return rate, _chunks(s, f)
    raise Unavailable("voice worker unavailable")


def _chunks(s: socket.socket, f):
    try:
        while True:
            kind, payload = read_frame(f)
            if kind == b"A":
                yield payload
            elif kind == b"Z":
                return
            elif kind == b"E":
                raise RuntimeError(f"voice worker failed mid-speech: {json.loads(payload)['error']}")
            else:
                raise RuntimeError("voice worker stopped mid-speech")
    finally:
        s.close()


def stop_worker() -> bool:
    """Ask this model's worker to exit. True if one was running."""
    _, sock, _ = _worker_paths(model_dir(), qwen_python())
    try:
        s = _connect(sock)
    except OSError:
        return False
    with s:
        s.sendall(b'{"cmd": "stop"}\n')
        s.recv(16)
    return True


# ---------------------------------------------------------------------------
# Playback and rendering.

def _raw_in(rate: int) -> list:
    return ["-f", "s16le", "-sample_rate", str(rate), "-ch_layout", "mono", "-i", "-"]


def _fx() -> list:
    fx = voice_fx()
    return ["-af", fx] if fx else []


def player_pipeline(rate: int) -> list:
    """Commands that read raw s16le mono on stdin, process it and play it."""
    if sys.platform == "darwin":
        # ffmpeg straight to Core Audio. Not ffplay: started before the first
        # audio arrives, ffplay clips the first syllable.
        return [["ffmpeg", "-loglevel", "quiet", *_raw_in(rate), *_fx(), "-f", "audiotoolbox", "-"]]
    play = ["paplay", "--raw", f"--rate={rate}", "--channels=1", "--format=s16le"]
    if not voice_fx():
        return [play]
    return [["ffmpeg", "-loglevel", "quiet", *_raw_in(rate), *_fx(), "-f", "s16le", "-"], play]


def render_pipeline(rate: int, out_wav: str) -> list:
    return [["ffmpeg", "-loglevel", "quiet", "-y", *_raw_in(rate), *_fx(), out_wav]]


def pump(chunks, cmds: list) -> bool:
    """Feed PCM chunks through a chain of commands. True if every stage exits 0."""
    procs = []
    upstream = subprocess.PIPE
    for i, cmd in enumerate(cmds):
        last = i == len(cmds) - 1
        p = subprocess.Popen(cmd, stdin=upstream,
                             stdout=subprocess.DEVNULL if last else subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        if procs:
            procs[-1].stdout.close()   # the new stage owns the read end now
        procs.append(p)
        upstream = p.stdout
    ok = True
    try:
        for chunk in chunks:
            procs[0].stdin.write(chunk)
            procs[0].stdin.flush()
    except RuntimeError as e:
        sys.stderr.write(f"say.py: {e}\n")
        ok = False
    except BrokenPipeError:
        ok = False
    finally:
        try:
            procs[0].stdin.close()
        except BrokenPipeError:
            pass
        for p in procs:
            p.wait()
    return ok and all(p.returncode == 0 for p in procs)


def system_say(text: str, out: str | None, reason: str) -> bool:
    """The fallback: a macOS system voice. Says why, once, on stderr."""
    if sys.platform != "darwin" or not shutil.which("say"):
        sys.stderr.write(f"say.py: {reason}; no fallback voice on this system\n")
        return False
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voice = pick_system_voice(parse_say_voices(listing))
    sys.stderr.write(f"say.py: {reason}; speaking with the macOS voice "
                     f"{voice or '(system default)'} instead\n")
    cmd = ["say", *(["-v", voice] if voice else []), "-f", "-"]
    if out:
        cmd += ["-o", out, "--file-format=WAVE", "--data-format=LEI16@24000"]
    return subprocess.run(cmd, input=text.encode()).returncode == 0


def speak(text: str, out: str | None = None) -> bool:
    """Speak `text` (or render it to `out`), falling back to the system voice."""
    if not shutil.which("ffmpeg"):
        return system_say(text, out, "ffmpeg not found")
    try:
        rate, chunks = open_stream(text)
    except Unavailable as e:
        return system_say(text, out, str(e))
    return pump(chunks, render_pipeline(rate, out) if out else player_pipeline(rate))


# ---------------------------------------------------------------------------
# The worker: runs in the venv's python, imports the model code.

def _speaker(model: Path) -> str:
    """The fine-tune's one speaker name, from its config ("computer" if unclear)."""
    try:
        cfg = json.loads((model / "config.json").read_text())
        names = list(cfg.get("talker_config", {}).get("spk_id", {}))
    except (OSError, ValueError):
        names = []
    return names[0] if len(names) == 1 else "computer"


class _Log:
    """One JSONL file per worker session in <run>/logs/; keeps the newest 20."""

    def __init__(self, rd: Path):
        d = rd / "logs"
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        for old in sorted(d.glob("worker-*.jsonl"))[:-19]:
            old.unlink()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.path = d / f"worker-{stamp}-{os.getpid()}.jsonl"

    def __call__(self, event: str, **fields) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def worker_main(model: Path, sock_path: Path) -> int:
    rd = sock_path.parent
    log = _Log(rd)
    idle = float(os.environ.get("SHIPS_COMPUTER_QWEN_IDLE", "600"))
    log("start", pid=os.getpid(), model=str(model), python=sys.executable, idle_s=idle)
    t0 = time.monotonic()
    try:
        import numpy as np
        from mlx_audio.tts.utils import load_model
        tts = load_model(str(model))
        speaker = _speaker(model)
        rate = int(tts.sample_rate)
        loaded = time.monotonic() - t0

        def generate(text):
            for r in tts.generate(text=text, voice=speaker, lang_code="auto", stream=True,
                                  streaming_interval=0.5, max_tokens=token_cap(text)):
                a = np.asarray(r.audio, dtype=np.float32)
                if a.size:
                    yield (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

        # Warm-up: the first generation after loading compiles kernels and is
        # several times slower to first audio; pay that before taking requests.
        for _ in generate("Ready."):
            pass
        log("loaded", load_s=round(loaded, 2), warmup_s=round(time.monotonic() - t0 - loaded, 2),
            speaker=speaker, sample_rate=rate)
    except Exception as e:  # noqa: BLE001 - reported to the log and stderr, then exit
        log("load_failed", error=f"{type(e).__name__}: {e}")
        print(f"load failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    tmp = sock_path.with_name(f"{sock_path.name}.{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    srv.bind(str(tmp))
    os.chmod(tmp, 0o600)
    srv.listen(16)
    os.replace(tmp, sock_path)   # ready only once loaded: a connect means it can speak
    ino = os.stat(sock_path).st_ino
    state = rd / f"{sock_path.stem}.json"
    state.write_text(json.dumps({"pid": os.getpid(), "model": str(model), "log": str(log.path)}))
    srv.settimeout(idle)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                log("idle_exit", idle_s=idle)
                return 0
            with conn:
                conn.settimeout(None)
                if not _serve(conn, generate, rate, log):
                    log("stop")
                    return 0
    finally:
        try:
            if os.stat(sock_path).st_ino == ino:
                sock_path.unlink()
                state.unlink()
        except OSError:
            pass


def _serve(conn: socket.socket, generate, rate: int, log) -> bool:
    """Answer one request. False if it asked the worker to stop."""
    f = conn.makefile("rb")
    try:
        req = json.loads(f.readline(1 << 20) or b"{}")
    except ValueError:
        req = {}
    if req.get("cmd") == "stop":
        conn.sendall(frame(b"Z"))
        return False
    text = str(req.get("text", "")).strip()
    t0, first, samples, sent = time.monotonic(), None, 0, False
    try:
        if not text:
            raise ValueError("empty text")
        for part in split_text(text):
            for pcm in generate(part):
                if not sent:
                    first = time.monotonic() - t0
                    conn.sendall(frame(b"H", json.dumps({"sample_rate": rate}).encode()))
                    sent = True
                conn.sendall(frame(b"A", pcm))
                samples += len(pcm) // 2
        if not sent:
            raise RuntimeError("the model produced no audio")
        conn.sendall(frame(b"Z"))
    except (BrokenPipeError, ConnectionResetError):
        log("client_gone", chars=len(text), audio_s=round(samples / rate, 2))
        return True
    except Exception as e:  # noqa: BLE001 - one bad request must not kill the worker
        log("request_failed", text=text, error=f"{type(e).__name__}: {e}")
        try:
            conn.sendall(frame(b"E", json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()))
        except OSError:
            pass
        return True
    total = time.monotonic() - t0
    audio = samples / rate
    log("request", text=text, chars=len(text), first_audio_s=round(first, 3),
        total_s=round(total, 3), audio_s=round(audio, 2), rtf=round(total / audio, 3) if audio else None)
    return True


# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="*", help="text to speak (default: stdin)")
    ap.add_argument("--out", help="render to this WAV file instead of speaking")
    ap.add_argument("--say-as", action="append", default=[], metavar="WORD=RESPELLING",
                    help="pronounce WORD as RESPELLING for this request (repeatable)")
    ap.add_argument("--print-text", action="store_true",
                    help="print the text after pronunciation tables and hints, and exit")
    ap.add_argument("--stop-worker", action="store_true", help="stop the running worker, if any")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-model", help=argparse.SUPPRESS)
    ap.add_argument("--worker-socket", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        return worker_main(Path(a.worker_model), Path(a.worker_socket))
    if a.stop_worker:
        print("stopped" if stop_worker() else "no worker running")
        return 0
    try:
        table, warnings = load_tables(a.say_as, os.environ.get("SHIPS_COMPUTER_PRONOUNCE"))
    except ValueError as e:
        ap.error(str(e))
    for w in warnings:
        sys.stderr.write(f"say.py: {w}\n")
    text = " ".join(a.text).strip() or ("" if sys.stdin.isatty() else sys.stdin.read().strip())
    if not text:
        return 0
    spoken = apply_pronunciations(text, table)
    if a.print_text:
        print(spoken)
        return 0
    return 0 if speak(spoken, a.out) else 1


if __name__ == "__main__":
    sys.exit(main())
