#!/usr/bin/env python3
"""Speak text in the ship's-computer voice, streamed, or render it to a WAV file.

    say.py "Diagnostic complete."              # speak (stdin works too)
    say.py --out reply.wav "Working."          # render to a WAV file instead
    say.py --say-as project=PRAH-ject "..."    # a pronunciation hint for this request
    say.py --print-text "..."                  # print the text the voice will read
    say.py --take 1 --out reply.wav "Working." # a different take of the same line

The voice is a Qwen3-TTS fine-tune, run by mlx-audio on Apple silicon. It runs in
a worker: a separate process, in its own venv, that loads the model on first use,
takes requests over a Unix socket one at a time and exits after
SHIPS_COMPUTER_QWEN_IDLE seconds without one. The worker streams 16-bit mono PCM
as it is generated; this file pipes it through ffmpeg (VOICE_FX) to the speakers
or into the WAV file. This file is standard library only: it never imports the
model code, so it runs with any python3 (3.9+).

Repeatability: each chunk of text is sampled with a seed derived from its text,
the model's hash, the sampling settings and --take, so the same text, model and
settings give the same audio. Sampling is cooler than the model's defaults
(--temperature, --top-p, --top-k). Each generated chunk is kept as a WAV in a
render cache (.cache/ next to this file; SHIPS_COMPUTER_CACHE), so a line that
did not change is replayed, not regenerated. With --out, each chunk is also
brought to the same speech level before it is joined to the next.

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
                               (default: /tmp/ships-computer-voice-<uid>)
  SHIPS_COMPUTER_QWEN_IDLE     seconds the worker waits for a request before
                               exiting (default: 600)
  SHIPS_COMPUTER_PRONOUNCE     extra pronunciation tables, colon-separated
  SHIPS_COMPUTER_VOICE_FX      ffmpeg audio filter chain; "" disables processing
  SHIPS_COMPUTER_CACHE         render cache folder (default: .cache/ next to this
                               file); "" disables the cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import wave
from array import array
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

# Sampling. mlx-audio's Qwen3-TTS uses these for both the talker (the first
# codebook) and the code predictor (the other 15); the model's own defaults are
# temperature 0.9, top_p 1.0, top_k 50. Cooler sampling narrows how far one take
# can stray from another; the seed (chunk_seed) makes a take repeatable.
DEFAULT_TEMPERATURE = 0.65
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
LANG_CODE = "auto"          # no language token, as training/sanity.py runs the voice
STREAM_INTERVAL = 0.5       # seconds of audio per streamed piece
WIRE_VERSION = 2            # bump when the request or the rendering changes
SPEECH_LEVEL_DB = -19.5     # --out: each chunk's gated RMS level, dBFS (the model's typical level)


def model_dir() -> Path:
    return Path(os.environ.get("SHIPS_COMPUTER_QWEN_MODEL") or HERE / "ships-computer-v1-mlx-6bit")


def qwen_python() -> Path:
    return Path(os.environ.get("SHIPS_COMPUTER_QWEN_PYTHON") or HERE / ".qwen" / "bin" / "python")


def run_dir() -> Path:
    explicit = os.environ.get("SHIPS_COMPUTER_QWEN_RUN")
    if explicit:
        return Path(explicit)
    # /tmp, not $TMPDIR: macOS's $TMPDIR is long, and a Unix socket path must
    # stay under 104 bytes.
    return Path("/tmp") / f"ships-computer-voice-{os.getuid()}"


def ensure_run_dir(rd: Path) -> None:
    """Create the run folder (mode 0700), or refuse one another user could write."""
    rd.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = rd.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise Unavailable(f"run folder {rd} is not private to this user (owner and mode 0700)")


def voice_fx() -> str:
    return os.environ.get("SHIPS_COMPUTER_VOICE_FX", DEFAULT_FX)


def cache_dir() -> Path | None:
    """The render cache folder, or None when SHIPS_COMPUTER_CACHE is ""."""
    explicit = os.environ.get("SHIPS_COMPUTER_CACHE")
    if explicit is not None:
        return Path(explicit) if explicit else None
    return HERE / ".cache"


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


# ---------------------------------------------------------------------------
# Repeatability: seeds, the render cache and chunk levels. Standard library only.

def render_settings(base: dict, temperature: float, top_p: float, top_k: int, take: int) -> dict:
    """Everything besides the text and the model that decides a chunk's audio.

    `base` comes from the worker: speaker, language, streaming interval,
    repetition penalty and the mlx / mlx-audio versions.
    """
    return {**base, "temperature": float(temperature), "top_p": float(top_p),
            "top_k": int(top_k), "take": int(take), "wire": WIRE_VERSION}


def _digest(text: str, model_id: str, settings: dict) -> bytes:
    blob = json.dumps({"text": text, "model": model_id, "settings": settings},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).digest()


def chunk_seed(text: str, model_id: str, settings: dict) -> int:
    """The sampling seed for one chunk: 32 bits of a hash of what decides its audio."""
    return int.from_bytes(_digest(text, model_id, settings)[:4], "big")


def cache_key(text: str, model_id: str, settings: dict) -> str:
    """The render cache's name for one chunk (the seed is a function of the same inputs)."""
    return _digest(text, model_id, settings).hex()[:40]


def model_id(model: Path, memo: Path | None = None) -> str:
    """SHA-256 over the model folder: every file's relative path and contents.

    Hashing 1.8 GB takes seconds, so the result is kept in `memo` (a JSON file)
    under a fingerprint of each file's path, size and modification time.
    """
    files = sorted(f for f in model.resolve().rglob("*") if f.is_file())
    root = model.resolve()
    stamp = hashlib.sha256(json.dumps(
        [[str(f.relative_to(root)), f.stat().st_size, f.stat().st_mtime_ns] for f in files]
    ).encode()).hexdigest()
    if memo is not None:
        try:
            saved = json.loads(memo.read_text())
            if saved.get("stamp") == stamp:
                return saved["model_id"]
        except (OSError, ValueError, KeyError):
            pass
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(root)).encode() + b"\0")
        fh_hash = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                fh_hash.update(block)
        h.update(fh_hash.digest())
    digest = h.hexdigest()
    if memo is not None:
        try:
            memo.parent.mkdir(parents=True, exist_ok=True)
            tmp = memo.with_name(f"{memo.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"stamp": stamp, "model_id": digest}))
            os.replace(tmp, memo)
        except OSError:
            pass
    return digest


def wav_write(path: Path, pcm: bytes, rate: int) -> None:
    """Write s16le mono PCM as a WAV file, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    os.replace(tmp, path)


def wav_read(path: Path) -> tuple:
    """(sample_rate, s16le mono PCM) from a WAV file wav_write wrote."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: not 16-bit mono")
        return w.getframerate(), w.readframes(w.getnframes())


class ChunkCache:
    """One WAV per generated chunk, named by cache_key, in one folder."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, key: str) -> Path:
        return self.root / "chunks" / key[:2] / f"{key}.wav"

    def get(self, key: str, rate: int) -> bytes | None:
        try:
            r, pcm = wav_read(self.path(key))
        except (OSError, ValueError, EOFError, wave.Error):
            return None
        return pcm if r == rate and pcm else None

    def put(self, key: str, pcm: bytes, rate: int) -> None:
        try:
            wav_write(self.path(key), pcm, rate)
        except OSError:
            pass   # a cache that cannot be written only costs a regeneration


def _samples(pcm: bytes) -> array:
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def speech_level_db(pcm: bytes, rate: int, gate_db: float = -50.0) -> float | None:
    """RMS level in dBFS over the 20 ms frames louder than `gate_db`.

    Pauses and breaths below the gate do not pull the level down. None if no
    frame passes the gate.
    """
    a = _samples(pcm)
    n = max(1, rate // 50)
    gate = (10 ** (gate_db / 20) * 32768.0) ** 2
    total, frames = 0.0, 0
    for i in range(0, len(a) - n + 1, n):
        ms = sum(x * x for x in a[i:i + n]) / n
        if ms > gate:
            total += ms
            frames += 1
    if not frames:
        return None
    return 10 * math.log10(total / frames / 32768.0 ** 2)


def normalize_pcm(pcm: bytes, rate: int, target_db: float = SPEECH_LEVEL_DB,
                  max_gain_db: float = 12.0) -> bytes:
    """Scale one chunk so its speech level is `target_db`.

    The gain is held within +/- `max_gain_db` and never lifts the peak past full
    scale, so a chunk is never clipped here.
    """
    level = speech_level_db(pcm, rate)
    a = _samples(pcm)
    if level is None or not a:
        return pcm
    gain_db = max(-max_gain_db, min(max_gain_db, target_db - level))
    peak = max(abs(x) for x in a) or 1
    gain = min(10 ** (gain_db / 20), 32767.0 / peak)
    out = array("h", (max(-32768, min(32767, round(x * gain))) for x in a))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


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
    # The wire version is in the id: after an upgrade, a worker still running the
    # old code is left to exit on its own and a new one is started.
    wid = hashlib.sha1(f"{model.resolve()}\0{python}\0{WIRE_VERSION}".encode()).hexdigest()[:10]
    rd = run_dir()
    return rd, rd / f"worker-{wid}.sock", rd / f"worker-{wid}.err"


SUN_PATH_MAX = 103   # bytes a Unix socket path may have on macOS (104 with the NUL)


def _connect(path: Path) -> socket.socket:
    """Connect to a Unix socket. A path too long for sun_path is reached by a
    short relative name from its folder (the working directory is restored)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if len(os.fsencode(str(path))) <= SUN_PATH_MAX:
            s.connect(str(path))
        else:
            cwd = os.getcwd()
            os.chdir(path.parent)
            try:
                s.connect(path.name)
            finally:
                os.chdir(cwd)
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
    ensure_run_dir(rd)
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


def render_options(temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                   top_k: int = DEFAULT_TOP_K, take: int = 0, normalize: bool = False,
                   cache: Path | None | bool = True) -> dict:
    """The request fields besides the text. cache=True means cache_dir()."""
    root = cache_dir() if cache is True else (cache or None)
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "take": take,
            "normalize": normalize, "cache": str(root) if root else None}


def open_stream(text: str, model: Path | None = None, python: Path | None = None,
                options: dict | None = None) -> tuple:
    """Ask the worker for `text`. Returns (sample_rate, iterator of PCM bytes).

    Raises Unavailable if no audio arrives. The iterator raises RuntimeError if
    the worker stops after audio has started.
    """
    model, python = model or model_dir(), python or qwen_python()
    for attempt in (1, 2):
        s = connect_worker(model, python)
        f = s.makefile("rb")
        try:
            req = {**(options or render_options()), "text": text}
            s.sendall(json.dumps(req).encode() + b"\n")
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


def speak(text: str, out: str | None = None, options: dict | None = None) -> bool:
    """Speak `text` (or render it to `out`), falling back to the system voice."""
    if not shutil.which("ffmpeg"):
        return system_say(text, out, "ffmpeg not found")
    try:
        rate, chunks = open_stream(text, options=options)
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


class Voice:
    """The loaded model, as the worker uses it. Imports mlx-audio: venv only."""

    def __init__(self, model: Path, memo: Path | None = None):
        from importlib.metadata import version

        import mlx.core as mx
        import numpy as np
        from mlx_audio.tts.utils import load_model
        self._mx, self._np = mx, np
        self.tts = load_model(str(model))
        self.speaker = _speaker(model)
        self.rate = int(self.tts.sample_rate)
        self.model_id = model_id(model, memo)
        try:
            cfg = json.loads((model / "generation_config.json").read_text())
        except (OSError, ValueError):
            cfg = {}
        self.base = {"speaker": self.speaker, "lang_code": LANG_CODE,
                     "streaming_interval": STREAM_INTERVAL,
                     "repetition_penalty": float(cfg.get("repetition_penalty", 1.05)),
                     "mlx": version("mlx"), "mlx_audio": version("mlx-audio")}

    def generate(self, text: str, seed: int, settings: dict):
        """s16le PCM pieces for one chunk. The seed is set first, so the same
        text, seed and settings sample the same tokens."""
        self._mx.random.seed(seed)
        np = self._np
        for r in self.tts.generate(
                text=text, voice=settings["speaker"], lang_code=settings["lang_code"],
                stream=True, streaming_interval=settings["streaming_interval"],
                max_tokens=token_cap(text), temperature=settings["temperature"],
                top_p=settings["top_p"], top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"]):
            a = np.asarray(r.audio, dtype=np.float32)
            if a.size:
                yield (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def worker_main(model: Path, sock_path: Path) -> int:
    rd = sock_path.parent
    log = _Log(rd)
    idle = float(os.environ.get("SHIPS_COMPUTER_QWEN_IDLE", "600"))
    log("start", pid=os.getpid(), model=str(model), python=sys.executable, idle_s=idle)
    t0 = time.monotonic()
    try:
        memo = cache_dir()
        voice = Voice(model, memo / "model-id.json" if memo else None)
        loaded = time.monotonic() - t0

        # Warm-up: the first generation after loading compiles kernels and is
        # several times slower to first audio; pay that before taking requests.
        warm = render_settings(voice.base, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, 0)
        for _ in voice.generate("Ready.", 0, warm):
            pass
        log("loaded", load_s=round(loaded, 2), warmup_s=round(time.monotonic() - t0 - loaded, 2),
            speaker=voice.speaker, sample_rate=voice.rate, model_id=voice.model_id,
            runtime={"mlx": voice.base["mlx"], "mlx_audio": voice.base["mlx_audio"]})
    except Exception as e:  # noqa: BLE001 - reported to the log and stderr, then exit
        log("load_failed", error=f"{type(e).__name__}: {e}")
        print(f"load failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    tmp = sock_path.with_name(f"{sock_path.name}.{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    os.chdir(rd)
    srv.bind(tmp.name)   # relative: no sun_path limit on the folder's length
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
                if not _serve(conn, voice, log):
                    log("stop")
                    return 0
    finally:
        try:
            if os.stat(sock_path).st_ino == ino:
                sock_path.unlink()
                state.unlink()
        except OSError:
            pass


def _render_chunk(voice, part: str, settings: dict, cache: ChunkCache | None, buffered: bool):
    """PCM pieces for one chunk, and a record of how they were made.

    A cached chunk is read back; a new one is generated with its seed, streamed
    as it comes unless `buffered`, and stored once complete.
    """
    seed = chunk_seed(part, voice.model_id, settings)
    key = cache_key(part, voice.model_id, settings)
    info = {"seed": seed, "key": key, "cached": False}
    pcm = cache.get(key, voice.rate) if cache else None
    if pcm is not None:
        info["cached"] = True
        if buffered:
            return info, iter([pcm])
        step = int(voice.rate * STREAM_INTERVAL) * 2
        return info, (pcm[i:i + step] for i in range(0, len(pcm), step))

    def pieces():
        got = []
        for piece in voice.generate(part, seed, settings):
            got.append(piece)
            if not buffered:
                yield piece
        whole = b"".join(got)
        if cache and whole:
            cache.put(key, whole, voice.rate)
        if buffered:
            yield whole
    return info, pieces()


def _serve(conn: socket.socket, voice, log) -> bool:
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
    rate = voice.rate
    t0, first, samples, sent, chunks = time.monotonic(), None, 0, False, []
    try:
        if not text:
            raise ValueError("empty text")
        settings = render_settings(voice.base, req.get("temperature", DEFAULT_TEMPERATURE),
                                   req.get("top_p", DEFAULT_TOP_P), req.get("top_k", DEFAULT_TOP_K),
                                   req.get("take", 0))
        normalize = bool(req.get("normalize", False))
        cache = ChunkCache(Path(req["cache"])) if req.get("cache") else None
        for part in split_text(text):
            info, pieces = _render_chunk(voice, part, settings, cache, buffered=normalize)
            for pcm in pieces:
                if normalize:
                    pcm = normalize_pcm(pcm, rate)
                if not sent:
                    first = time.monotonic() - t0
                    conn.sendall(frame(b"H", json.dumps({"sample_rate": rate}).encode()))
                    sent = True
                conn.sendall(frame(b"A", pcm))
                samples += len(pcm) // 2
            chunks.append(info)
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
        total_s=round(total, 3), audio_s=round(audio, 2), rtf=round(total / audio, 3) if audio else None,
        settings={k: settings[k] for k in ("temperature", "top_p", "top_k", "take")},
        normalize=normalize, chunks=chunks)
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
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"sampling temperature, 0 for greedy (default {DEFAULT_TEMPERATURE}; "
                         "the model's own is 0.9)")
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                    help=f"nucleus sampling, 1 to turn it off (default {DEFAULT_TOP_P})")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"sample from the k likeliest tokens (default {DEFAULT_TOP_K})")
    ap.add_argument("--take", type=int, default=0,
                    help="another take of the same text: a different seed (default 0)")
    ap.add_argument("--no-cache", action="store_true",
                    help="generate every chunk afresh and store nothing")
    norm = ap.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true", default=None,
                      help="bring each chunk to the same speech level (default with --out)")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false",
                      help="leave chunk levels as generated (default when speaking)")
    ap.add_argument("--stop-worker", action="store_true", help="stop the running worker, if any")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-model", help=argparse.SUPPRESS)
    ap.add_argument("--worker-socket", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        return worker_main(Path(a.worker_model), Path(a.worker_socket))
    if a.temperature < 0 or not 0 < a.top_p <= 1 or a.top_k < 0:
        ap.error("--temperature must be >= 0, --top-p in (0, 1], --top-k >= 0")
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
    # Speaking streams each chunk as it is generated, so its level cannot be
    # measured first: normalising there would delay the first word by a chunk.
    options = render_options(a.temperature, a.top_p, a.top_k, a.take,
                             normalize=bool(a.out) if a.normalize is None else a.normalize,
                             cache=not a.no_cache)
    return 0 if speak(spoken, a.out, options) else 1


if __name__ == "__main__":
    sys.exit(main())
