"""Unit tests for say.py. No model, no audio: python3 -m unittest -v

ModelRender renders with the real voice; it runs only with SHIPS_COMPUTER_MODEL_TESTS=1
and the model and venv in place.
"""
import io
import json
import math
import random
import re
import socket
import threading
import unittest.mock
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import say

HERE = Path(__file__).resolve().parent


class Table(unittest.TestCase):
    def test_parses_entries_comments_and_blank_lines(self):
        t, w = say.parse_table("# note\n\nproject\tPRAH-ject\nGit  Log\tgit-log\n")
        self.assertEqual(t, {"project": "PRAH-ject", "git log": "git-log"})
        self.assertEqual(w, [])

    def test_bad_line_is_skipped_with_its_line_number(self):
        t, w = say.parse_table("ok\tfine\nno tab here\na\tb\tc\n", "x.tsv")
        self.assertEqual(t, {"ok": "fine"})
        self.assertEqual([s.split(":")[1] for s in w], ["2", "3"])

    def test_later_line_and_later_table_win(self):
        t, _ = say.parse_table("data\tDAY-ta\ndata\tDAH-ta\n")
        self.assertEqual(t["data"], "DAH-ta")
        self.assertEqual(say.merge_tables({"a": "1", "b": "1"}, {"b": "2"})["b"], "2")

    def test_hint(self):
        self.assertEqual(say.parse_hint("Project=PRAH-ject"), ("project", "PRAH-ject"))
        self.assertEqual(say.parse_hint("a=b=c"), ("a", "b=c"))
        for bad in ("project", "=x", "x="):
            with self.assertRaises(ValueError):
                say.parse_hint(bad)


class Apply(unittest.TestCase):
    T = {"project": "PRAH-ject", "git log": "git-log", "log": "LAWG", "c++": "see plus plus"}

    def a(self, text):
        return say.apply_pronunciations(text, self.T)

    def test_whole_words_case_insensitive(self):
        self.assertEqual(self.a("The project's projector."), "The PRAH-ject's projector.")
        self.assertEqual(self.a("PROJECT"), "PRAH-ject")

    def test_capital_first_letter_is_kept(self):
        self.assertEqual(say.apply_pronunciations("Data ready.", {"data": "day-ta"}), "Day-ta ready.")
        self.assertEqual(say.apply_pronunciations("data ready.", {"data": "day-ta"}), "day-ta ready.")

    def test_longest_entry_first_and_any_spacing(self):
        self.assertEqual(self.a("Read the git   log, then the log."), "Read the git-log, then the LAWG.")

    def test_one_pass(self):
        self.assertEqual(say.apply_pronunciations("a b", {"a": "b", "b": "c"}), "b c")

    def test_keys_with_symbols(self):
        self.assertEqual(self.a("Written in C++."), "Written in See plus plus.")

    def test_empty_table_is_identity(self):
        self.assertEqual(say.apply_pronunciations("Working.", {}), "Working.")


class Load(unittest.TestCase):
    def test_order_builtin_then_env_then_hints(self):
        with tempfile.TemporaryDirectory() as d:
            b, e1, e2 = (Path(d) / n for n in ("b.tsv", "e1.tsv", "e2.tsv"))
            b.write_text("x\tB\ny\tB\nz\tB\n")
            e1.write_text("y\tE1\nz\tE1\n")
            e2.write_text("z\tE2\n")
            t, w = say.load_tables([], f"{e1}:{e2}", b)
            self.assertEqual(t, {"x": "B", "y": "E1", "z": "E2"})
            t, _ = say.load_tables(["z=H"], f"{e1}:{e2}", b)
            self.assertEqual(t["z"], "H")
            self.assertEqual(w, [])

    def test_local_tables_beside_the_builtin_load_sorted_before_env(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "pronounce.tsv"
            b.write_text("x\tB\ny\tB\nz\tB\n")
            (Path(d) / "pronounce-b.tsv").write_text("y\tLB\nz\tLB\n")
            (Path(d) / "pronounce-a.tsv").write_text("x\tLA\ny\tLA\n")
            e = Path(d) / "env.tsv"
            e.write_text("z\tE\n")
            t, _ = say.load_tables([], str(e), b)
            self.assertEqual(t, {"x": "LA", "y": "LB", "z": "E"})

    def test_missing_env_table_warns_and_continues(self):
        t, w = say.load_tables(["a=b"], "/nonexistent/x.tsv", Path("/nonexistent/builtin.tsv"))
        self.assertEqual(t, {"a": "b"})
        self.assertEqual(len(w), 1)

    def test_shipped_table_parses_cleanly(self):
        _, w = say.parse_table(say.BUILTIN_TABLE.read_text(), "pronounce.tsv")
        self.assertEqual(w, [])


class Split(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(say.split_text("  Working.\n"), ["Working."])
        self.assertEqual(say.split_text(""), [])

    def test_long_text_splits_at_sentences_within_limit(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        chunks = say.split_text(text, 100)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_token_cap(self):
        self.assertEqual(say.token_cap("Working."), 74)
        self.assertEqual(say.token_cap("x" * 5000), 2048)


class SystemVoice(unittest.TestCase):
    LISTING = ("Albert              en_US    # Hello! My name is Albert.\n"
               "Eddy (English (US)) en_US    # Hello! My name is Eddy.\n"
               "Samantha            en_US    # Hello! My name is Samantha.\n"
               "Allison (Enhanced)  en_US    # Hello! My name is Allison.\n")

    def test_parse_and_pick_in_preference_order(self):
        names = say.parse_say_voices(self.LISTING)
        self.assertEqual(names, ["Albert", "Eddy (English (US))", "Samantha", "Allison (Enhanced)"])
        self.assertEqual(say.pick_system_voice(names), "Allison (Enhanced)")
        self.assertEqual(say.pick_system_voice(["Samantha", "Albert"]), "Samantha")
        self.assertIsNone(say.pick_system_voice(["Albert"]))

    def test_language_qualified_names_count_us_first(self):
        names = ["Samantha (English (UK))", "Samantha (English (US))", "Kathy"]
        self.assertEqual(say.pick_system_voice(names), "Samantha (English (US))")
        self.assertEqual(say.pick_system_voice(["Kathy", "Samantha (English (UK))"]),
                         "Samantha (English (UK))")


class Frames(unittest.TestCase):
    def test_round_trip(self):
        buf = io.BytesIO(say.frame(b"H", b'{"sample_rate": 24000}') + say.frame(b"A", b"\x01\x02")
                         + say.frame(b"Z"))
        self.assertEqual(say.read_frame(buf), (b"H", b'{"sample_rate": 24000}'))
        self.assertEqual(say.read_frame(buf), (b"A", b"\x01\x02"))
        self.assertEqual(say.read_frame(buf), (b"Z", b""))
        self.assertEqual(say.read_frame(buf), (None, b""))


class Socket(unittest.TestCase):
    def test_default_socket_path_fits_sun_path_with_the_tmp_name(self):
        old = os.environ.pop("SHIPS_COMPUTER_QWEN_RUN", None)
        try:
            _, sock, _ = say._worker_paths(Path("/any/model"), Path("/any/python"))
        finally:
            if old is not None:
                os.environ["SHIPS_COMPUTER_QWEN_RUN"] = old
        self.assertTrue(str(sock).startswith("/tmp/"))
        self.assertLessEqual(len(os.fsencode(f"{sock}.9999999.tmp")), say.SUN_PATH_MAX)

    def test_connect_reaches_a_socket_in_a_folder_too_long_for_sun_path(self):
        import socket
        with tempfile.TemporaryDirectory() as d:
            deep = Path(d, "x" * 60, "y" * 60)
            deep.mkdir(parents=True)
            path = deep / "worker-0123456789.sock"
            self.assertGreater(len(os.fsencode(str(path))), say.SUN_PATH_MAX)
            cwd = os.getcwd()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            os.chdir(deep)
            try:
                srv.bind(path.name)
            finally:
                os.chdir(cwd)
            srv.listen(1)
            with srv, say._connect(path):
                pass
            self.assertEqual(os.getcwd(), cwd)

    def test_run_dir_must_be_private(self):
        with tempfile.TemporaryDirectory() as d:
            rd = Path(d, "run")
            say.ensure_run_dir(rd)
            self.assertEqual(rd.stat().st_mode & 0o777, 0o700)
            rd.chmod(0o755)
            with self.assertRaises(say.Unavailable):
                say.ensure_run_dir(rd)


class PrintText(unittest.TestCase):
    def run_say(self, *args, env=None, stdin=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith("SHIPS_COMPUTER_")}
        e.update(env or {})
        return subprocess.run([sys.executable, str(HERE / "say.py"), *args], input=stdin,
                              capture_output=True, text=True, env=e)

    def test_print_text_applies_tables_and_hints_without_a_model(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write("build\tBILD\n")
        try:
            r = self.run_say("--print-text", "--say-as", "complete=kum-PLEET", "Project build complete.",
                             env={"SHIPS_COMPUTER_PRONOUNCE": f.name,
                                  "SHIPS_COMPUTER_QWEN_MODEL": "/nonexistent"})
        finally:
            os.unlink(f.name)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "PRAH-ject BILD kum-PLEET.\n")

    def test_print_text_reads_stdin(self):
        r = self.run_say("--print-text", stdin="Working.\n")
        self.assertEqual((r.returncode, r.stdout), (0, "Working.\n"))

    def test_bad_hint_is_a_usage_error(self):
        self.assertEqual(self.run_say("--print-text", "--say-as", "oops", "x").returncode, 2)



# ---------------------------------------------------------------------------
# Repeatability

BASE = {"speaker": "computer", "lang_code": "auto", "streaming_interval": 0.5,
        "repetition_penalty": 1.05, "mlx": "0.32.4", "mlx_audio": "0.6.4"}


def settings(**kw):
    o = {"temperature": 0.65, "top_p": 0.9, "top_k": 50, "take": 0}
    o.update(kw)
    return say.render_settings(BASE, **o)


class FakeVoice:
    """Stands in for say.Voice: audio is a function of (text, seed), and every call is counted."""
    rate = 8000
    model_id = "fake-model"
    base = BASE

    def __init__(self):
        self.calls = []

    def generate(self, text, seed, st):
        self.calls.append((text, seed, st))
        rng = random.Random(f"{text}|{seed}")
        for _ in range(3):
            yield bytes(rng.randrange(256) for _ in range(400))


def serve(voice, request, log=None):
    """Run say._serve on one request; returns the PCM it sent."""
    a, b = socket.socketpair()
    events = []
    t = threading.Thread(target=say._serve, args=(b, voice, log or (lambda e, **f: events.append((e, f)))))
    t.start()
    a.sendall(json.dumps(request).encode() + b"\n")
    f, pcm = a.makefile("rb"), b""
    while True:
        kind, payload = say.read_frame(f)
        if kind == b"A":
            pcm += payload
        elif kind in (b"Z", b"E", None):
            break
    t.join()
    a.close()
    b.close()
    serve.events = events
    return pcm


class Seed(unittest.TestCase):
    def test_same_inputs_same_seed(self):
        self.assertEqual(say.chunk_seed("Working.", "m", settings()), say.chunk_seed("Working.", "m", settings()))

    def test_seed_changes_with_text_model_settings_and_take(self):
        s0 = say.chunk_seed("Working.", "m", settings())
        for other in (say.chunk_seed("Working!", "m", settings()),
                      say.chunk_seed("Working.", "m2", settings()),
                      say.chunk_seed("Working.", "m", settings(temperature=0.7)),
                      say.chunk_seed("Working.", "m", settings(top_p=1.0)),
                      say.chunk_seed("Working.", "m", settings(take=1))):
            self.assertNotEqual(s0, other)
        self.assertTrue(0 <= s0 < 2 ** 32)

    def test_each_chunk_gets_its_own_seed_from_its_own_text(self):
        voice = FakeVoice()
        text = " ".join(f"Sentence number {i} is here and it is fairly long." for i in range(20))
        serve(voice, {"text": text, "cache": None})
        parts = say.split_text(text)
        self.assertGreater(len(parts), 1)
        self.assertEqual([c[0] for c in voice.calls], parts)
        self.assertEqual([c[1] for c in voice.calls],
                         [say.chunk_seed(p, voice.model_id, voice.calls[0][2]) for p in parts])

    def test_voice_seed_is_the_seed_for_every_chunk_plus_the_take(self):
        st = settings(voice_seed=1701)
        self.assertEqual(say.chunk_seed("Working.", "m", st), 1701)
        self.assertEqual(say.chunk_seed("Something else entirely.", "m2", st), 1701)
        self.assertEqual(say.chunk_seed("Working.", "m", settings(voice_seed=1701, take=2)), 1703)
        self.assertEqual(say.chunk_seed("Working.", "m", settings(voice_seed=2 ** 32 - 1, take=1)), 0)

    def test_no_voice_seed_leaves_settings_and_text_seeds_as_before(self):
        self.assertNotIn("voice_seed", settings())
        self.assertNotIn("voice_seed", settings(voice_seed=None))
        self.assertNotEqual(say.chunk_seed("Working.", "m", settings()),
                            say.chunk_seed("Working!", "m", settings()))

    def test_voice_seed_reaches_every_chunk_and_keys_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            text = " ".join(f"Sentence number {i} is here and it is fairly long." for i in range(20))
            serve(voice, {"text": text, "cache": d, "voice_seed": 42})
            self.assertGreater(len(voice.calls), 1)
            self.assertEqual({c[1] for c in voice.calls}, {42})
            n = len(voice.calls)
            serve(voice, {"text": text, "cache": d, "voice_seed": 43})
            self.assertEqual(len(voice.calls), 2 * n)
            serve(voice, {"text": text, "cache": d, "voice_seed": 42})
            self.assertEqual(len(voice.calls), 2 * n)

    def test_resolve_voice_seed_numbers_names_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "voices.tsv"
            t.write_text("# name<TAB>seed<TAB>note\nVega\t7\twarmer\nbad line\n")
            self.assertEqual(say.resolve_voice_seed(None, t), say.DEFAULT_VOICE_SEED)
            self.assertEqual(say.resolve_voice_seed("", t), say.DEFAULT_VOICE_SEED)
            self.assertIsNone(say.resolve_voice_seed("text", t))
            self.assertIsNone(say.resolve_voice_seed("TEXT", t))
            self.assertEqual(say.resolve_voice_seed("123", t), 123)
            self.assertEqual(say.resolve_voice_seed("vega", t), 7)
            self.assertEqual(say.resolve_voice_seed("VEGA", t), 7)
            with self.assertRaises(ValueError):
                say.resolve_voice_seed("rigel", t)
            with self.assertRaises(ValueError):
                say.resolve_voice_seed(str(2 ** 32), t)
        voices, warnings = say.parse_voices("a\t1\nb\tx\nc\t-1\n")
        self.assertEqual(voices, {"a": 1})
        self.assertEqual(len(warnings), 2)

    def test_shipped_voices_table_parses_cleanly(self):
        voices, warnings = say.parse_voices(say.VOICES_TABLE.read_text(encoding="utf-8"))
        self.assertEqual(warnings, [])
        self.assertTrue(voices)

    def test_default_voice_is_antares_and_needs_no_table(self):
        self.assertEqual((say.DEFAULT_VOICE, say.DEFAULT_VOICE_SEED), ("Antares", 4077))
        voices, _ = say.parse_voices(say.VOICES_TABLE.read_text(encoding="utf-8"))
        self.assertEqual(voices[say.DEFAULT_VOICE.lower()], say.DEFAULT_VOICE_SEED)
        missing = Path(tempfile.gettempdir()) / "no-such-dir" / "voices.tsv"
        self.assertEqual(say.resolve_voice_seed(None, missing), 4077)

    def _cli_voice_seed(self, argv, env_value):
        """The voice_seed main() hands to speak() for argv, with SHIPS_COMPUTER_VOICE_SEED
        set to env_value (None: unset)."""
        env = {k: v for k, v in os.environ.items() if k != "SHIPS_COMPUTER_VOICE_SEED"}
        if env_value is not None:
            env["SHIPS_COMPUTER_VOICE_SEED"] = env_value
        with unittest.mock.patch.dict(os.environ, env, clear=True), \
                unittest.mock.patch.object(say, "speak", return_value=True) as speak:
            self.assertEqual(say.main(argv + ["--no-cache", "Working."]), 0)
        return speak.call_args[0][2]["voice_seed"]

    def test_cli_default_is_antares_and_flag_or_env_overrides(self):
        self.assertEqual(self._cli_voice_seed([], None), 4077)
        self.assertEqual(self._cli_voice_seed([], ""), 4077)
        self.assertEqual(self._cli_voice_seed(["--voice-seed", "vega"], None), 7)
        self.assertEqual(self._cli_voice_seed(["--voice-seed", "123"], None), 123)
        self.assertEqual(self._cli_voice_seed([], "rigel"), 42)
        self.assertEqual(self._cli_voice_seed(["--voice-seed", "1701"], "rigel"), 1701)
        self.assertIsNone(self._cli_voice_seed(["--voice-seed", "text"], None))

    def test_default_voice_renders_the_same_text_identically(self):
        text = " ".join(f"Sentence number {i} is here and it is fairly long." for i in range(20))
        req = {"text": text, **say.render_options(cache=None, voice_seed=say.resolve_voice_seed(None))}
        a, b = FakeVoice(), FakeVoice()
        pcm_a, pcm_b = serve(a, dict(req)), serve(b, dict(req))
        self.assertGreater(len(pcm_a), 0)
        self.assertEqual(pcm_a, pcm_b)
        self.assertEqual({c[1] for c in a.calls}, {4077})

    def test_model_id_hashes_contents_and_uses_the_memo_only_when_files_match(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "model")
            (m / "sub").mkdir(parents=True)
            (m / "config.json").write_text("{}")
            (m / "sub" / "w.bin").write_bytes(b"\x00" * 10)
            memo = Path(d, "memo.json")
            first = say.model_id(m, memo)
            self.assertEqual(say.model_id(m, memo), first)
            self.assertEqual(json.loads(memo.read_text())["model_id"], first)
            (m / "sub" / "w.bin").write_bytes(b"\x01" * 11)
            self.assertNotEqual(say.model_id(m, memo), first)
            self.assertEqual(say.model_id(m, None), say.model_id(m, memo))


class Sampling(unittest.TestCase):
    def test_request_settings_reach_generate(self):
        voice = FakeVoice()
        serve(voice, {"text": "Working.", "temperature": 0.6, "top_p": 0.8, "top_k": 20, "take": 2,
                      "cache": None})
        st = voice.calls[0][2]
        self.assertEqual((st["temperature"], st["top_p"], st["top_k"], st["take"]), (0.6, 0.8, 20, 2))
        self.assertEqual(st["speaker"], "computer")

    def test_defaults(self):
        o = say.render_options(cache=None)
        self.assertEqual((o["temperature"], o["top_p"], o["top_k"], o["take"], o["normalize"], o["cache"]),
                         (0.65, 0.9, 50, 0, False, None))
        voice = FakeVoice()
        serve(voice, {"text": "Working."})   # an old-style request: defaults apply
        st = voice.calls[0][2]
        self.assertEqual((st["temperature"], st["top_p"], st["top_k"]), (0.65, 0.9, 50))

    def test_cache_env(self):
        old = os.environ.get("SHIPS_COMPUTER_CACHE")
        try:
            os.environ["SHIPS_COMPUTER_CACHE"] = ""
            self.assertIsNone(say.cache_dir())
            os.environ.pop("SHIPS_COMPUTER_CACHE")
            self.assertEqual(say.cache_dir(), HERE / ".cache")
        finally:
            if old is not None:
                os.environ["SHIPS_COMPUTER_CACHE"] = old


class Cache(unittest.TestCase):
    def test_second_render_of_the_same_text_makes_no_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            req = {"text": "Diagnostic complete.", "cache": d}
            first = serve(voice, req)
            self.assertEqual(len(voice.calls), 1)
            second = serve(voice, req)
            self.assertEqual(len(voice.calls), 1)
            self.assertEqual(second, first)
            self.assertEqual(serve.events[-1][1]["chunks"][0]["cached"], True)

    def test_a_changed_sentence_regenerates_only_its_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            lines = [f"Line {i} of the report says something long enough to fill a chunk on its own, "
                     f"more or less, with a few extra words to be sure of it {'x' * 300}." for i in range(3)]
            serve(voice, {"text": " ".join(lines), "cache": d})
            self.assertEqual(len(voice.calls), 3)
            lines[1] = lines[1].replace("Line 1", "Line one")
            serve(voice, {"text": " ".join(lines), "cache": d})
            self.assertEqual(len(voice.calls), 4)
            self.assertTrue(voice.calls[-1][0].startswith("Line one"))

    def test_other_settings_or_take_or_no_cache_generate_again(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            serve(voice, {"text": "Working.", "cache": d})
            serve(voice, {"text": "Working.", "cache": d, "take": 1})
            serve(voice, {"text": "Working.", "cache": d, "temperature": 0.7})
            serve(voice, {"text": "Working.", "cache": None})
            self.assertEqual(len(voice.calls), 4)
            self.assertEqual(len(list(Path(d).rglob("*.wav"))), 3)

    def test_cache_file_is_a_wav_of_the_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            pcm = serve(voice, {"text": "Working.", "cache": d})
            [f] = Path(d).rglob("*.wav")
            self.assertEqual(say.wav_read(f), (voice.rate, pcm))


class CachePrivacy(unittest.TestCase):
    def mode(self, path):
        return path.stat().st_mode & 0o777

    def test_cache_folders_are_0700_and_files_0600_whatever_the_umask(self):
        old = os.umask(0o000)
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d, "a", "cache")
                pcm = serve(FakeVoice(), {"text": "Working.", "cache": str(root)})
                self.assertTrue(pcm)
                [wav] = root.rglob("*.wav")
                self.assertEqual(self.mode(wav), 0o600)
                for folder in (Path(d, "a"), root, root / "chunks", wav.parent):
                    self.assertEqual(self.mode(folder), 0o700, folder)
                m = Path(d, "model")
                m.mkdir()
                (m / "config.json").write_text("{}")
                say.model_id(m, root / "model-id.json")
                self.assertEqual(self.mode(root / "model-id.json"), 0o600)
                self.assertEqual([f.name for f in root.rglob("*.tmp")], [])
        finally:
            os.umask(old)

    def test_a_cache_folder_open_to_others_is_refused_and_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "cache")
            root.mkdir(mode=0o755)
            root.chmod(0o755)
            with self.assertRaises(say.Unavailable):
                say.ChunkCache(root)
            voice = FakeVoice()
            self.assertTrue(serve(voice, {"text": "Working.", "cache": str(root)}))
            self.assertEqual(serve.events[0][0], "cache_refused")
            self.assertEqual(list(root.iterdir()), [])
            old = os.environ.get("SHIPS_COMPUTER_CACHE")
            os.environ["SHIPS_COMPUTER_CACHE"] = str(root)
            try:
                with unittest.mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                    self.assertIsNone(say.private_cache_dir())
                self.assertIn("not private", err.getvalue())
            finally:
                if old is None:
                    os.environ.pop("SHIPS_COMPUTER_CACHE")
                else:
                    os.environ["SHIPS_COMPUTER_CACHE"] = old

    def test_a_cache_folder_owned_by_another_user_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "cache")
            root.mkdir(mode=0o700)
            me = os.getuid()
            with unittest.mock.patch.object(say.os, "getuid", return_value=me + 1):
                with self.assertRaises(say.Unavailable):
                    say.ChunkCache(root)

    def test_a_cached_file_writable_by_others_is_not_played(self):
        with tempfile.TemporaryDirectory() as d:
            voice = FakeVoice()
            serve(voice, {"text": "Working.", "cache": d})
            [wav] = Path(d).rglob("*.wav")
            wav.chmod(0o666)
            serve(voice, {"text": "Working.", "cache": d})
            self.assertEqual(len(voice.calls), 2)


def tone(seconds, rate, rms_db, silence=0.0):
    amp = 10 ** (rms_db / 20) * math.sqrt(2) * 32767
    n = int(seconds * rate)
    a = [int(amp * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n)]
    a += [0] * int(silence * rate)
    return say.array("h", a).tobytes()


class Level(unittest.TestCase):
    RATE = 8000

    def test_two_clips_at_different_levels_come_out_matched(self):
        quiet, loud = tone(1, self.RATE, -29), tone(1, self.RATE, -11)
        self.assertGreater(say.speech_level_db(loud, self.RATE) - say.speech_level_db(quiet, self.RATE), 17)
        a = say.speech_level_db(say.normalize_pcm(quiet, self.RATE), self.RATE)
        b = say.speech_level_db(say.normalize_pcm(loud, self.RATE), self.RATE)
        self.assertAlmostEqual(a, say.SPEECH_LEVEL_DB, delta=0.2)
        self.assertAlmostEqual(b, say.SPEECH_LEVEL_DB, delta=0.2)

    def test_pauses_do_not_lower_the_measured_level(self):
        self.assertAlmostEqual(say.speech_level_db(tone(1, self.RATE, -20, silence=2), self.RATE),
                               say.speech_level_db(tone(1, self.RATE, -20), self.RATE), delta=0.1)

    def test_never_clips(self):
        out = say._samples(say.normalize_pcm(tone(1, self.RATE, -40), self.RATE, target_db=-1))
        self.assertLessEqual(max(abs(x) for x in out), 32767)
        self.assertEqual(say.normalize_pcm(b"\x00\x00" * 100, self.RATE), b"\x00\x00" * 100)

    def test_normalize_request_levels_each_whole_chunk(self):
        class Two(FakeVoice):
            def generate(self, text, seed, st):
                self.calls.append(text)
                yield tone(0.5, self.rate, -30 if text.startswith("Quiet") else -12)
        voice = Two()
        text = "Quiet " + "a" * 400 + ". Loud " + "b" * 400 + "."
        pcm = serve(voice, {"text": text, "normalize": True, "cache": None})
        half = len(pcm) // 2
        for part in (pcm[:half], pcm[half:]):
            self.assertAlmostEqual(say.speech_level_db(part, voice.rate), say.SPEECH_LEVEL_DB, delta=0.2)


class Pins(unittest.TestCase):
    def test_requirements_are_exact_versions(self):
        for req in HERE.glob("requirements-*.txt"):
            for line in req.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    self.assertRegex(line, r"^[A-Za-z0-9_.\-\[\]]+==[0-9][^ ]*$", f"{req.name}: {line}")

    def test_sha256sums_cover_both_weight_tarballs(self):
        lines = (HERE / "SHA256SUMS").read_text().split("\n")
        names = [m.group(1) for l in lines if (m := re.match(r"^[0-9a-f]{64}  (\S+)$", l))]
        self.assertEqual(sorted(names), ["ships-computer-v1-hf-bf16.tar", "ships-computer-v1-mlx-6bit.tar"])


class Cli(unittest.TestCase):
    def test_bad_sampling_values_are_usage_errors(self):
        for args in (["--top-p", "0"], ["--top-p", "1.5"], ["--temperature", "-1"]):
            r = subprocess.run([sys.executable, str(HERE / "say.py"), *args, "--print-text", "x"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, args)

    def test_new_flags_leave_print_text_alone(self):
        r = subprocess.run([sys.executable, str(HERE / "say.py"), "--temperature", "0.7", "--top-p", "0.95",
                            "--take", "3", "--no-cache", "--print-text", "Working."],
                           capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (0, "Working.\n"))


@unittest.skipUnless(os.environ.get("SHIPS_COMPUTER_MODEL_TESTS") == "1"
                     and (say.model_dir() / "config.json").is_file() and os.access(say.qwen_python(), os.X_OK),
                     "set SHIPS_COMPUTER_MODEL_TESTS=1 with the model and venv installed")
class ModelRender(unittest.TestCase):
    SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import say
v = say.Voice(say.model_dir(), None)
st = say.render_settings(v.base, say.DEFAULT_TEMPERATURE, say.DEFAULT_TOP_P, say.DEFAULT_TOP_K, 0)
text = "Diagnostic complete."
seed = say.chunk_seed(text, v.model_id, st)
a = b"".join(v.generate(text, seed, st))
b = b"".join(v.generate(text, seed, st))
c = b"".join(v.generate(text, seed ^ 1, st))
print(json.dumps({"same": a == b, "n": len(a), "other_seed_differs": a != c}))
"""

    def test_same_sentence_twice_gives_identical_samples(self):
        r = subprocess.run([str(say.qwen_python()), "-c", self.SCRIPT, str(HERE)],
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertGreater(res["n"], 0)
        self.assertTrue(res["same"], "two renders with one seed differ")
        self.assertTrue(res["other_seed_differs"])

    def test_default_voice_renders_one_line_to_identical_files(self):
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k != "SHIPS_COMPUTER_VOICE_SEED"}
            outs = []
            for n in (1, 2):
                out = Path(d) / f"{n}.wav"
                r = subprocess.run([sys.executable, str(HERE / "say.py"), "--no-cache", "--out", str(out),
                                    "Good morning. You have three briefs waiting."],
                                   capture_output=True, text=True, timeout=600, env=env)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertNotIn("macOS voice", r.stderr)
                outs.append(out.read_bytes())
            self.assertGreater(len(outs[0]), 1000)
            self.assertEqual(outs[0], outs[1], "two default-voice renders of one line differ")

if __name__ == "__main__":
    unittest.main()
