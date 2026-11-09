"""Unit tests for say.py's text handling. No model, no audio: python3 -m unittest -v"""
import io
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


if __name__ == "__main__":
    unittest.main()
