"""Tests for agent.control_words: standalone control word detection.

Behavior contracts pinned here:

1. RECOGNITION — only a message that is EXACTLY the control word, after whitespace
   stripping and case-folding, triggers anything. Anything else (extra words, punctuation
   after "done" or "halt", multimodal content-part lists, etc.) passes through unchanged.
2. DONE INJECTION — on recognition of "done", the model-facing copy of the user message carries a
   prepended instruction loaded from ~/.hermes/messages/DONE.md when present and non-blank,
   else a built-in default; the message body itself stays in place (appended after the note).
3. CLEAN PERSISTENCE — regardless of injection, ``persist_user_message`` is pinned to the
   original clean text so the durable transcript row is exactly what the user typed.
4. ROBUSTNESS — a missing/unreadable/blank DONE.md (or missing messages/ dir) must never
   raise or change behavior: the default instruction is used instead.
5. HALT DETECTION — "halt" triggers an emergency brake flag that blocks tool execution;
   recognition rules are identical to "done".
"""

import pytest

from agent import control_words
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Point get_hermes_home() at an empty temp dir by default so file-reading tests never
    touch the real ~/.hermes and so absence of messages/DONE.md is the default state unless
    a test explicitly writes one."""
    token = set_hermes_home_override(tmp_path)
    yield
    reset_hermes_home_override(token)


class TestRecognition:
    def test_exact_word_matches(self):
        assert control_words.is_standalone_control_word("done", "done")

    def test_case_insensitive(self):
        for variant in ("Done", "DONE", "dOnE"):
            assert control_words.is_standalone_control_word(variant, "done"), variant

    def test_surrounding_whitespace_ignored(self):
        for variant in ("  done  ", "\tdone\n", "done ", " done"):
            assert control_words.is_standalone_control_word(variant, "done"), variant

    @pytest.mark.parametrize("text", [
        "done with step 3",
        "I'm done.",
        "not done yet",
        "doned",
        "do ne",
        "",
        "   ",
        "done!\n",  # trailing punctuation is part of the message, not just whitespace
    ])
    def test_non_exact_word_does_not_match(self, text):
        assert not control_words.is_standalone_control_word(text, "done")

    def test_non_string_input_never_matches(self):
        for value in ([{"type": "text", "text": "done"}], None, 42, ["done"]):
            assert not control_words.is_standalone_control_word(value, "done"), value


class TestDoneInstructionContent:
    def test_missing_DONE_md_falls_back_to_default(self):
        # No DONE.md written into the isolated tmp hermes home by the autouse fixture.
        assert control_words._read_control_word_file("DONE.md", "FALLBACK") == "FALLBACK"

    def test_reads_DONE_md_when_present(self, tmp_path):
        custom = "Wrap up by updating concept.txt and saving any open notes."
        msg_dir = tmp_path / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)
        (msg_dir / "DONE.md").write_text(custom, encoding="utf-8")
        assert control_words._read_control_word_file("DONE.md", "UNUSED") == custom

    def test_blank_DONE_md_falls_back_to_default(self, tmp_path):
        msg_dir = tmp_path / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)
        (msg_dir / "DONE.md").write_text("   \n  \n", encoding="utf-8")
        assert control_words._read_control_word_file("DONE.md", "FALLBACK") == "FALLBACK"

    def test_unreadable_DONE_md_falls_back_to_default(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("simulated read error")
        monkeypatch.setattr(control_words.Path, "read_text", boom)
        assert control_words._read_control_word_file("DONE.md", "FALLBACK") == "FALLBACK"


class TestHaltControlWordRecognition:
    def test_exact_word_matches(self):
        assert control_words.is_standalone_halt_control_word("halt")

    def test_case_insensitive(self):
        for variant in ("Halt", "HALT", "hAlT"):
            assert control_words.is_standalone_halt_control_word(variant), variant

    def test_surrounding_whitespace_ignored(self):
        for variant in ("  halt  ", "\thalt\n", "halt ", " halt"):
            assert control_words.is_standalone_halt_control_word(variant), variant

    @pytest.mark.parametrize("text", [
        "halt the server",  # extra words should not trigger
        "I'm halted.",
        "not a halt yet",
        "halted",
        "hal t",
        "",
        "   ",
        "halt!\n",  # trailing punctuation is part of the message, not just whitespace
    ])
    def test_non_exact_word_does_not_match(self, text):
        assert not control_words.is_standalone_halt_control_word(text), text

    def test_non_string_input_never_matches(self):
        for value in ([{"type": "text", "text": "halt"}], None, 42, ["halt"]):
            assert not control_words.is_standalone_halt_control_word(value), value


class TestApplyDoneControlWord:
    def test_non_control_word_message_passes_through_unchanged(self):
        msg = "please summarize the diff"
        assert control_words.apply_done_control_word(msg, None) == (msg, None)

    def test_recognized_done_injects_default_and_keeps_clean_persist(self):
        prefixed, persist = control_words.apply_done_control_word("done", None)
        assert persist == "done"
        assert isinstance(prefixed, str)
        assert prefixed.endswith("done")
        assert control_words.DEFAULT_DONE_INSTRUCTION in prefixed

    def test_recognized_done_prefers_user_written_DONE_md(self, tmp_path):
        custom = "Custom wrap-up: save everything to notes.md."
        msg_dir = tmp_path / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)
        (msg_dir / "DONE.md").write_text(custom, encoding="utf-8")
        prefixed, persist = control_words.apply_done_control_word("  DONE  ", None)
        assert persist == "  DONE  "  # exact bytes the user typed, including whitespace/case
        assert prefixed.startswith(custom)
        assert prefixed.endswith("  DONE  ")

    def test_existing_persist_user_message_is_preserved(self):
        # Callers (e.g. voice-input prefix handling) may already pin a clean persist value; we
        # must not overwrite it with our own logic — only prepend to the API-facing copy.
        prefixed, persist = control_words.apply_done_control_word("done", "already-clean")
        assert persist == "already-clean"
        assert prefixed.endswith("done")

    def test_multimodal_message_never_recognized(self):
        content = [{"type": "text", "text": "done"}]
        assert control_words.apply_done_control_word(content, None) == (content, None)
