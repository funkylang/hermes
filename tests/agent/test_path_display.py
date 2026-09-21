"""Tests for agent/path_display.py — display_path() shortening for user-visible previews."""

from agent.path_display import display_path

HOME = "/home/hermes"


def test_home_prefix_replaced_with_tilde():
    assert display_path("/home/hermes/workspace/x/concept.txt", home=HOME) == "~/workspace/x/concept.txt"


def test_home_dir_itself_becomes_tilde():
    assert display_path(HOME, home=HOME) == "~"


def test_already_tilde_prefixed_unchanged():
    assert display_path("~/workspace/x/concept.txt", home=HOME) == "~/workspace/x/concept.txt"


def test_outside_home_unchanged():
    assert display_path("/tmp/foo/readme.md", home=HOME) == "/tmp/foo/readme.md"


def test_relative_paths_unchanged():
    assert display_path("concept.txt", home=HOME) == "concept.txt"


def test_bare_filename_unchanged():
    # The regression this fixes: same-named files must stay distinguishable, so the
    # basename is no longer returned for absolute paths.
    assert display_path("/home/hermes/workspace/concept.txt", home=HOME) == "~/workspace/concept.txt"
    assert display_path(HOME + "/concept.txt", home=HOME) == "~/concept.txt"


def test_backslashes_normalized_on_posix():
    # Windows-style separators still collapse so the prefix check cannot be fooled.
    assert display_path("/home/hermes/docs/notes.md".replace("/", "\\"), home=HOME) == "~/docs/notes.md"


def test_no_sneaky_prefix_match():
    # /home/hermes2 must NOT match home=/home/hermes (prefix must end at a separator).
    assert display_path("/home/hermes2/file.txt", home=HOME) == "/home/hermes2/file.txt"


def test_empty_and_none_safe():
    assert display_path("", home=HOME) == ""
    assert display_path(None, home=HOME) == ""


def _fake_home(monkeypatch):
    import os.path
    monkeypatch.setattr(os.path, "expanduser", lambda s: HOME)


def test_trim_error_shortens_every_path_occurrence(monkeypatch):
    # Regression for the bug where only the first path occurrence was ~-shortened.
    _fake_home(monkeypatch)
    from agent.display import _trim_error
    msg = (f"Refusing to overwrite {HOME}/workspace/x/concept.txt: "
           f"{HOME}/workspace/x/concept.txt exists but this task has not seen its content")
    out = _trim_error(msg)
    # Second occurrence (and the first) must be ~-shortened, never a raw home path.
    assert HOME not in out
    assert "~/workspace" in out


def test_trim_error_leaves_outside_home_paths_alone(monkeypatch):
    _fake_home(monkeypatch)
    from agent.display import _trim_error
    out = _trim_error("File not found: /tmp/outside/file.txt")
    assert "/tmp/outside/file.txt" in out
