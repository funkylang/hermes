"""Display-path helpers: shorten filesystem paths for user-visible previews."""

import os


def display_path(path, home: str | None = None) -> str:
    """Return ``path`` in a short, unambiguous form for display.

    Paths under the user's home directory become ``~/...`` (e.g.
    ``/home/hermes/workspace/x/concept.txt`` -> ``~/workspace/x/concept.txt``),
    so same-named files stay distinguishable while staying short. Already
    tilde-prefixed and all other paths are returned unchanged (backslashes
    normalized to ``/``).

    ``home`` defaults to ``os.path.expanduser("~")``; inject in tests.
    """
    text = str(path or "").replace("\\", "/")
    if not text or text.startswith("~"):
        return text
    home_path = (home or os.path.expanduser("~")).replace("\\", "/").rstrip("/")
    if home_path and text == home_path:
        return "~"
    prefix = home_path + "/"
    if home_path and text.startswith(prefix):
        return "~/" + text[len(prefix):]
    return text
