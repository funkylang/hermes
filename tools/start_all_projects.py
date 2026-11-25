#!/usr/bin/env python3
"""start-all-projects — one tmux session, one window per workspace project.

Creates (or extends) a tmux session named "projects" on the CURRENT user's
tmux server. Each top-level workspace directory that has a repo/ subdir
gets a window named after the project, starting in its repo/ directory —
the operator's git review/push tab for each project. Projects without a
repo/ are skipped (ruling 2026-11-25).

Idempotent: existing session and windows are kept; missing projects are
added; all windows are reindexed alphabetically at the end.

Stdlib only — must run on any host python3 without a hermes installation.
Runs unprivileged: it manages only the invoking user's own tmux server.
"""

from __future__ import annotations

import os
import subprocess
import sys

WORKSPACE = "/home/agents/workspace"
SESSION = "projects"


def _tmux(*args: str, check: bool = True) -> tuple[int, str]:
    """Run a tmux command, return (returncode, stdout)."""
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if proc.returncode != 0 and check:
        sys.stderr.write(f"start-all-projects: tmux {' '.join(args)} failed:\n"
                         f"  {proc.stderr.strip()}\n")
        raise SystemExit(1)
    return (proc.returncode, proc.stdout)


def _window_names(session: str) -> list[tuple[int, str]]:
    """[(index, name), ...] for the session's windows."""
    _rc, out = _tmux("list-windows", "-t", session, "-F", "#{window_index} #{window_name}")
    return [(int(i), n) for i, n in (line.split(" ", 1) for line in out.splitlines())]


def _projects() -> list[str]:
    """Project dirs that have a repo/ — nothing else gets a window."""
    try:
        entries = sorted(os.listdir(WORKSPACE))
    except FileNotFoundError:
        sys.stderr.write(f"start-all-projects: no workspace at {WORKSPACE}\n")
        raise SystemExit(1)
    return [name for name in entries
            if os.path.isdir(os.path.join(WORKSPACE, name, "repo"))]


def _reindex() -> None:
    """Sort windows alphabetically by swapping (preserves all panes)."""
    n = len(_window_names(SESSION))
    for target in range(n):
        names = {name: i for i, name in _window_names(SESSION)}
        wanted = sorted(names)[target]
        if names[wanted] != target:
            subprocess.run(["tmux", "swap-window", "-s", f"{SESSION}:{names[wanted]}",
                            "-t", f"{SESSION}:{target}"], check=True, capture_output=True)


def main() -> int:
    projects = _projects()
    if not projects:
        print("start-all-projects: no project with a repo/ found — nothing to do.")
        return 0

    # Ensure the session exists (first window = first project).
    rc, _out = _tmux("has-session", "-t", SESSION, check=False)
    if rc != 0:
        first = projects[0]
        _tmux("new-session", "-d", "-s", SESSION,
              "-n", first, "-c", os.path.join(WORKSPACE, first, "repo"))
        projects = projects[1:]
        print(f"created tmux session '{SESSION}' (first window: {first})")

    # Add missing project windows.
    existing = {name for _, name in _window_names(SESSION)}
    added = []
    for name in projects:
        if name not in existing:
            _tmux("new-window", "-d", "-t", SESSION,
                  "-n", name, "-c", os.path.join(WORKSPACE, name, "repo"))
            added.append(name)
    if added:
        print(f"added windows: {', '.join(added)}")

    # Reindex alphabetically.
    _reindex()
    final_names = [n for _, n in _window_names(SESSION)]
    if not added and final_names == sorted(existing):
        print(f"nothing to do — session '{SESSION}' already has all projects in order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
