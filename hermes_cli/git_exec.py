"""Git execution helpers shared by checkout-touching tools.

Plain subprocess wrappers: no update-lock plumbing (the in-product updater that
owned it is retired). Callers pass git argv fragments; this module assembles and
runs the command.
"""

from __future__ import annotations

import subprocess
from typing import Any, Iterator, Sequence


def git_argv(git_cmd: Sequence[str], args: Sequence[str]) -> list[str]:
    """git executable + subcommand/flags as one argv."""
    return [*git_cmd, *args]


def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` pass-through for command argvs built here or elsewhere."""
    return subprocess.run(list(argv), **kwargs)


def run_git(git_cmd: Sequence[str], args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run one git command. ``kwargs`` are ``subprocess.run``'s."""
    return run(git_argv(git_cmd, args), **kwargs)


def contained_command(argv: Sequence[str], *, inherit_lock: bool = True, root=None) -> Iterator[tuple[list, dict]]:
    """``(argv, kwargs)`` for a checkout-writing child of a runner that hides its Popen
    (the Node build in ``pm.progress.run_contained``). Yields once."""
    yield list(argv), {}
