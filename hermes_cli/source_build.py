"""Source launch/update composition over the shared JavaScript builders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def source_product_current(project_root: Path, product: str, out: Path) -> bool:
    """Read the compiler's receipt without acquiring tools or dependencies."""
    from pm import env_for, installed_package

    installed = installed_package("node")
    if installed is None or installed.binary is None:
        return False  # Only PM's Node may run the receipt reader; never the user's PATH copy.
    node, env = str(installed.binary), env_for("node")
    try:
        result = subprocess.run(
            [node, str(project_root / "scripts/build/freshness.mjs"),
             "--source", str(project_root), "--product", product, "--out", str(out)],
            cwd=project_root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
        return result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def source_build_env(base_env: dict | None = None, *, explicit: bool = False) -> dict[str, str]:
    from pm import ensure
    from pm.environments import project_python, running_from_selected_environment
    from pm.paths import repo_root
    from hermes_constants import get_hermes_home

    # The historical update runs on store Python with the selected environment
    # activated in-process. Icon generation starts an isolated child, which needs
    # the selected venv executable rather than the store interpreter.
    root = repo_root()
    python = str(project_python(root)) if running_from_selected_environment(root) else sys.executable
    env = {**os.environ, **(base_env or {}), "CI": "1", "HERMES_PYTHON": python,
           "PYTHON": python}
    env.pop("ESBUILD_BINARY_PATH", None)
    npmrc = get_hermes_home() / "npmrc"
    if npmrc.is_file():
        env.setdefault("NPM_CONFIG_USERCONFIG", str(npmrc))
    return ensure("npm", base_env=env, explicit=explicit).env


def run_in_custody(project_root: Path, command: list, label: str, **kwargs):
    """``pm.progress.run_contained`` for a build command that writes the checkout (node, npm):
    it and everything it starts stay in the update's custody (POSIX: the checkout lock fd, the
    caller's process group, and every descendant killed when it exits; Windows: the owner's
    kill-on-close job), so the checkout is never handed to a contender while one of them still
    writes. Outside an update it is ``run_contained`` as is."""
    from pm.progress import run_contained
    from hermes_cli.git_exec import contained_command

    with contained_command(command, root=project_root) as (argv, custody):
        return run_contained(argv, label, **kwargs, **custody)


def run_source_script(project_root: Path, script: str, *args: str, env: dict, label: str) -> None:
    command = [shutil.which("node", path=env["PATH"]), str(project_root / script), *args]
    # npm's deprecation warnings are the loudest lines and never actionable
    # here; they still land in the failure tail.
    run_in_custody(project_root, command, label, hide=lambda line: line.lower().startswith("npm warn"),
                   indent="  ", cwd=project_root, env=env)


def prepare_source_dependencies(project_root: Path, workspaces: tuple[str, ...], *, env: dict,
                                explicit: bool = False) -> None:
    from pm import lazy_installs_allowed

    run_source_script(
        project_root, "scripts/build/node-deps.mjs", "--source", str(project_root), "--reuse",
        *(() if explicit or lazy_installs_allowed() else ("--no-install",)),
        *(arg for workspace in workspaces for arg in ("--workspace", workspace)), env=env,
        label="Preparing Node dependencies",
    )
