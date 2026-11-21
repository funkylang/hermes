"""``hermes agents`` — multi-agent host management (tier + launcher roster).

Subcommands:
  list   Show every agent on this host, its tier, run status, last activity.
  add    Create a new agent (user + launcher + seeded config). [planned]

Design and rulings: ~/workspace/hermes/concepts/agents-command.txt (Superhermes fork).
Stdlib only; read-only in v1 for ``list``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

_LIKE_TIME_FORMAT = "%Y-%m-%d %H:%M"

# Tier path pattern: hermes-agent, hermes-agent-testing, etc., under any root.
_TIER_PATH_RE = re.compile(r"(/[\w./-]*hermes-agent(?:-\w+)?)/?")

# FORK_VERSION is defined in the fork's hermes_cli/fork_version.py as a plain string.
_FORK_VERSION_DEF_RE = re.compile(r"^FORK_VERSION\s*=\s*[\"'](.+?)[\"']", re.MULTILINE)


def _read_launcher_meta(launcher_path: Path):
    """Parse (tier_tree, hermes_home) out of a launcher script's text.

    Handles both launcher styles on this host:
      - thin sh wrapper:   export HERMES_HOME=/home/x/.hermes + exec <tree>/hermes
      - self-contained:    sys.path.insert(0, "<tree>") + HERMES_HOME in env code
    Returns (None, None) for unrecognized shapes.
    """
    try:
        text = launcher_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (None, None)

    tree_m = _TIER_PATH_RE.search(text)
    if not tree_m:
        return (None, None)
    tree = tree_m.group(1).rstrip("/")

    # HERMES_HOME= assignments (sh style) or "os.environ[...HERMES_HOME...]" context
    home_m = re.search(r"HERMES_HOME\s*=\s*[\"']?(/home/[\w-]+/\.hermes)[\"'\s]", text)
    if home_m:
        return (tree, home_m.group(1))

    # Self-contained launchers derive HERMES_HOME from the tree's home dir.
    # The launcher is at /home/<user>/.local/bin/hermes; infer its owner home.
    owner = launcher_path.parent.parent.parent  # .../x/.local/bin/hermes -> /home/x
    if owner.parent.name == "home":
        return (tree, str(owner / ".hermes"))
    return (tree, None)


def _read_fork_version(tree: Path) -> str:
    fv_path = tree / "hermes_cli" / "fork_version.py"
    try:
        text = fv_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no fork banner)"
    m = _FORK_VERSION_DEF_RE.search(text)
    return m.group(1) if m else "(no fork banner)"


def _running_pids_by_tree() -> dict[str, list[int]]:
    """Map tier tree path -> list of PIDs whose cmdline references that tree."""
    found: dict[str, list[int]] = {}
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        if "python" not in cmd:
            continue
        for m in _TIER_PATH_RE.finditer(cmd):
            found.setdefault(m.group(1).rstrip("/"), []).append(int(entry))
    return found


def _tier_label(tree: str) -> str:
    """Short tier name for display.

    Tier convention on this host (see concepts/hermes-shared-deployment.txt):
      /home/agents/hermes-agent          -> prod
      /home/agents/hermes-agent-<name>   -> <name>  (e.g. "testing")
      anything else                      -> "(dev)"  personal/dev trees, not a managed tier
    """
    parts = tree.rstrip("/").split("/")  # ["", "home", "agents", ...] for absolute paths
    if len(parts) > 2 and parts[1] == "home" and parts[2] == "agents":
        last = parts[-1]
        if last == "hermes-agent":
            return "prod"
        if last.startswith("hermes-agent-"):
            return last[len("hermes-agent-"):]
    return "(dev)"


def _relative_age(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return time.strftime(_LIKE_TIME_FORMAT, time.localtime(time.time() - seconds))


def cmd_agents_list(_args=None) -> int:
    home_root = Path("/home")
    if not home_root.is_dir():
        print("hermes agents list: no /home on this host — nothing to report.", file=sys.stderr)
        return 1

    # 1. Collect agents from launchers.
    agents = {}          # name -> dict(tree, hermes_home)
    unreadable = []      # names whose launcher we could not read
    unreadable_homes = []  # names whose HERMES_HOME we cannot stat (mode 700, no root)
    for entry in sorted(home_root.iterdir()):
        if not entry.is_dir():
            continue
        launcher = entry / ".local" / "bin" / "hermes"
        if not launcher.exists():
            continue
        tree, home = _read_launcher_meta(launcher)
        if tree is None:
            unreadable.append(entry.name)
        else:
            agents[entry.name] = {"tree": tree, "home": home}

    # 2. Tiers: distinct trees, with fork versions.
    tiers: dict[str, str] = {}
    for meta in agents.values():
        if meta["tree"] not in tiers:
            tiers[meta["tree"]] = _read_fork_version(Path(meta["tree"]))

    # 3. Run status per tree.
    running = _running_pids_by_tree()
    now = time.time()

    # 4. Print roster.
    try:
        hostname = os.uname().nodename
    except AttributeError:
        hostname = "unknown host"
    print(f"Agents on {hostname}:\n")
    if not agents and not unreadable:
        print("  (none)")
    else:
        name_w = max([len(n) for n in agents] + [4])
        tier_labels = {t: _tier_label(t) for t in tiers}
        tier_w = max([len(v) for v in tier_labels.values()] + [4]) if tier_labels else 4

        for name in sorted(agents):
            meta = agents[name]
            tree = meta["tree"]
            label = tier_labels.get(tree, "?")
            pids = running.get(tree, [])
            status = "running" if pids else "idle"
            age = ""
            if meta["home"]:
                db = Path(meta["home"]) / "state.db"
                try:
                    age = _relative_age(now - db.stat().st_mtime)
                except PermissionError:
                    age = "-"
                    unreadable_homes.append(name)
                except OSError:
                    age = "-"
            else:
                age = "-"
            print(f"  {name:<{name_w}}  {label:<{tier_w}}  {status:<8}  {age}")

    if unreadable:
        print(f"\n  Unreadable launchers (run as root): {', '.join(unreadable)}")
    if unreadable_homes:
        print(f"\n  Last-activity hidden (home unreadable; run with sudo): {', '.join(sorted(set(unreadable_homes)))}")

    if tiers:
        print("\n  Tiers:")
        for tree in sorted(tiers):
            count = sum(1 for a in agents.values() if a["tree"] == tree)
            print(f"    {_tier_label(tree):<8} {tree:<45} {tiers[tree]}  [{count} agent(s)]")

    return 0


def build_agents_parser(subparsers) -> None:
    """Attach the ``agents`` subcommand to ``subparsers``."""
    agents_parser = subparsers.add_parser(
        "agents", help="Manage the multi-agent roster on this host (list/add)")
    agents_subparsers = agents_parser.add_subparsers(dest="agents_action")

    agents_subparsers.add_parser("list", help="List agents, tiers, run status, last activity")
    # `add` lands in v1 (see concept doc); parser stub withheld until implemented.

    agents_parser.set_defaults(func=cmd_agents_list)
