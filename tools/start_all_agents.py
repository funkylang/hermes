#!/usr/bin/env python3
"""start-all-agents — one tmux session, one window per prod/stable agent.

Creates (or extends) a tmux session named "agents" on the CURRENT user's
tmux server. Each agent on the production or stable tier that is NOT
already running hermes gets a window named after the agent's configured
display name; in each window `hermes-agents start <user>` is typed (its
own sudo prompt appears there, as in the manual flow).

Agent discovery: /home/*/.local/bin/hermes launchers whose tier tree is
/home/agents/hermes-agent (prod) or /home/agents/hermes-agent-stable
(stable). Testing-tier and dev-tree agents are excluded.

Display names: read from the `name:` field under the `agent:` section of
each agent's config.yaml — requires root, so both name lookup and the
"already running" check run in ONE sudo-escalated helper call. Fallback
when unreadable: first letter capitalized.

Idempotent: existing session and windows are kept; missing agents are
added; all windows are reindexed alphabetically at the end. Unrecognized
windows are never destroyed. Agents whose hermes is already running
elsewhere are skipped entirely (no window, no start attempt).

Stdlib only — must run on any host python3 without a hermes installation.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

SESSION = "agents"
_HOME_ROOT = "/home"

# Tier trees that count as production/stable (see concepts/hermes-shared-deployment.txt).
_PROD_STABLE_TREES = {"/home/agents/hermes-agent", "/home/agents/hermes-agent-stable"}

_TIER_PATH_RE = re.compile(r"(/[\w./-]*hermes-agent(?:-\w+)?)/?")


def _tmux(*args: str, check: bool = True) -> tuple[int, str]:
    """Run a tmux command as the current user (their own server), return
    (returncode, stdout)."""
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if proc.returncode != 0 and check:
        sys.stderr.write(f"start-all-agents: tmux {' '.join(args)} failed:\n"
                         f"  {proc.stderr.strip()}\n")
        raise SystemExit(1)
    return (proc.returncode, proc.stdout)


def _window_names(session: str) -> list[tuple[int, str]]:
    """[(index, name), ...] for the session's windows."""
    _rc, out = _tmux("list-windows", "-t", session, "-F", "#{window_index} #{window_name}")
    return [(int(i), n) for i, n in (line.split(" ", 1) for line in out.splitlines())]


def _launcher_agent(launcher_path: str):
    """Return agent username if the launcher runs a prod/stable tier."""
    try:
        text = open(launcher_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = _TIER_PATH_RE.search(text)
    if not m:
        return None
    tree = m.group(1).rstrip("/")
    if tree not in _PROD_STABLE_TREES:
        return None
    return launcher_path.split("/")[2]  # /home/<user>/.local/bin/hermes


def _agents() -> list[str]:
    """Prod/stable tier agent usernames, alphabetical."""
    found = set()
    for entry in sorted(os.listdir(_HOME_ROOT)):
        launcher = os.path.join(_HOME_ROOT, entry, ".local", "bin", "hermes")
        if os.path.isfile(launcher):
            res = _launcher_agent(launcher)
            if res:
                found.add(res)
    return sorted(found)


# One-shot root helper: prints "user<TAB>display" per agent and "RUNNING<TAB>user"
# for every currently running hermes. Reuses hermes-agents' canonical roster
# scan; display names parsed from the `agent:` section of config.yaml.
_HELPER = r"""
import os, pwd, re

def running_agents():
    out = set()
    try:
        entries = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return out
    self_pid = str(os.getpid())
    for pid in entries:
        if pid == self_pid:
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        if "python" not in cmd:
            continue
        is_hermes = ("hermes_cli" in cmd) or ("hermes_bootstrap" in cmd)
        owner = None
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\x00")
            for item in env:
                if item.startswith(b"HERMES_HOME=/home/"):
                    is_hermes = True
                    parts = item.decode("utf-8", errors="replace").split("/")
                    if len(parts) >= 3:
                        owner = parts[2]
                    break
        except OSError:
            pass
        if not is_hermes:
            continue
        if not owner:
            try:
                st = os.stat(f"/proc/{pid}")
                owner = pwd.getpwuid(st.st_uid).pw_name
            except (OSError, KeyError):
                continue
        if owner:
            out.add(owner)
    return out


def display_name(user):
    path = f"/home/{user}/.hermes/config.yaml"
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return user[:1].upper() + user[1:]
    in_agent = False
    for line in text.splitlines():
        if re.match(r"^agent:\s*$", line):
            in_agent = True
            continue
        if in_agent:
            m = re.match(r"^\s+name:\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip("\"'")
            if line.strip() and not line.startswith((" ", "\t")):
                in_agent = False  # left the agent: block
    return user[:1].upper() + user[1:]


import sys
for user in sys.argv[1:]:
    print(f"{user}\t{display_name(user)}")
for user in sorted(running_agents()):
    print(f"RUNNING\t{user}")
"""


def _sudo_info(agents: list[str]) -> tuple[dict[str, str], set[str]]:
    """Via one sudo call: {username: display_name} and the running set."""
    proc = subprocess.run(["sudo", "python3", "-c", _HELPER, *agents],
                          capture_output=True, text=True)
    names: dict[str, str] = {}
    running: set[str] = set()
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0] == "RUNNING":
                running.add(parts[1])
            elif len(parts) == 2:
                names[parts[0]] = parts[1]
    for user in agents:  # fallback for anything the helper could not resolve
        names.setdefault(user, user[:1].upper() + user[1:])
    return (names, running)


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
    # Validate sudo credentials once at the start so subsequent sudo calls
    # in hermes-agents start don't prompt for a password repeatedly.
    proc = subprocess.run(["sudo", "-v"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("start-all-agents: sudo validation failed:\n")
        sys.stderr.write(proc.stderr.strip() + "\n")
        return 1

    agents = _agents()
    if not agents:
        print("start-all-agents: no prod/stable tier agents found — nothing to do.")
        return 0

    names, running = _sudo_info(agents)
    actionable = [a for a in agents if a not in running]
    skipped = [a for a in agents if a in running]
    if skipped:
        print(f"skipped (already running): {', '.join(names[a] for a in skipped)}")

    # Ensure the session exists (first window = first agent).
    started = []
    rc, _out = _tmux("has-session", "-t", SESSION, check=False)
    if rc != 0:
        _tmux("new-session", "-d", "-s", SESSION,
              "-n", names[actionable[0]],
              f"sh -c 'hermes-agents start {actionable[0]}; exec sh'")
        started.append(names[actionable[0]])
        actionable = actionable[1:]

    # Add missing agent windows. The window command runs hermes-agents as its
    # child: on success the exec chain takes over the pane, on refusal or
    # exit the shell prompt resumes (same as typing it in a manual tab).
    existing = {name for _, name in _window_names(SESSION)}
    for user in actionable:
        display = names[user]
        if display not in existing:
            _tmux("new-window", "-d", "-t", SESSION, "-n", display,
                  f"sh -c 'hermes-agents start {user}; exec sh'")
            started.append(display)
    if started:
        print(f"started agents in windows: {', '.join(started)}")

    # Reindex alphabetically.
    _reindex()
    final_names = [n for _, n in _window_names(SESSION)]
    if not started and final_names == sorted(final_names):
        print(f"nothing to do — session '{SESSION}' already has all agents in order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
