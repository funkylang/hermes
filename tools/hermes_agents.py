#!/usr/bin/env python3
"""hermes-agents — multi-agent host management, installed system-wide.

Subcommands:
  hermes-agents list   Roster of agents, tiers, run status, last activity.
  hermes-agents add    Create a new agent out-of-the-box. Requires root (sudo).
  hermes-agents start  Launch an agent's hermes in the CURRENT terminal.

Design and rulings: ~/workspace/hermes/concepts/agents-command.txt (Superhermes fork).
This script is the CANONICAL implementation, installed as /usr/local/bin/hermes-agents.
Stdlib only — must run on any host python3, without a hermes installation.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

_LIKE_TIME_FORMAT = "%Y-%m-%d %H:%M"

# Tier path pattern: hermes-agent, hermes-agent-testing, etc., under any root.
_TIER_PATH_RE = re.compile(r"(/[\w./-]*hermes-agent(?:-\w+)?)/?")

# FORK_VERSION is defined in the fork's hermes_cli/fork_version.py as a plain string.
_FORK_VERSION_DEF_RE = re.compile(r"^FORK_VERSION\s*=\s*[\"'](.+?)[\"']", re.MULTILINE)

# Tier path convention on this host (see concepts/hermes-shared-deployment.txt).
_TIER_PATHS = {
    "prod": "/home/agents/hermes-agent",
    "testing": "/home/agents/hermes-agent-testing",
}
_SHARED_SKILLS = Path("/home/agents/skills")

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


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

    # HERMES_HOME= assignments (sh style) or "os.environ[...HERMES_HOME...]\" context
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


def _running_agents() -> set[str]:
    """Agent names with an actively running hermes process.

    "running" means the agent actually runs hermes — not logged in, not merely
    sharing a tier with someone else who does. Attribution per PID:
      0. never our own process (no self-illusion when listing via hermes)
      1. process must be hermes itself: cmdline carries hermes_cli /
         hermes_bootstrap markers or HERMES_HOME in environ
      2. owner from HERMES_HOME=/home/<u>/.hermes in /proc/<pid>/environ
         (thin launchers; needs same-user or root)
      3. owner from os.stat(/proc/<pid>).st_uid -> passwd (works without root;
         only meaningful for #1 processes)
    """
    import pwd

    running: set[str] = set()
    self_pid = str(os.getpid())
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return running
    for entry in proc_entries:
        if not entry.isdigit() or entry == self_pid:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        if "python" not in cmd:
            continue
        owner = None
        is_hermes = ("hermes_cli" in cmd) or ("hermes_bootstrap" in cmd)
        try:
            with open(f"/proc/{entry}/environ", "rb") as f:
                env = f.read().split(b"\x00")
            for item in env:
                if item.startswith(b"HERMES_HOME=/home/"):
                    is_hermes = True
                    parts = item.decode("utf-8", errors="replace").split("/")
                    if len(parts) >= 3:
                        owner = parts[2]
                    break
        except OSError:
            pass  # unreadable environ (other user, no root): uid fallback below
        if not is_hermes:
            continue  # must actually be running hermes, not any python process
        if not owner:
            try:
                st = os.stat(f"/proc/{entry}")
                owner = pwd.getpwuid(st.st_uid).pw_name
            except (OSError, KeyError):
                continue
        if owner:
            running.add(owner)
    return running


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



def _escalate_sudo(command_args):
    """Re-exec via sudo if not root. Sets confirmation flag to prevent re-prompt."""
    try:
        euid = os.geteuid()
    except AttributeError:
        print("Cannot verify privileges.", file=sys.stderr)
        return False
    
    if euid == 0:
        return True
    
    exe = None
    for candidate in (os.path.realpath(sys.argv[0]), "/usr/local/bin/hermes-agents"):
        if candidate and os.path.exists(candidate):
            exe = candidate
            break
    
    if exe is None:
        print("Cannot locate hermes-agents binary.", file=sys.stderr)
        return False
    
    sys.stderr.write(f"elevating (sudo {exe} {' '.join(command_args)})\n")
    env = os.environ.copy()
    env["HERMES_AGENTS_CONFIRMED"] = "1"
    os.execve("/usr/bin/sudo", ["sudo", exe] + command_args, env)
    return False


def cmd_agents_list(_args=None) -> int:
    home_root = Path("/home")
    if not home_root.is_dir():
        print("hermes-agents list: no /home on this host — nothing to report.",
              file=sys.stderr)
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

    # 3. Run status per agent (active hermes process, attributed per PID).
    running = _running_agents()
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
            status = "running" if name in running else "idle"
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
        print(f"\n  Last-activity hidden (home unreadable; run with sudo): "
              f"{', '.join(sorted(set(unreadable_homes)))}")

    if tiers:
        print("\n  Tiers:")
        for tree in sorted(tiers):
            count = sum(1 for a in agents.values() if a["tree"] == tree)
            unit = "agent" if count == 1 else "agents"
            print(f"    {_tier_label(tree):<8} {tree}  [{count} {unit}]")
            print(f"             {tiers[tree]}")

    return 0


def _launcher_script(name: str, tree: str) -> str:
    return (
        "#!/bin/sh\n"
        f"export HERMES_HOME=/home/{name}/.hermes\n"
        f'exec {tree}/hermes "$@"\n'
    )


def _set_config_name(config_path: Path, display_name: str) -> bool:
    """Set the agent identity fields (transition: BOTH locations).

    Writes top-level `name:` (legacy trees) and `agent.name:` (new canonical
    location) so the display name works regardless of which code version runs.
    Returns True when both are written or already correct.
    """
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Top-level `name:`
    for i, line in enumerate(lines):
        if line.startswith("name:"):
            if line.rstrip("\n") != f"name: {display_name}":
                lines[i] = f"name: {display_name}\n"
            break
    else:
        if text and not text.endswith("\n"):
            lines.append("\n")
        lines.append(f"name: {display_name}\n")

    # agent.name under the `agent:` block (only when the block exists)
    agent_idx = next((i for i, l in enumerate(lines) if l.rstrip() == "agent:"), None)
    if agent_idx is not None:
        j = agent_idx + 1
        found = False
        while j < len(lines):
            l = lines[j]
            if not l.startswith("  ") and l.strip():  # left the agent block
                break
            if l.startswith("  name:") or l.startswith("\tname:"):
                lines[j] = f"  name: {display_name}\n"
                found = True
            j += 1
        if not found:
            lines.insert(agent_idx + 1, f"  name: {display_name}\n")

    config_path.write_text("".join(lines), encoding="utf-8")
    return True


def _verify_agent(name: str) -> tuple[bool, str]:
    """Run `hermes --version` as the new user via their launcher. (ok, detail)."""
    import pwd

    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        return (False, "user does not exist")

    def drop_privs():
        os.setgid(pw.pw_gid)
        os.initgroups(name, pw.pw_gid)
        os.setuid(pw.pw_uid)

    env = {
        "HOME": pw.pw_dir,
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    try:
        proc = subprocess.run(
            [f"/home/{name}/.local/bin/hermes", "--version"],
            preexec_fn=drop_privs, cwd=pw.pw_dir, env=env,
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (False, f"launcher failed to execute: {exc}")
    out = proc.stdout.strip() or proc.stderr.strip()
    if proc.returncode != 0:
        return (False, f"exit {proc.returncode}: {out[:200]}")
    first_line = out.splitlines()[0] if out else "(no output)"
    return (True, first_line)


def _resolve_agent(name: str) -> tuple[str | None, str]:
    """Resolve an agent identifier to an OS username.

    Accepts the OS username directly (if it has a launcher), or the display
    name configured in an agent's config.yaml (top-level `name:` or
    `agent.name:`). Returns (username, reason); username is None on failure.
    """
    import pwd

    # Fast path: direct username with an existing launcher.
    # Try both the given name and lowercase (useradd creates lowercase dirs).
    for candidate in (name, name.lower()):
        if Path(f"/home/{candidate}/.local/bin/hermes").exists():
            try:
                pwd.getpwnam(candidate)
                return (candidate, "")
            except KeyError:
                pass

    # Slow path: display-name match through each agent's config.yaml.
    want = name.strip().lower()
    home_root = Path("/home")
    if not home_root.is_dir():
        return (None, "no /home on this host")
    for entry in sorted(home_root.iterdir()):
        cfg = entry / ".hermes" / "config.yaml"
        try:
            if not cfg.exists():
                continue
            text = cfg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = (re.match(r"^name:\s*(.+?)\s*$", line)
                 or re.match(r"^\s+name:\s*(.+?)\s*$", line))
            if not m:
                continue
            val = m.group(1).strip().strip("\"'")
            if val.lower() == want:
                return (entry.name, "")
    return (None, f"no agent '{name}' (no matching user launcher or config name)")


def cmd_agents_start(args) -> int:
    name = args.name

    user, why = _resolve_agent(name)
    if not user:
        print(f"start: {why}", file=sys.stderr)
        return 1

    launcher = Path(f"/home/{user}/.local/bin/hermes")
    if not launcher.exists():
        print(f"start: no launcher for '{user}' at {launcher}", file=sys.stderr)
        return 1

    # Handing THIS terminal over to the agent requires a real interactive
    # session on both ends — no pipes, no background jobs.
    if not (os.isatty(sys.stdin.fileno()) and os.isatty(sys.stdout.fileno())):
        print("start: run this from an interactive terminal (e.g. your SSH "
              "session); no TTY is attached here.", file=sys.stderr)
        return 1

    try:
        euid = os.geteuid()
    except AttributeError:
        print("Cannot verify privileges on this platform; refusing to continue.",
              file=sys.stderr)
        return 1

    if euid != 0:
        # Self-escalate in place: sudo prompts for the password on THIS
        # terminal, then continues with root. Operators type
        # `hermes-agents start X` without sudo.
        exe = None
        for candidate in (os.path.realpath(sys.argv[0]),
                          "/usr/local/bin/hermes-agents"):
            if candidate and os.path.exists(candidate):
                exe = candidate
                break
        if exe is None:
            print("start: not running as root, and cannot locate the "
                  "hermes-agents binary to re-exec via sudo.", file=sys.stderr)
            return 1
        sys.stderr.write(f"start: elevating (sudo {exe} start {name})\n")
        os.execv("/usr/bin/sudo", ["sudo", exe, "start", name])
        # execv only returns on failure.

    # Refuse to create a duplicate live session.
    if user in _running_agents():
        print(f"'{user}' is already running hermes — not starting a duplicate.")
        return 1

    print(f"Starting hermes as '{user}' in this terminal...")
    print("You are now in that agent's session; when hermes exits, "
          "your own shell resumes.")
    # Hand the controlling terminal over: exec replaces this process with the
    # agent's hermes under a login environment — equivalent to a manual
    # `sudo su - <user>` followed by running the agent's launcher.
    os.execv("/usr/bin/su",
             ["su", "-", "-c", shlex.quote(str(launcher)), user])
    return 1  # unreachable; execv replaces the process


def cmd_agents_add(args) -> int:
    # Accept mixed case - normalize to lowercase for username
    username = args.name.lower()
    display_name = args.name  # Preserve original casing for config.yaml

    if not _NAME_RE.match(username):
        print(f"Invalid agent name '{username}' — use lowercase letters, digits, "
              f"hyphens; must start with a letter.", file=sys.stderr)
        return 1

    name = username

    # Pass args.name (original case) so the agent's display name survives the
    # sudo round-trip; the second run re-derives the lowercase username itself.
    if not _escalate_sudo(["add", args.name, "--tier", args.tier]):
        # execv will not return on success; we failed to escalate
        print("hermes-agents add requires root privileges.", file=sys.stderr)
        return 1

    tree = _TIER_PATHS[args.tier]
    if not Path(tree, "hermes").exists():
        print(f"Tier '{args.tier}' not found on this host: {tree}", file=sys.stderr)
        return 1

    # display_name already set from args.name above - preserve original casing
    actions = []

    # 1. Unix user (idempotent). Intentionally NOT a sudoer — agents need root
    #    only for install/deployment work, which the managing operator decides.
    import pwd as _pwd
    try:
        _pwd.getpwnam(name)
        actions.append(f"user '{name}' already exists — kept")
    except KeyError:
        proc = subprocess.run(
            ["useradd", "-m", "-g", "agents", "-c", display_name, name],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"useradd failed: {proc.stderr.strip()}", file=sys.stderr)
            return 1
        actions.append(f"created user '{name}' (group agents, home /home/{name}; "
                       f"not a sudoer)")

    # 2. Launcher.
    launcher_dir = Path(f"/home/{name}/.local/bin")
    launcher = launcher_dir / "hermes"
    content = _launcher_script(name, tree)
    if launcher.exists():
        old = launcher.read_text(encoding="utf-8", errors="replace")
        if old == content:
            actions.append("launcher already correct — kept")
        else:
            launcher.write_text(content, encoding="utf-8")
            actions.append(f"launcher updated (tier: {args.tier})")
    else:
        launcher_dir.mkdir(parents=True, exist_ok=True)
        launcher.write_text(content, encoding="utf-8")
        launcher.chmod(0o755)
        actions.append(f"created launcher {launcher}")

    # 3. Config from template or existing agent.
    hermes_home = Path(f"/home/{name}/.hermes")
    config_path = hermes_home / "config.yaml"
    if config_path.exists():
        actions.append("config.yaml already exists — kept")
    else:
        hermes_home.mkdir(parents=True, exist_ok=True)
        
        # Try template files first
        template_dir = Path("/home/agents/templates")
        template_config = template_dir / "config.yaml"
        
        if template_config.exists():
            try:
                config_text = template_config.read_text(encoding="utf-8")
                # Substitute placeholders
                config_text = config_text.replace("TEMPLATE_AGENT_NAME", display_name)
                config_text = config_text.replace("TEMPLATE_AGENT_USERNAME", name)
                config_path.write_text(config_text, encoding="utf-8")
                actions.append(f"config.yaml created from template (name: {display_name})")
            except OSError as exc:
                print(f"Failed to read template or write config: {exc}", file=sys.stderr)
                return 1
        else:
            # Fall back to copying from existing agent
            src = _resolve_template(name, args.tier, args.from_agent)
            if src is None:
                print(f"No usable config.yaml found (no template and no agent source). "
                      f"Pass --from <agent> explicitly or create /home/agents/templates/config.yaml.", 
                      file=sys.stderr)
                return 1
            try:
                config_path.write_text(src[1], encoding="utf-8")
            except OSError as exc:
                print(f"Failed to write config: {exc}", file=sys.stderr)
                return 1
            _set_config_name(config_path, display_name)
            actions.append(f"config.yaml created from '{src[0]}' "
                           f"(name: {display_name})")

    # 3.5 SOUL.md (agent identity) - REQUIRED for correct agent name/identity.
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        actions.append("SOUL.md already exists — kept")
    else:
        template_dir = Path("/home/agents/templates")
        template_soul = template_dir / "SOUL.md"
        if template_soul.exists():
            try:
                soul_text = template_soul.read_text(encoding="utf-8")
                # Substitute placeholders with agent's identity
                soul_text = soul_text.replace("TEMPLATE_AGENT_NAME", display_name)
                soul_text = soul_text.replace("TEMPLATE_AGENT_USERNAME", name)
                soul_path.write_text(soul_text, encoding="utf-8")
                actions.append(f"SOUL.md created from template (identity: {display_name})")
            except OSError as exc:
                print(f"Failed to read SOUL.md template or write file: {exc}", file=sys.stderr)
                return 1
        else:
            # No template - use a minimal default identity
            soul_text = f"You are {display_name} Superhermes.\n\nYour username is {name}. You are part of the Superhermes family of agents.\n"
            soul_path.write_text(soul_text, encoding="utf-8")
            actions.append(f"SOUL.md created with default identity ({display_name})")

    # 4. Shared skills symlink (skip over a pre-existing dir, e.g. legacy copies).
    skills_link = hermes_home / "skills"
    if not skills_link.exists() and not skills_link.is_symlink():
        if _SHARED_SKILLS.is_dir():
            skills_link.symlink_to(_SHARED_SKILLS)
            actions.append(f"symlinked skills → {_SHARED_SKILLS}")
        else:
            actions.append(f"NOTE: shared skills dir {_SHARED_SKILLS} not found — skipped")

    # 5. Tree link inside HERMES_HOME (matches athena's proven layout).
    tree_link = hermes_home / "hermes-agent"
    if not tree_link.exists() and not tree_link.is_symlink():
        tree_link.symlink_to(tree)
        actions.append(f"symlinked hermes-agent → {tree}")


    # 5.5. Create .bashrc from template
    bashrc_path = Path(f"/home/{name}/.bashrc")
    if not bashrc_path.exists():
        template_bashrc = Path("/home/agents/templates/.bashrc")
        if template_bashrc.exists():
            try:
                bashrc_text = template_bashrc.read_text(encoding="utf-8")
                # Substitute placeholders
                bashrc_text = bashrc_text.replace("TEMPLATE_AGENT_USERNAME", name)
                bashrc_path.write_text(bashrc_text, encoding="utf-8")
                actions.append(".bashrc created from template")
            except OSError as exc:
                print(f"Warning: Failed to create .bashrc: {exc}", file=sys.stderr)

    # 6. Fix ownership: ensure all created files belong to the agent (not root)
    # The agent needs write access to .hermes and launcher dirs
    for path in [launcher_dir, hermes_home]:
        if path.exists():
            proc = subprocess.run(
                ["chown", "-R", f"{name}:agents", str(path)],
                capture_output=True, text=True
            )
            if proc.returncode != 0:
                print(f"Warning: Could not chown {path}: {proc.stderr.strip()}", 
                      file=sys.stderr)

    # 7. Print summary
    print(f"Agent '{name}' ({args.tier} tier):")
    for a in actions:
        print(f"  - {a}")
    
    # 8. Verification: run their launcher as them.
    ok, detail = _verify_agent(name)
    if ok:
        print(f"  VERIFIED out-of-the-box: hermes --version → {detail}")
    else:
        print(f"  VERIFICATION FAILED: {detail}", file=sys.stderr)
        print("The agent may need manual inspection before use.", file=sys.stderr)

    print("\n  Next steps:")
    print("  - Terminal window appears on next desktop restart (automatic).")
    print("  - Optional theme: copy ~/.themes/Mint-X-Classic and set the")
    print("    org.mate.interface gtk-theme key in their desktop session.")
    return 0 if ok else 1


def _resolve_template(skip_name: str, tier: str, explicit: str | None):
    """Find (agent_name, config_text) to seed from. None if unusable."""
    home_root = Path("/home")
    candidates = []
    for entry in sorted(home_root.iterdir()):
        if not entry.is_dir() or entry.name == skip_name:
            continue
        launcher = entry / ".local" / "bin" / "hermes"
        cfg = entry / ".hermes" / "config.yaml"
        if not (launcher.exists() and cfg.exists()):
            continue
        tree, _home = _read_launcher_meta(launcher)
        if explicit:
            if entry.name != explicit:
                continue
        else:
            expected = _TIER_PATHS[tier].rstrip("/")
            if (tree or "").rstrip("/") != expected:
                continue
        try:
            text = cfg.read_text(encoding="utf-8")
        except OSError:
            continue
        candidates.append((entry.name, tree, text))

    if explicit:
        match = [c for c in candidates if c[0] == explicit]
        if not match:
            print(f"--from '{explicit}' has no readable launcher + config.yaml.",
                  file=sys.stderr)
            return None
        name, _tree, text = match[0]
        return (name, text)

    for cand in candidates:  # already alphabetical; first on the tier wins
        return (cand[0], cand[2])
    return None

def cmd_agents_delete(args) -> int:
    """Delete an agent and all associated files."""
    # Accept mixed case - normalize to lowercase for username (same as add)
    name = args.name.lower()

    if not _NAME_RE.match(name):
        print(f"Invalid agent name '{name}' — use lowercase letters, digits, "
              f"hyphens; must start with a letter.", file=sys.stderr)
        return 1

    # Confirm deletion - user must type "yes" (skip if already confirmed via escalation or running as root)
    # When we escalate to sudo and come back as root, skip the prompt - user already confirmed
    try:
        is_root = os.geteuid() == 0
    except AttributeError:
        is_root = False
    
    # Check if user exists FIRST (before asking for confirmation)
    import pwd
    user_exists = True
    try:
        pwd.getpwnam(name)
    except KeyError:
        user_exists = False
        print(f"User '{name}' does not exist, nothing to delete.")
        return 0
    
    if not is_root:
        print(f"You are about to delete agent '{name}'.")
        print("This will remove:")
        print(f"  - The user account '{name}'")
        print(f"  - Their home directory /home/{name}")
        print()
        answer = input("Type 'yes' to confirm: ")
        if answer.strip() != "yes":
            print("Deletion cancelled.")
            return 0
    
    # Now escalate and continue as root (if not already root)
    if not _escalate_sudo(["delete", name]):
        # execv will not return on success; we failed to escalate
        print("hermes-agents delete requires root privileges.", file=sys.stderr)
        return 1

    # Kill any running processes owned by this user
    try:
        subprocess.run(["pkill", "-u", name], capture_output=True)
        print(f"Killed any running processes for '{name}'.")
    except FileNotFoundError:
        pass

    # Remove the user account and home directory
    proc = subprocess.run(
        ["userdel", "-r", name],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"userdel failed: {proc.stderr.strip()}", file=sys.stderr)
        # Try without removing home directory
        proc2 = subprocess.run(
            ["userdel", name],
            capture_output=True, text=True,
        )
        if proc2.returncode != 0:
            print(f"userdel failed: {proc2.stderr.strip()}", file=sys.stderr)
            return 1

    # Clean up any remaining launcher files
    launcher = Path(f"/home/{name}/.local/bin/hermes")
    if launcher.exists():
        try:
            launcher.unlink()
            print(f"Removed launcher at {launcher}")
        except OSError:
            pass

    # Clean up any remaining home directory contents
    home_dir = Path(f"/home/{name}")
    if home_dir.exists():
        import shutil
        try:
            shutil.rmtree(home_dir)
            print(f"Removed home directory {home_dir}")
        except OSError as exc:
            print(f"Warning: could not fully remove {home_dir}: {exc}", file=sys.stderr)

    print(f"Agent '{name}' deleted successfully.")
    return 0



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-agents",
        description=(
            "Multi-agent host management — the agent roster on this machine:\n"
            "who is here, which tier they run, whether hermes is actively running,\n"
            "and last activity. `hermes agents <command>` delegates to this tool."),
        epilog="""subcommands:

  list
      Show every agent on this host: tier, active hermes processes, and when
      each was last active. Also prints the tiers themselves with their build
      banners. No privileges needed; run with sudo for full visibility of
      other agents' process state.
      Options: none.

  add NAME [--tier {prod,testing}] [--from AGENT]
      Create a new agent out-of-the-box. Requires root (sudo).
      Idempotent: re-running keeps existing pieces and reports what was done.
        NAME                 agent name: lowercase letters, digits, hyphens;
                             must start with a letter.
        --tier               which tier the agent runs on:
                               prod       /home/agents/hermes-agent (default)
                               testing    /home/agents/hermes-agent-testing
        --from AGENT         template agent whose config.yaml is copied as
                             starting point; default: first agent (alphabetical)
                             on the target tier.
      Creates: user in group 'agents' (intentionally NOT a sudoer), launcher,
      seeded config.yaml with the name field set, shared skills symlink —
      then verifies by running `hermes --version` as the new user.

  start NAME
      Launch an agent's hermes in the CURRENT terminal. Typed from your SSH
      session, that terminal becomes the agent's interactive hermes session;
      when hermes exits, your own shell resumes. Accepts the OS username or
      the configured display name. Refuses to start a second live session
      for an agent already running. Escalates via sudo if needed (the
      password prompt appears in this terminal).

examples:

  hermes-agents list
      The agent roster, no privileges needed.

  sudo hermes-agents add cronus
      New agent 'cronus' on the production tier, config seeded from an
      existing prod peer.

  sudo hermes-agents add probe --tier testing --from sisyphus
      A testing-tier agent cloned from sisyphus's configuration.

  hermes-agents start cronus
      Bring 'cronus's hermes up in this terminal — one command instead of
      `sudo su - cronus` plus running its launcher by hand.

  delete NAME
      Remove an agent and all their associated files (user account, home directory).
      Requires root (sudo). Prompts for confirmation before proceeding.
        NAME                 agent name to delete (lowercase: letters, digits, hyphens)

examples:

  sudo hermes-agents delete fortuna
      Permanently remove agent 'fortuna' and all their files.

notes:

  - 'running' means the agent has an active hermes process — not merely a
    login session on this machine.
  - 'delete' is destructive: it removes the user account AND home directory.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", metavar="{list,add,delete,start}")

    p_list = sub.add_parser(
        "list", help="List agents, tiers, run status, last activity (no privileges needed)",
        description="Show every agent on this host: tier, active hermes processes, "
                    "and when they were last active.")
    p_list.set_defaults(func=cmd_agents_list)

    p_add = sub.add_parser(
        "add", help="Create a new agent out-of-the-box (requires root)",
        description="Create user + launcher + seeded config + shared skills for a new "
                    "agent, then verify it runs. Idempotent: re-running keeps existing "
                    "pieces and reports what was done.")
    p_add.add_argument("name", help="Agent name (lowercase: letters, digits, hyphens)")
    p_add.add_argument(
        "--tier", choices=sorted(_TIER_PATHS), default="prod",
        help="Tier to run on (default: prod)")
    p_add.add_argument(
        "--from", dest="from_agent", metavar="AGENT",
        help="Template agent whose config.yaml to copy "
             "(default: first agent on the target tier)")
    p_add.set_defaults(func=cmd_agents_add)

    p_delete = sub.add_parser(
        "delete", help="Remove an agent and their files (requires root)",
        description="Permanently delete an agent: removes their user account, home directory, "
                    "and all associated files. Prompts for confirmation before proceeding.")
    p_delete.add_argument("name", help="Agent username to delete (lowercase: letters, digits, hyphens)")
    p_delete.set_defaults(func=cmd_agents_delete)

    p_start = sub.add_parser(
        "start", help="Launch an agent's hermes in the current terminal",
        description="Hands this terminal over to the agent's hermes session "
                    "(sudo self-escalates; password prompt here). Refuses to "
                    "duplicate a session that is already running.")
    p_start.add_argument("name", help="Agent username or configured display name")
    p_start.set_defaults(func=cmd_agents_start)

    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        # No subcommand given: show the full help instead of a bare usage error.
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
