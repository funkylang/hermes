"""Cross-platform detection helpers.

Relocated from hermes_cli.gateway (removed with the messaging daemon): surviving
callers need only OS/service-capability detection, not the gateway runtime.
Self-contained: no dependency on the removed service-management internals.
"""

from __future__ import annotations

import shutil
import sys

from hermes_constants import is_container, is_termux, is_wsl


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def _systemd_running_probe(system: bool = False) -> bool:
    """True when the requested systemd scope reports a live init (PID 1)."""
    import subprocess

    cmd = ["sudo", "systemctl"] if system else ["systemctl"]
    try:
        result = subprocess.run(
            [*cmd, "is-system-running"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip().lower() in {"running", "degraded", "starting", "initializing"}


def supports_systemd_services() -> bool:
    """Whether systemd user/system services are usable on this host."""
    if not is_linux() or is_termux() or shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _systemd_running_probe(system=True)
    if is_container():
        # A container whose init is systemd (nspawn, some k8s pods) behaves like a host.
        return _systemd_running_probe(system=False) or _systemd_running_probe(system=True)
    return True
