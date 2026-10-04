"""Input hook execution: reads config, runs user script, parses decision.

The input_hook is a user-configured script that runs after each assistant turn.
It can decide whether to auto-reply (exit 0 + non-empty stdout) or prompt human
(exit non-zero or empty stdout). This enables agent control systems where external
scripts manage orchestration while LLMs handle creative tasks.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default timeout for input_hook scripts (seconds)
DEFAULT_INPUT_HOOK_TIMEOUT = 30


def _input_hook_config() -> dict:
    """Read input_hook configuration from config.yaml."""
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get("input_hook") or {}
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("input_hook config read failed: %s", exc)
        return {}


def get_input_hook_script_path() -> Optional[str]:
    """Return the input_hook script path from config, or None if not configured.

    Supports both:
    1. config.yaml input_hook.script
    2. HERMES_INPUT_HOOK environment variable (for testing/quick setup)
    """
    # Environment variable takes precedence for testing
    env_path = os.environ.get("HERMES_INPUT_HOOK", "").strip()
    if env_path:
        return os.path.expanduser(env_path)

    # Read from config
    config = _input_hook_config()
    script = str(config.get("script", "") or "").strip()
    if not script:
        return None

    # Expand ~ in path
    return os.path.expanduser(script)


def get_input_hook_timeout() -> int:
    """Return the input_hook timeout from config (default 30s, minimum 1s)."""
    config = _input_hook_config()
    try:
        return max(1, int(config.get("timeout", DEFAULT_INPUT_HOOK_TIMEOUT)))
    except Exception:
        return DEFAULT_INPUT_HOOK_TIMEOUT


def build_metadata(
    context_used: Optional[int],
    context_total: Optional[int],
    total_messages: Optional[int],
    finish_reason: Optional[str],
) -> dict[str, str]:
    """Build the metadata dict passed to the script as a multi-line key=value block.

    Args:
        context_used: Tokens used in current context (or None if unavailable)
        context_total: Total context window size (or None if unavailable)
        total_messages: Number of messages in conversation history
        finish_reason: The last API call's finish_reason (stop, tool_calls, etc.)

    Returns:
        A dict that will be formatted as multi-line key=value pairs
    """
    meta = {}
    if context_used is not None:
        meta["context_used"] = str(context_used)
    if context_total is not None:
        meta["context_size"] = str(context_total)
    if total_messages is not None:
        meta["total_messages"] = str(total_messages)
    if finish_reason:
        meta["finish_reason"] = finish_reason
    return meta


def format_metadata_block(metadata: dict[str, str]) -> str:
    """Format metadata dict as a multi-line key=value string for script argument."""
    return "\n".join(f"{k}={v}" for k, v in metadata.items())


def run_input_hook(
    reasoning: str,
    content: str,
    metadata_block: str,
) -> Optional[str]:
    """Run the input_hook script and return the auto-reply text if it should be used.

    Decision logic:
    - exit_code == 0 AND stdout non-empty → auto-reply with stdout content
    - exit_code != 0 OR stdout empty     → prompt human (return None)

    Args:
        reasoning: The assistant's reasoning if available (may be empty)
        content: The assistant's final response text
        metadata_block: Multi-line key=value metadata string

    Returns:
        The auto-reply text to queue as the next user message, or None to prompt human
    """
    script_path = get_input_hook_script_path()
    if not script_path:
        return None  # Not configured

    # Verify script exists
    path = Path(script_path)
    if not path.exists():
        logger.warning("input_hook script not found: %s", script_path)
        print(f"⚠️  input_hook script not found: {script_path}", file=sys.stderr)
        return None

    timeout = get_input_hook_timeout()

    # Invoke the script with our three arguments
    try:
        proc = subprocess.run(
            [script_path, reasoning, content, metadata_block],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,  # We handle the exit code ourselves
        )

        # Display stderr (diagnostics) if present — label first, then content on fresh lines
        if proc.stderr.strip():
            print("⚠️  Input hook warning/error:", file=sys.stderr)
            # Add a blank line, then print the stderr content with leading space
            print(file=sys.stderr)
            for line in proc.stderr.rstrip("\n").splitlines():
                print(f"  {line}", file=sys.stderr)

        # Apply decision logic
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.warning("input_hook script timed out after %ds", timeout)
        print(f"⚠️  input_hook script timed out after {timeout}s", file=sys.stderr)

    except FileNotFoundError as e:
        logger.error("input_hook script not executable: %s", e)
        print(f"⚠️  input_hook script error: {e}", file=sys.stderr)

    except Exception as e:
        logger.error("input_hook execution failed: %s", e, exc_info=True)
        print(f"⚠️  input_hook execution error: {e}", file=sys.stderr)

    # Prompt human on any failure or non-zero exit
    return None


# Import sys at the bottom to avoid circular imports at module load time
import sys
