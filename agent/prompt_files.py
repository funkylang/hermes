"""Load prompt text from files in /home/agents/prompts with fallback to hardcoded defaults.

This allows external customization of agent behavior without modifying Python code.
Files are read once and cached; changes require a restart to take effect.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Directory where prompt text files are stored
PROMPTS_DIR = Path("/home/agents/prompts")

# Cache for loaded prompt texts
_prompt_cache: Dict[str, str] = {}
_lock = threading.Lock()


def get_prompt_text(filename: str, default: Optional[str] = None) -> Optional[str]:
    """Load prompt text from a file, falling back to the provided default.

    Args:
        filename: Name of the file to load (e.g., "identity/default_identity.md")
        default: Fallback text if file doesn't exist or can't be read

    Returns:
        The loaded text or the default fallback
    """
    global _prompt_cache

    # Check cache first
    with _lock:
        if filename in _prompt_cache:
            return _prompt_cache[filename]

    # Load from file
    prompt_file = PROMPTS_DIR / filename
    text: Optional[str] = None

    try:
        if prompt_file.exists():
            text = prompt_file.read_text(encoding="utf-8").strip()
            logger.debug(f"Loaded prompt from {prompt_file}")
        else:
            logger.warning(f"Prompt file not found: {prompt_file}, using default")
    except Exception as e:
        logger.error(f"Failed to read prompt file {prompt_file}: {e}, using default")

    # Fall back to default if needed
    if text is None:
        text = default

    # Cache the result (only if we have non-None text)
    if text is not None:
        with _lock:
            _prompt_cache[filename] = text

    return text


def get_identity_prompt(name: str, default: Optional[str] = None) -> Optional[str]:
    """Load an identity prompt from the identity/ directory."""
    return get_prompt_text(f"identity/{name}.md", default)


def get_behavior_prompt(name: str, default: Optional[str] = None) -> Optional[str]:
    """Load a behavior/guidance prompt from the behavior/ directory."""
    return get_prompt_text(f"behavior/{name}.md", default)
