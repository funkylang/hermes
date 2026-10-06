"""Tests for hermes_cli.input_hook — config loading and script execution."""

from __future__ import annotations

import os
import tempfile
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.input_hook import (
    DEFAULT_INPUT_HOOK_TIMEOUT,
    _input_hook_config,
    get_input_hook_script_path,
    get_input_hook_timeout,
    build_metadata,
    format_metadata_block,
    run_input_hook,
)


# ── Config loading tests ──────────────────────────────────────────────

class TestInputHookConfig:
    def test_empty_config_returns_empty_dict(self):
        """When config has no input_hook section, return empty dict."""
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _input_hook_config() == {}

    def test_config_with_input_hook_section(self):
        """Read the input_hook section from config."""
        fake_config = {
            "input_hook": {
                "script": "/path/to/script.sh",
                "timeout": 60,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=fake_config):
            result = _input_hook_config()
            assert result == {"script": "/path/to/script.sh", "timeout": 60}

    def test_non_dict_input_hook_returns_empty(self):
        """Handle non-dict values gracefully."""
        with patch("hermes_cli.config.load_config", return_value={"input_hook": "not-a-dict"}):
            assert _input_hook_config() == {}


# ── Script path resolution tests ─────────────────────────────────────

class TestGetScriptPath:
    def test_env_var_takes_precedence(self):
        """HERMES_INPUT_HOOK env var overrides config."""
        with patch.dict(os.environ, {"HERMES_INPUT_HOOK": "/env/script.sh"}), \
             patch("hermes_cli.config.load_config", return_value={
                 "input_hook": {"script": "/config/script.sh"}
             }):
            path = get_input_hook_script_path()
            assert path == "/env/script.sh"

    def test_reads_from_config_when_no_env(self):
        """Read script path from config.yaml when env var not set."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("hermes_cli.config.load_config", return_value={
                 "input_hook": {"script": "~/.hermes/hooks/script.sh"}
             }):
            path = get_input_hook_script_path()
            assert path is not None
            # Check tilde expansion worked (path contains home dir)
            assert "~" not in path

    def test_returns_none_when_not_configured(self):
        """Return None when no script configured."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("hermes_cli.config.load_config", return_value={}):
            path = get_input_hook_script_path()
            assert path is None


# ── Timeout tests ─────────────────────────────────────────────────────

class TestGetTimeout:
    def test_default_timeout_when_no_config(self):
        """Return default timeout when not configured."""
        with patch("hermes_cli.config.load_config", return_value={}):
            assert get_input_hook_timeout() == DEFAULT_INPUT_HOOK_TIMEOUT

    def test_reads_custom_timeout_from_config(self):
        """Read custom timeout from config."""
        with patch("hermes_cli.config.load_config", return_value={
            "input_hook": {"timeout": 120}
        }):
            assert get_input_hook_timeout() == 120

    def test_minimum_one_second(self):
        """Enforce minimum of 1 second timeout."""
        with patch("hermes_cli.config.load_config", return_value={
            "input_hook": {"timeout": 0}
        }):
            # max(1, 0) = 1
            assert get_input_hook_timeout() == 1


# ── Metadata building tests ───────────────────────────────────────────

class TestBuildMetadata:
    def test_builds_complete_metadata(self):
        """Include all provided fields in metadata."""
        meta = build_metadata(
            context_used=5000,
            context_total=262144,
            total_messages=10,
            finish_reason="stop",
            session_id="sess-abc-123",
        )
        assert meta == {
            "context_used": "5000",
            "context_size": "262144",
            "total_messages": "10",
            "finish_reason": "stop",
            "session_id": "sess-abc-123",
        }

    def test_skips_none_values(self):
        """Omit fields that are None."""
        meta = build_metadata(
            context_used=None,
            context_total=262144,
            total_messages=None,
            finish_reason="stop",
            session_id="sess-xyz-456",
        )
        assert "context_used" not in meta
        assert "total_messages" not in meta
        assert meta == {
            "context_size": "262144",
            "finish_reason": "stop",
            "session_id": "sess-xyz-456",
        }

    def test_empty_when_all_none(self):
        """Return only session_id when all other values are None."""
        meta = build_metadata(None, None, None, None, session_id="sess-123")
        assert meta == {"session_id": "sess-123"}

    def test_session_id_empty_string_skipped(self):
        """Omit session_id when empty string."""
        meta = build_metadata(5000, 262144, 10, "stop", session_id="")
        assert "session_id" not in meta


# ── Metadata formatting tests ─────────────────────────────────────────

class TestFormatMetadataBlock:
    def test_formats_key_value_pairs(self):
        """Format metadata dict as multi-line key=value string."""
        meta = {"key1": "value1", "key2": "value2"}
        block = format_metadata_block(meta)
        lines = block.split("\n")
        assert "key1=value1" in lines
        assert "key2=value2" in lines

    def test_empty_dict_returns_empty_string(self):
        """Return empty string for empty metadata."""
        assert format_metadata_block({}) == ""


# ── Script execution tests ────────────────────────────────────────────

class TestRunInputHook:
    def test_returns_none_when_not_configured(self):
        """Don't run anything when script not configured."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("hermes_cli.config.load_config", return_value={}):
            result = run_input_hook("test reasoning", "test content", "")
            assert result is None

    def test_returns_none_for_missing_script(self):
        """Log warning and return None for missing script file."""
        with patch.dict(os.environ, {"HERMES_INPUT_HOOK": "/nonexistent/script.sh"}):
            result = run_input_hook("reasoning", "content", "")
            assert result is None

    @pytest.fixture
    def temp_script(self):
        """Create a temporary test script."""
        script_content = """#!/bin/bash
# Test script that echoes back the content
echo "$2"
exit 0
"""
        fd, path = tempfile.mkstemp(suffix='.sh')
        try:
            os.write(fd, script_content.encode())
        finally:
            os.close(fd)
        os.chmod(path, 0o755)
        yield path
        os.unlink(path)

    def test_successful_script_execution(self, temp_script):
        """Run script and return its stdout."""
        with patch.dict(os.environ, {"HERMES_INPUT_HOOK": temp_script}):
            result = run_input_hook("reasoning", "Hello World", "")
            assert result == "Hello World"

    def test_script_with_nonzero_exit_returns_none(self):
        """Return None when script exits non-zero."""
        fd, path = tempfile.mkstemp(suffix='.sh')
        try:
            os.write(fd, b"#!/bin/bash\nexit 1\n")
        finally:
            os.close(fd)
        os.chmod(path, 0o755)
        with patch.dict(os.environ, {"HERMES_INPUT_HOOK": path}):
            result = run_input_hook("reasoning", "content", "")
            assert result is None
        os.unlink(path)

    def test_script_with_empty_stdout_returns_none(self):
        """Return None when script exits 0 but produces empty stdout."""
        fd, path = tempfile.mkstemp(suffix='.sh')
        try:
            os.write(fd, b'#!/bin/bash\necho "debug" >&2\nexit 0\n')
        finally:
            os.close(fd)
        os.chmod(path, 0o755)
        with patch.dict(os.environ, {"HERMES_INPUT_HOOK": path}):
            result = run_input_hook("reasoning", "content", "")
            assert result is None
        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
