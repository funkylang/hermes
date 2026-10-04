"""Input hook integration for the interactive CLI.

Provides the post-turn hook that runs the user-configured input_hook script
after each assistant turn completes. The script can decide whether to
auto-reply or prompt human, enabling agent control systems.
"""

from __future__ import annotations

import logging
import sys


class CLIInputHookMixin:
    """Post-turn hook for running user-configured input_hook scripts."""

    def _maybe_run_input_hook(self) -> None:
        """Run the configured input_hook script after a completed turn.

        If the script exits 0 with non-empty stdout, that text is queued as
        the next user message (auto-reply). Otherwise, nothing is queued and
        the REPL waits for human input.

        Follows the same pattern as goal continuation and loop completion hooks:
        - Only runs when there's a completed turn result
        - Fail-open: script errors are logged but never break the CLI
        - Respects existing pending input (won't double-queue)
        """
        # Quick bail if not configured or no result
        try:
            from hermes_cli.input_hook import get_input_hook_script_path, run_input_hook
            script_path = get_input_hook_script_path()
            if not script_path:
                return  # Not configured
            
            # Skip if this was an interrupted turn
            if getattr(self, "_last_turn_interrupted", False):
                return

            # Get the last turn result
            result = getattr(self, "_last_turn_result", None)
            if not result or not result.get("final_response"):
                return  # No final response to base decision on

            # Extract content and reasoning from the last turn
            content = result.get("final_response", "")
            reasoning = result.get("last_reasoning") or ""

            # Get finish_reason from conversation history (last assistant msg)
            finish_reason = None
            try:
                history = self.conversation_history
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        finish_reason = msg.get("finish_reason")
                        break
            except Exception:
                pass

            # Get context metrics (optional, best-effort)
            context_used = None
            context_total = None
            try:
                agent = getattr(self, "agent", None)
                if agent and hasattr(agent, "context_compressor"):
                    compressor = agent.context_compressor
                    context_total = getattr(compressor, "context_length", None)
                    # Try to get current usage - different implementations vary
                    # Look at usage_anchor or similar state
                    from agent.usage_anchor import anchored_context_tokens
                    messages = self.conversation_history
                    anchor = getattr(agent, "_usage_anchor", None)
                    context_used = anchored_context_tokens(messages, anchor) if anchor else None
            except Exception as exc:
                logging.debug("input_hook context metrics unavailable: %s", exc)

            total_messages = len(self.conversation_history or [])

            # Build metadata
            from hermes_cli.input_hook import build_metadata, format_metadata_block
            metadata = build_metadata(
                context_used=context_used,
                context_total=context_total,
                total_messages=total_messages,
                finish_reason=finish_reason,
            )
            metadata_block = format_metadata_block(metadata)

            # Run the script
            auto_reply = run_input_hook(
                reasoning=reasoning,
                content=content,
                metadata_block=metadata_block,
            )

            # If we got an auto-reply, queue it as the next user message
            if auto_reply:
                from cli import _DIM, _RST, _cprint
                preview = auto_reply[:60] + ("..." if len(auto_reply) > 60 else "")
                _cprint(f"  {_DIM}⚡ Input hook auto-reply: {preview}{_RST}")
                self._pending_input.put(auto_reply)

        except Exception as exc:
            # Fail-open: never break the CLI on input_hook errors
            logging.warning("input_hook execution failed: %s", exc, exc_info=True)
            print(f"⚠️  input_hook error: {exc}", file=sys.stderr)
