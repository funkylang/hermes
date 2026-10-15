"""Tests for steering-wait mechanism: user interruptions during reasoning."""
import pytest
import threading
from unittest.mock import Mock, patch


from agent.interrupt_control import InterruptControlMixin

# Import the phase constants for use in tests
STREAMING_PHASE_REASONING = InterruptControlMixin.STREAMING_PHASE_REASONING
STREAMING_PHASE_CONTENT = InterruptControlMixin.STREAMING_PHASE_CONTENT
STREAMING_PHASE_TOOL_ARGS = InterruptControlMixin.STREAMING_PHASE_TOOL_ARGS
STREAMING_PHASE_INACTIVE = InterruptControlMixin.STREAMING_PHASE_INACTIVE


def make_test_agent():
    """Create a minimal test agent with the necessary attributes for steering-wait."""
    agent = Mock()
    
    # Required by InterruptControlMixin
    agent._streaming_phase = STREAMING_PHASE_INACTIVE
    agent._pending_redirect_lock = threading.Lock()
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_steer = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    
    # Required by steering-wait mechanism
    agent._steer_completion_wait_active = False
    agent._steering_wait_completed = False
    agent._steering_wait_reasoning = ""
    agent._current_streaming_reasoning_text = ""
    agent._steer_completion_deadline = 0
    
    # Mock methods from InterruptControlMixin
    from agent.interrupt_control import InterruptControlMixin
    
    # Bind the mixin's methods to our mock
    for method_name in [
        '_set_streaming_phase',
        '_get_streaming_phase',
        '_check_steering_wait_completion',
        '_clear_steering_wait_markers',
        '_append_streaming_reasoning_text',
        '_reset_streaming_reasoning_text',
        '_get_accumulated_reasoning_text',
        'redirect',
    ]:
        setattr(agent, method_name, getattr(InterruptControlMixin, method_name).__get__(agent))
    
    return agent


class TestSteeringWaitMechanism:
    """Test the steering-wait completion detection logic."""
    
    def test_steering_wait_not_active_by_default(self):
        """Steering wait should not be active by default."""
        agent = make_test_agent()
        
        assert not getattr(agent, "_steer_completion_wait_active", False)
        assert not agent._check_steering_wait_completion("some reasoning text")
    
    def test_steering_wait_completes_on_newline(self):
        """Steering wait should complete when newline is detected in reasoning."""
        agent = make_test_agent()
        
        # Activate steering wait
        agent._steer_completion_wait_active = True
        agent._current_streaming_reasoning_text = "I'm thinking about this\n"
        
        # Should detect completion
        result = agent._check_steering_wait_completion("")
        
        assert result is True
        assert not agent._steer_completion_wait_active
        assert agent._steering_wait_completed
    
    def test_steering_wait_not_complete_without_newline(self):
        """Steering wait should not complete without a newline."""
        agent = make_test_agent()
        
        # Activate steering wait
        agent._steer_completion_wait_active = True
        agent._current_streaming_reasoning_text = "thinking without newline"
        
        # Should NOT detect completion
        result = agent._check_steering_wait_completion("")
        
        assert result is False
        assert agent._steer_completion_wait_active
        assert not agent._steering_wait_completed
    
    def test_steering_wait_timeout(self):
        """Steering wait should complete on timeout."""
        import time
        
        agent = make_test_agent()
        
        # Activate steering wait with past deadline
        agent._steer_completion_wait_active = True
        agent._current_streaming_reasoning_text = "thinking"
        agent._steer_completion_deadline = time.time() - 1  # Already expired
        
        # Should detect timeout completion
        result = agent._check_steering_wait_completion("")
        
        assert result is True
        assert not agent._steer_completion_wait_active
        assert agent._steering_wait_completed


class TestSteeringWaitPhaseDetection:
    """Test streaming phase tracking for steering behavior."""
    
    def test_reasoning_phase_detected(self):
        """Reasoning deltas should set the reasoning phase."""
        agent = make_test_agent()
        
        agent._set_streaming_phase(STREAMING_PHASE_REASONING)
        
        assert agent._get_streaming_phase() == STREAMING_PHASE_REASONING
    
    def test_content_phase_detected(self):
        """Content deltas should set the content phase."""
        agent = make_test_agent()
        
        agent._set_streaming_phase(STREAMING_PHASE_CONTENT)
        
        assert agent._get_streaming_phase() == STREAMING_PHASE_CONTENT
    
    def test_tool_args_phase_detected(self):
        """Tool args deltas should set the tool args phase."""
        agent = make_test_agent()
        
        agent._set_streaming_phase(STREAMING_PHASE_TOOL_ARGS)
        
        assert agent._get_streaming_phase() == STREAMING_PHASE_TOOL_ARGS


class TestReasoningTextAccumulation:
    """Test reasoning text tracking for steering completion."""
    
    def test_append_reasoning_text(self):
        """Reasoning text should be accumulated correctly."""
        agent = make_test_agent()
        
        agent._append_streaming_reasoning_text("First part")
        agent._append_streaming_reasoning_text(" second part\n")
        
        assert agent._get_accumulated_reasoning_text() == "First part second part\n"
    
    def test_reset_reasoning_text(self):
        """Reasoning text should be cleared on reset."""
        agent = make_test_agent()
        
        agent._append_streaming_reasoning_text("Some reasoning")
        agent._reset_streaming_reasoning_text()
        
        assert agent._get_accumulated_reasoning_text() == ""


class TestRedirectSteeringBehavior:
    """Test redirect behavior based on streaming phase."""
    
    def test_redirect_during_reasoning_waits(self):
        """Redirect in smart mode during reasoning should set steering wait, not abort."""
        agent = make_test_agent()
        
        # Set up as if we're during reasoning
        agent._set_streaming_phase(STREAMING_PHASE_REASONING)
        agent._executing_tools = False
        agent._execution_thread_id = None
        agent._model_request_active = Mock()
        agent._model_request_active.is_set.return_value = True
        
        # Redirect in smart mode should succeed and set steering wait
        result = agent.redirect("please change direction", busy_input_mode="smart")
        
        assert result is True
        assert agent._steer_completion_wait_active
        assert "please change direction" in (agent._pending_redirect or "")

    def test_redirect_outside_smart_reasoning_aborts(self):
        """Redirect in any other situation uses the upstream immediate-abort path."""
        agent = make_test_agent()
        
        # Set up as if we're during content (not smart+reasoning)
        agent._set_streaming_phase(STREAMING_PHASE_CONTENT)
        agent._executing_tools = False
        agent._execution_thread_id = None
        agent._model_request_active = Mock()
        agent._model_request_active.is_set.return_value = True
        
        result = agent.redirect("please change direction", busy_input_mode="smart")
        
        assert result is True
        # Upstream abort path: interrupt requested, steering wait NOT armed
        assert agent._interrupt_requested
        assert not getattr(agent, "_steer_completion_wait_active", False)
        assert "please change direction" in (agent._pending_redirect or "")


class TestSmartSteerToolPhase:
    """Smart mode deferral during tool execution / args streaming (no interrupt)."""

    def test_redirect_during_tool_execution_defers(self):
        """Smart + tools executing: defer via _pending_redirect, no abort, no steer()."""
        agent = make_test_agent()
        agent._executing_tools = True
        agent._set_streaming_phase(STREAMING_PHASE_INACTIVE)

        result = agent.redirect("steer one", busy_input_mode="smart")

        assert result is True
        assert agent._pending_redirect == "steer one"
        assert agent._smart_steer_pending is True
        assert not agent._interrupt_requested
        # steer() must not be touched (dead mechanism) — its Mock attr stays un-called text-wise
        assert agent._pending_steer is None

    def test_redirect_during_tool_args_defers(self):
        """Smart + args streaming: same deferral, stream left to finish."""
        agent = make_test_agent()
        agent._executing_tools = False
        agent._set_streaming_phase(STREAMING_PHASE_TOOL_ARGS)

        result = agent.redirect("steer one", busy_input_mode="smart")

        assert result is True
        assert agent._pending_redirect == "steer one"
        assert agent._smart_steer_pending is True
        assert not agent._interrupt_requested

    def test_smart_steers_append_plainly(self):
        """Multiple smart-mode steers concatenate with a plain newline (no decoration)."""
        agent = make_test_agent()
        agent._executing_tools = True

        agent.redirect("first", busy_input_mode="smart")
        agent.redirect("second", busy_input_mode="smart")

        assert agent._pending_redirect == "first\nsecond"
        assert agent._smart_steer_pending is True


class TestSmartSteerConsumption:
    """_apply_active_turn_redirect must emit a plain user row for smart tool-phase steers."""

    def _consume(self, agent, messages, text):
        from agent.conversation_loop import _apply_active_turn_redirect
        _apply_active_turn_redirect(agent, messages, text)

    def test_plain_user_row_after_tools(self):
        """No assistant placeholder, no api_content decoration; text stays verbatim."""
        agent = make_test_agent()
        agent._smart_steer_pending = True
        agent._current_streamed_assistant_text = ""
        messages = [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "result 1"},
            {"role": "tool", "content": "result 2"},
        ]

        self._consume(agent, messages, "steer text")

        # Flag cleared once consumed
        assert not agent._smart_steer_pending
        last = messages[-1]
        assert last["role"] == "user"
        assert last["content"] == "steer text"
        assert len(messages) == 5  # exactly one row added (the user steer)
        # No decoration sneaked in anywhere
        for m in messages:
            assert "[Context from the interrupted assistant response]" not in str(m.get("content", ""))
            assert "api_content" not in m

    def test_multiple_concatenated_steers_single_row(self):
        """Pre-concatenated steers land as ONE user row."""
        agent = make_test_agent()
        agent._smart_steer_pending = True
        messages = [{"role": "tool", "content": "r"}]

        self._consume(agent, messages, "first\nsecond")

        assert len(messages) == 2
        last = messages[-1]
        assert last["role"] == "user"
        assert last["content"] == "first\nsecond"


class TestClearSteeringWaitMarkers:
    """Test clearing steering wait state after consumption."""
    
    def test_clear_markers_resets_state(self):
        """All steering wait markers should be cleared."""
        agent = make_test_agent()
        
        # Set all the markers
        agent._steer_completion_wait_active = True
        agent._steering_wait_completed = True
        agent._steering_wait_reasoning = "preserved reasoning text"
        
        # Clear them
        agent._clear_steering_wait_markers()
        
        assert not agent._steer_completion_wait_active
        assert not agent._steering_wait_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
