"""Focused guards for unbounded tool loops and context growth."""

from agent.chat_completion_helpers import handle_max_iterations
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


class _FakeAgent:
    def __init__(self):
        self.max_iterations = 40
        self.session_id = "test-session"
        self._circuit_broken = False
        self.prints = []

    def _safe_print(self, msg):
        self.prints.append(msg)


def test_max_turn_guard_stops_without_summary_api_call():
    agent = _FakeAgent()
    out = handle_max_iterations(agent, messages=[{"role": "user", "content": "hi"}], api_call_count=40)
    assert "LOOP_GUARD_TRIGGERED" in out
    assert agent._circuit_broken is True
    assert "40" in out
    assert "summary" not in out.lower() or "instead of requesting a long summary" in out


def test_circuit_breaker_resets_for_next_request():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, identical_success_block_after=2)
    )
    args = {"path": "x"}
    controller.after_call("read_file", args, "ok", failed=False)
    halt = controller.after_call("read_file", args, "ok", failed=False)
    assert halt.should_halt is True
    controller.reset_for_turn()
    assert controller.halt_decision is None
    assert controller.before_call("read_file", args).action == "allow"


def test_config_parses_new_loop_and_result_caps():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "hard_stop_enabled": True,
            "max_tool_result_chars": 8000,
            "warn_after": {"identical_success": 2},
            "hard_stop_after": {"identical_success": 3, "exact_failure": 3},
        }
    )
    assert cfg.hard_stop_enabled is True
    assert cfg.max_tool_result_chars == 8000
    assert cfg.identical_success_block_after == 3
    assert cfg.exact_failure_block_after == 3


def test_different_arguments_are_not_a_loop():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, identical_success_block_after=2)
    )
    for i in range(5):
        args = {"path": f"file-{i}.md"}
        assert controller.before_call("read_file", args).action == "allow"
        assert controller.after_call("read_file", args, "ok", failed=False).action == "allow"
