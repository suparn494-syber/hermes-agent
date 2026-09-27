"""Tests that the background review agent restricts tools at runtime, not at schema time.

Regression coverage for issue #15204 (the background skill-review agent must
not perform non-skill side effects like terminal, send_message, delegate_task)
combined with issue #25322 / PR #17276 (the review fork must hit the parent's
Anthropic/OpenRouter prefix cache).

Reconciling the two: the fork now inherits the parent's full ``tools`` schema
so the cache-key matches, and enforces the memory+skills restriction at
runtime via a thread-local whitelist on the existing
``get_pre_tool_call_block_message`` gate. Safety is preserved mechanically
(any non-whitelisted dispatch is blocked) without the schema-level narrowing
that caused the prefix-cache miss.
"""

from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_matches_parent_toolset_config():
    """Fork must receive parent's toolset config so ``tools[]`` cache key matches."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {captured['enabled_toolsets']!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert captured["disabled_toolsets"] == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {captured['disabled_toolsets']!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def test_background_review_installs_thread_local_whitelist():
    """The review fork must install a memory/skills-only thread-local whitelist.

    The schema-level toolset narrowing was lifted (for prefix-cache parity),
    so #15204's safety contract now relies on the runtime whitelist gate to
    deny terminal/send_message/delegate_task at dispatch time. Verify the
    whitelist is set with exactly the memory+skills tool names.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        captured["deny_msg_fmt"] = deny_msg_fmt
        # Stop here — we just want to see what gets installed.
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        # Don't crash AIAgent.__init__; let execution flow reach
        # set_thread_tool_whitelist.
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "whitelist" in captured, "set_thread_tool_whitelist was not called"
    whitelist = captured["whitelist"]
    # Memory analysis remains allowed, but skill_manage is suppressed unless
    # the task explicitly authorizes skill maintenance.
    assert "memory" in whitelist
    assert "skill_manage" not in whitelist
    assert "skill_view" in whitelist
    assert "skills_list" in whitelist
    # dangerous tools must NOT be in the whitelist
    assert "terminal" not in whitelist
    assert "send_message" not in whitelist
    assert "delegate_task" not in whitelist
    assert "web_search" not in whitelist
    assert "execute_code" not in whitelist


def test_background_review_explicit_owner_scope_reaches_thread_whitelist():
    """Trusted explicit maintenance must survive the background thread boundary."""
    import run_agent
    from hermes_cli import plugins as _plugins
    from tools.skill_write_policy import SkillWritePolicy, skill_write_policy

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    with skill_write_policy(SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE), \
         patch.object(run_agent.AIAgent, "__init__", lambda self, *args, **kwargs: None), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    assert "skill_manage" in captured["whitelist"]
    assert "terminal" not in captured["whitelist"]


def test_background_review_explicit_owner_scope_reaches_real_thread_whitelist():
    """Explicit maintenance policy must propagate through an actual OS thread."""
    import threading
    import run_agent
    from hermes_cli import plugins as _plugins
    from tools.skill_write_policy import SkillWritePolicy, skill_write_policy

    captured = {}
    done = threading.Event()

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        done.set()
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    with skill_write_policy(SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE), \
         patch.object(run_agent.AIAgent, "__init__", lambda self, *args, **kwargs: None), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )
        assert done.wait(timeout=2), "background review thread did not reach whitelist setup"

    assert "skill_manage" in captured["whitelist"]
    assert "terminal" not in captured["whitelist"]


def test_background_review_read_only_prompt_does_not_request_skill_mutation():
    """Default-deny reviews should spend their turn analysing, not attempting writes."""
    import run_agent

    captured = {}

    def _capture_run(self, *args, **kwargs):
        captured["user_message"] = kwargs.get("user_message", "")
        return {"final_response": "ok"}

    agent = _make_agent_stub(run_agent.AIAgent)
    with patch.object(run_agent.AIAgent, "__init__", lambda self, *args, **kwargs: None), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    prompt = captured["user_message"].lower()
    assert "analysis-only" in prompt
    assert "do not create, update, patch, or delete skills" in prompt






