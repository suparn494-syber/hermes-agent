"""Focused tests for task-scoped skill-write policy."""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import skill_manage
from tools.skill_write_policy import (
    SkillWritePolicy,
    get_skill_write_policy,
    scoped_skill_write_policy,
    skill_write_policy,
)


CONTENT = """---
name: policy-skill
description: A policy test skill.
---

# Policy Skill

Do the thing.
"""


def _roots(tmp_path):
    return (
        patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]),
    )


def test_read_only_task_denies_all_skill_mutations(tmp_path):
    actions = [
        ("create", {"content": CONTENT}),
        ("patch", {"old_string": "Do the thing.", "new_string": "Changed."}),
        ("edit", {"content": CONTENT}),
        ("write_file", {"file_path": "references/x.md", "file_content": "x"}),
        ("remove_file", {"file_path": "references/x.md"}),
        ("delete", {}),
    ]
    p1, p2 = _roots(tmp_path)
    with p1, p2, skill_write_policy(SkillWritePolicy.READ_ONLY):
        for action, kwargs in actions:
            result = json.loads(skill_manage(action=action, name="policy-skill", **kwargs))
            assert result["success"] is False
            assert result.get("_read_only_policy") is True
    assert not (tmp_path / "policy-skill").exists()


def test_explicit_skill_maintenance_allows_foreground_write(tmp_path):
    p1, p2 = _roots(tmp_path)
    with p1, p2, skill_write_policy(SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE):
        result = json.loads(skill_manage(action="create", name="policy-skill", content=CONTENT))
    assert result["success"] is True
    assert (tmp_path / "policy-skill" / "SKILL.md").is_file()


def test_normal_policy_preserves_existing_foreground_behavior(tmp_path):
    p1, p2 = _roots(tmp_path)
    with p1, p2:
        result = json.loads(skill_manage(action="create", name="policy-skill", content=CONTENT))
    assert result["success"] is True


def test_background_review_without_owner_authorization_denies_skill_write(tmp_path):
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin
    p1, p2 = _roots(tmp_path)
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        with p1, p2:
            result = json.loads(skill_manage(action="create", name="policy-skill", content=CONTENT))
    finally:
        reset_current_write_origin(token)
    assert result["success"] is False
    assert result.get("_read_only_policy") is True
    assert not (tmp_path / "policy-skill").exists()


def test_background_review_explicit_skill_maintenance_allows_write(tmp_path):
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin
    p1, p2 = _roots(tmp_path)
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        with p1, p2, skill_write_policy(SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE):
            result = json.loads(skill_manage(action="create", name="policy-skill", content=CONTENT))
    finally:
        reset_current_write_origin(token)
    assert result["success"] is True
    assert (tmp_path / "policy-skill" / "SKILL.md").is_file()


def test_scoped_policy_decorator_applies_and_resets_at_task_boundary():
    observed = []

    @scoped_skill_write_policy
    def run_task(*, skill_write_policy=None):
        observed.append(get_skill_write_policy())
        return "ok"

    assert run_task(skill_write_policy=SkillWritePolicy.READ_ONLY) == "ok"
    assert observed == [SkillWritePolicy.READ_ONLY]
    assert get_skill_write_policy() == SkillWritePolicy.NORMAL


def test_conversation_entrypoint_exposes_trusted_skill_write_policy_scope():
    from agent.conversation_loop import run_conversation

    assert "skill_write_policy" in inspect.signature(run_conversation).parameters
    assert getattr(run_conversation, "_skill_write_policy_scoped", False) is True


def test_agent_forwarder_accepts_trusted_skill_write_policy_scope():
    from run_agent import AIAgent

    assert "skill_write_policy" in inspect.signature(AIAgent.run_conversation).parameters
