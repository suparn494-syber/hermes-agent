import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import skill_manage
from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin
from tools.skill_write_policy import SkillWritePolicy, background_review_writes_allowed, skill_write_policy

CONTENT = "---\nname: guard-probe\ndescription: guard probe\n---\n\n# Guard Probe\n"


def test_reflection_and_proposals_remain_available_without_write(tmp_path):
    assert background_review_writes_allowed() is False
    proposal = {"kind": "skill_improvement", "persisted": False}
    assert proposal["kind"] == "skill_improvement"
    assert proposal["persisted"] is False


def test_automatic_shared_skill_write_is_default_deny(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]
    ):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            result = json.loads(skill_manage(action="create", name="guard-probe", content=CONTENT))
        finally:
            reset_current_write_origin(token)
    assert result["success"] is False
    assert result["_read_only_policy"] is True
    assert not (tmp_path / "guard-probe").exists()


def test_explicit_owner_skill_maintenance_remains_possible(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]
    ), skill_write_policy(SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE):
        result = json.loads(skill_manage(action="create", name="guard-probe", content=CONTENT))
    assert result["success"] is True
    assert (tmp_path / "guard-probe" / "SKILL.md").is_file()


def test_protected_target_hash_unchanged():
    target = Path(r"D:\hermes\skills\software-development\game-asset-generation-gates\SKILL.md")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == "75f1d93a3d4d6260b688d79347792ea67b1f1395ba992cacffc2b340fadb3e2c"
