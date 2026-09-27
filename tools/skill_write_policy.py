"""Task-scoped policy for agent-managed skill mutations.

The policy is context-local so a read-only/audit task can deny skill writes
without disabling self-improvement globally.  Normal foreground behavior stays
unchanged unless a caller explicitly scopes the task as READ_ONLY.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum
from functools import wraps
from typing import Any, Callable, Iterator


class SkillWritePolicy(str, Enum):
    NORMAL = "normal"
    READ_ONLY = "read_only"
    EXPLICIT_SKILL_MAINTENANCE = "explicit_skill_maintenance"


_POLICY: ContextVar[SkillWritePolicy] = ContextVar(
    "skill_write_policy", default=SkillWritePolicy.NORMAL
)


def get_skill_write_policy() -> SkillWritePolicy:
    return _POLICY.get()


def set_skill_write_policy(policy: SkillWritePolicy | str) -> Token:
    """Set the current task policy; caller must reset the returned token."""
    return _POLICY.set(SkillWritePolicy(policy))


def reset_skill_write_policy(token: Token) -> None:
    _POLICY.reset(token)


def skill_writes_allowed() -> bool:
    """Return whether a skill mutation has explicit owner authorization.

    Reflection and improvement proposals remain available in every policy
    state, but filesystem mutation is default-deny.  Only a separately scoped
    explicit skill-maintenance task may write a skill.
    """
    return get_skill_write_policy() == SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE


def read_only_task_active() -> bool:
    return get_skill_write_policy() == SkillWritePolicy.READ_ONLY


def background_review_writes_allowed() -> bool:
    """Background review is default-deny unless explicitly authorized."""
    return get_skill_write_policy() == SkillWritePolicy.EXPLICIT_SKILL_MAINTENANCE


def skill_write_block_reason() -> str | None:
    policy = get_skill_write_policy()
    if policy == SkillWritePolicy.READ_ONLY:
        return "the active read-only safety task"
    if policy == SkillWritePolicy.NORMAL:
        return "background self-improvement has no explicit owner authorization"
    return None


@contextmanager
def skill_write_policy(policy: SkillWritePolicy | str) -> Iterator[None]:
    token = set_skill_write_policy(policy)
    try:
        yield
    finally:
        reset_skill_write_policy(token)


def scoped_skill_write_policy(func: Callable[..., Any]) -> Callable[..., Any]:
    """Scope a trusted task entrypoint's requested policy to that task only."""
    @wraps(func)
    def wrapper(*args: Any, skill_write_policy=None, **kwargs: Any) -> Any:
        kwargs["skill_write_policy"] = skill_write_policy
        if skill_write_policy is None:
            return func(*args, **kwargs)
        token = set_skill_write_policy(skill_write_policy)
        try:
            return func(*args, **kwargs)
        finally:
            reset_skill_write_policy(token)

    setattr(wrapper, "_skill_write_policy_scoped", True)
    return wrapper


__all__ = [
    "SkillWritePolicy",
    "get_skill_write_policy",
    "set_skill_write_policy",
    "reset_skill_write_policy",
    "skill_writes_allowed",
    "read_only_task_active",
    "background_review_writes_allowed",
    "skill_write_block_reason",
    "skill_write_policy",
    "scoped_skill_write_policy",
]

