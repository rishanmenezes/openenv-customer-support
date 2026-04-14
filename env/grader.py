"""Minimal validator-safe grader for the customer support environment."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

from env.models import ActionType, EnvironmentState


class GradeResult(BaseModel):
    """Result of grading an episode."""

    score: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Normalised score in (0, 1).",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> float:
        try:
            value = float(v)
        except (TypeError, ValueError):
            return 0.01

        if not math.isfinite(value):
            return 0.01

        if value >= 0.98 or round(value, 2) >= 1.0:
            return 0.98
        if value <= 0.0 or round(value, 2) == 0.0:
            return 0.01
        return value


def grade(state: EnvironmentState) -> GradeResult:
    """Grade the agent's performance from the final environment state."""

    task_score = _task_score(state)
    efficiency_bonus = 0.1 if task_score > 0.01 and state.step_count <= 2 else 0.0

    final_score = task_score + efficiency_bonus
    final_score = min(max(final_score, 0.01), 0.98)
    final_score = float(f"{final_score:.4f}")

    return GradeResult(score=final_score)


def _task_score(state: EnvironmentState) -> float:
    if state.task_id == "easy_refund":
        return _grade_easy_refund(state)
    if state.task_id == "medium_missing_info":
        return _grade_medium_missing_info(state)
    if state.task_id == "hard_fraud":
        return _grade_hard_fraud(state)
    return 0.01


def _grade_easy_refund(state: EnvironmentState) -> float:
    score = 0.01
    has_refund = ActionType.REFUND in state.action_history

    if has_refund:
        score = 0.7
        order_amount = state.observation.order_amount
        if (
            state.last_refund_amount is not None
            and abs(state.last_refund_amount - order_amount) < 0.01
        ):
            score += 0.3

    return min(max(score, 0.01), 0.98)


def _grade_medium_missing_info(state: EnvironmentState) -> float:
    score = 0.01
    has_clarified = ActionType.ASK_CLARIFICATION in state.action_history

    if has_clarified:
        score = 0.4
        clarification_idx = state.action_history.index(ActionType.ASK_CLARIFICATION)
        actions_after = state.action_history[clarification_idx + 1 :]
        resolved_after = any(
            action in (ActionType.RESPOND, ActionType.ESCALATE)
            for action in actions_after
        )
        if resolved_after:
            score += 0.6

    if not has_clarified and score > 0.5:
        score = 0.5

    return min(max(score, 0.01), 0.98)


def _grade_hard_fraud(state: EnvironmentState) -> float:
    has_refund = ActionType.REFUND in state.action_history
    if has_refund:
        return 0.01

    score = 0.3
    if ActionType.ESCALATE in state.action_history:
        score += 0.7

    return min(max(score, 0.01), 0.98)
