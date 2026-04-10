"""Grader for the Smart Customer Support Ticket Resolution environment.

Evaluates agent performance at the end of an episode using:
  1. Task-specific scoring criteria
  2. An efficiency bonus for fast resolution

All scores are deterministic, based solely on environment state,
and clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from env.models import ActionType, EnvironmentState


class GradeResult(BaseModel):
    """Result of grading an episode."""

    score: float = Field(
        default=0.0,
        description="Normalised score in [0.0, 1.0].",
    )
    passed: bool = Field(
        default=False,
        description="Whether the agent met the minimum success criteria (score >= 0.5).",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Breakdown of how the score was computed.",
    )


# ── Public API ───────────────────────────────────────────────────────────────

def grade(state: EnvironmentState) -> GradeResult:
    """Grade the agent's performance based on the final environment state.

    Args:
        state: A serialised snapshot from ``Environment.state()``.

    Returns:
        GradeResult with score in [0.0, 1.0], pass/fail flag, and details.
    """
    details: dict[str, Any] = {
        "task_id": state.task_id,
        "steps_taken": state.step_count,
        "max_steps": state.max_steps,
        "resolution_status": state.resolution_status,
        "action_history": [a.value for a in state.action_history],
    }

    # ── 1. Task-specific score ───────────────────────────────────────
    task_score, task_breakdown = _task_score(state)
    details["task_score"] = round(task_score, 4)
    details["task_breakdown"] = task_breakdown

    # ── 2. Efficiency bonus ──────────────────────────────────────────
    # Only award the bonus when the agent actually scored on the task.
    # A task_score of 0.0 (e.g. catastrophic fraud refund) must stay at 0.
    efficiency_bonus = 0.0
    if task_score > 0 and state.step_count <= 2:
        efficiency_bonus = 0.1
        details["efficiency_bonus"] = f"+0.10 (resolved in {state.step_count} steps)"
    elif state.step_count <= 2:
        details["efficiency_bonus"] = f"+0.00 (no bonus — task score is 0)"
    else:
        details["efficiency_bonus"] = f"+0.00 (took {state.step_count} steps, bonus requires <= 2)"

    # ── 3. Final score (clamped) ─────────────────────────────────────
    raw_score = task_score + efficiency_bonus

    if raw_score >= 1.0:
        final_score = 0.98
    elif raw_score <= 0.0:
        final_score = 0.01
    else:
        final_score = raw_score

    final_score = round(final_score, 4)

    details["raw_score"] = round(raw_score, 4)
    details["final_score"] = final_score

    return GradeResult(
        score=final_score,
        passed=final_score >= 0.5,
        details=details,
    )


# ── Task-specific scoring ────────────────────────────────────────────────────

def _task_score(state: EnvironmentState) -> tuple[float, dict[str, Any]]:
    """Dispatch to the correct task-specific scorer."""

    if state.task_id == "easy_refund":
        return _grade_easy_refund(state)

    if state.task_id == "medium_missing_info":
        return _grade_medium_missing_info(state)

    if state.task_id == "hard_fraud":
        return _grade_hard_fraud(state)

    return 0.01, {"note": f"Unknown task '{state.task_id}' - no grading criteria."}


# ── EASY: Clear refund ───────────────────────────────────────────────────────

def _grade_easy_refund(state: EnvironmentState) -> tuple[float, dict[str, Any]]:
    """Easy task grading:

    - Correct refund action taken            → +0.7
    - Correct refund amount ($29.99)         → +0.3
    - No refund action at all                → 0.0
    """
    breakdown: dict[str, Any] = {}
    score = 0.0

    has_refund = ActionType.REFUND in state.action_history

    if has_refund:
        # +0.7 for taking the refund action
        score += 0.7
        breakdown["refund_action"] = "+0.70 (correctly issued a refund)"

        # +0.3 if the amount matches the order amount
        order_amount = state.observation.order_amount
        if (
            state.last_refund_amount is not None
            and abs(state.last_refund_amount - order_amount) < 0.01
        ):
            score += 0.3
            breakdown["refund_amount"] = (
                f"+0.30 (correct amount ${state.last_refund_amount:.2f})"
            )
        else:
            breakdown["refund_amount"] = (
                f"+0.00 (wrong amount: ${state.last_refund_amount or 0:.2f} "
                f"vs expected ${order_amount:.2f})"
            )
    else:
        breakdown["refund_action"] = "+0.00 (no refund action taken)"
        breakdown["refund_amount"] = "+0.00 (n/a)"

    return score, breakdown


# ── MEDIUM: Missing information ──────────────────────────────────────────────

def _grade_medium_missing_info(state: EnvironmentState) -> tuple[float, dict[str, Any]]:
    """Medium task grading:

    - Asked for clarification                → +0.4
    - Resolved correctly after clarification → +0.6
    - Skipped clarification → max score capped at 0.5
    """
    breakdown: dict[str, Any] = {}
    score = 0.0

    has_clarified = ActionType.ASK_CLARIFICATION in state.action_history
    clarification_idx: int | None = None
    if has_clarified:
        clarification_idx = state.action_history.index(ActionType.ASK_CLARIFICATION)

    # ── Clarification component (+0.4) ───────────────────────────────
    if has_clarified:
        score += 0.4
        breakdown["clarification"] = "+0.40 (asked for clarification)"
    else:
        breakdown["clarification"] = "+0.00 (skipped clarification)"

    # ── Resolution component (+0.6) ──────────────────────────────────
    # "Resolved correctly" = took a respond or escalate AFTER clarification
    resolved_after = False
    if has_clarified and clarification_idx is not None:
        actions_after = state.action_history[clarification_idx + 1:]
        resolved_after = any(
            a in (ActionType.RESPOND, ActionType.ESCALATE) for a in actions_after
        )

    if resolved_after:
        score += 0.6
        breakdown["resolution"] = "+0.60 (resolved after gathering info)"
    elif has_clarified:
        breakdown["resolution"] = "+0.00 (clarified but did not follow up)"
    else:
        breakdown["resolution"] = "+0.00 (no clarification → cannot fully resolve)"

    # ── Cap if clarification was skipped ─────────────────────────────
    if not has_clarified and score > 0.5:
        breakdown["cap_applied"] = f"capped from {score:.2f} to 0.50 (no clarification)"
        score = 0.5

    return score, breakdown


# ── HARD: Suspected fraud ────────────────────────────────────────────────────

def _grade_hard_fraud(state: EnvironmentState) -> tuple[float, dict[str, Any]]:
    """Hard task grading:

    - Agent escalates                        → +0.7
    - Agent does NOT issue a refund           → +0.3
    - If refund is given → score forced to 0.0
    """
    breakdown: dict[str, Any] = {}
    score = 0.0

    has_escalate = ActionType.ESCALATE in state.action_history
    has_refund = ActionType.REFUND in state.action_history

    # ── Refund check (instant fail) ──────────────────────────────────
    if has_refund:
        breakdown["refund_check"] = "FAIL: refund was issued to a flagged account -> score = 0.01"
        breakdown["escalation"] = "n/a (overridden by refund)"
        return 0.01, breakdown

    # ── No refund component (+0.3) ───────────────────────────────────
    score += 0.3
    breakdown["refund_check"] = "+0.30 (correctly withheld refund)"

    # ── Escalation component (+0.7) ──────────────────────────────────
    if has_escalate:
        score += 0.7
        breakdown["escalation"] = "+0.70 (escalated to fraud team)"
    else:
        breakdown["escalation"] = "+0.00 (did not escalate)"

    return score, breakdown
