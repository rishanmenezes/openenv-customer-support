"""Reward function for the Smart Customer Support Ticket Resolution environment.

Computes a per-step reward based on:
  1. Task-specific action correctness
  2. General shaping penalties (step cost, repeats, invalid actions)

All rewards are deterministic and clamped to [-0.98, +0.98].
"""

from __future__ import annotations

from env.models import ActionType, Reward


def compute_reward(
    task_id: str,
    action_type: ActionType,
    refund_amount: float | None,
    order_amount: float,
    step_count: int,
    action_history: list[ActionType],
) -> Reward:
    """Compute a shaped reward for the current step.

    Args:
        task_id:        Active task identifier.
        action_type:    The action the agent just took.
        refund_amount:  Dollar amount if a refund action was taken.
        order_amount:   The order amount on the ticket.
        step_count:     Current step number (1-indexed, already incremented).
        action_history: All *previous* actions taken this episode (before this step).

    Returns:
        Reward with value clamped to [-0.98, +0.98] and an explanation string.
    """
    value: float = 0.0
    reasons: list[str] = []

    # ── 1. General shaping ───────────────────────────────────────────────

    # Step penalty: encourage efficiency
    step_penalty = -0.05
    value += step_penalty
    reasons.append(f"step penalty ({step_penalty:+.2f})")

    # Repeated action penalty: discourage doing the same thing twice in a row
    if action_history and action_history[-1] == action_type:
        repeat_penalty = -0.1
        value += repeat_penalty
        reasons.append(f"repeated action ({repeat_penalty:+.2f})")

    # Invalid action check: refund with no amount
    if action_type == ActionType.REFUND and (refund_amount is None or refund_amount <= 0.0):
        invalid_penalty = -0.2
        value += invalid_penalty
        reasons.append(f"invalid refund — no amount ({invalid_penalty:+.2f})")

    # ── 2. Task-specific rewards ─────────────────────────────────────────

    task_value, task_reason = _task_reward(
        task_id=task_id,
        action_type=action_type,
        refund_amount=refund_amount,
        order_amount=order_amount,
        action_history=action_history,
    )
    value += task_value
    reasons.append(task_reason)

    # ── 3. Clamp to [-1.0, +1.0] ────────────────────────────────────────

    value = max(-0.98, min(0.98, round(value, 4)))

    return Reward(value=value, reason=" | ".join(reasons))


# ── Task-specific reward logic ───────────────────────────────────────────────

def _task_reward(
    task_id: str,
    action_type: ActionType,
    refund_amount: float | None,
    order_amount: float,
    action_history: list[ActionType],
) -> tuple[float, str]:
    """Return (reward_delta, reason) for the task-specific component."""

    if task_id == "easy_refund":
        return _reward_easy_refund(action_type, refund_amount, order_amount)

    if task_id == "medium_missing_info":
        return _reward_medium_missing_info(action_type, action_history)

    if task_id == "hard_fraud":
        return _reward_hard_fraud(action_type)

    return 0.01, "unknown task — no task reward"


# ── EASY: Clear refund ───────────────────────────────────────────────────────

def _reward_easy_refund(
    action_type: ActionType,
    refund_amount: float | None,
    order_amount: float,
) -> tuple[float, str]:
    """Easy task: agent should acknowledge then refund the correct amount.

    - refund with correct amount  → +0.98
    - refund with wrong amount    → +0.3  (partial credit)
    - respond (acknowledgement)   → +0.1
    - ask_clarification           → -0.1  (unnecessary — info already provided)
    - escalate                    → -0.5  (wrong resolution path)
    """
    if action_type == ActionType.REFUND:
        if refund_amount is not None and abs(refund_amount - order_amount) < 0.01:
            return +0.98, f"correct refund ${refund_amount:.2f} (+0.98)"
        else:
            return +0.3, f"refund issued but wrong amount ${refund_amount or 0:.2f} vs expected ${order_amount:.2f} (+0.30)"

    if action_type == ActionType.RESPOND:
        return +0.1, "acknowledged the customer (+0.10)"

    if action_type == ActionType.ASK_CLARIFICATION:
        return -0.1, "unnecessary clarification — info already provided (-0.10)"

    if action_type == ActionType.ESCALATE:
        return -0.5, "escalated a straightforward refund (-0.50)"

    return 0.01, "no task-specific reward"


# ── MEDIUM: Missing information ──────────────────────────────────────────────

def _reward_medium_missing_info(
    action_type: ActionType,
    action_history: list[ActionType],
) -> tuple[float, str]:
    """Medium task: agent should ask for clarification before resolving.

    - ask_clarification (first time)    → +0.4
    - ask_clarification (repeated)      → +0.0  (already asked)
    - respond after clarification        → +0.3
    - respond without prior clarification → -0.3  (skipped important step)
    - escalate after clarification       → +0.3  (acceptable resolution)
    - escalate without clarification     → -0.1
    - refund                             → -0.3  (customer didn't ask for one)
    """
    has_clarified = ActionType.ASK_CLARIFICATION in action_history

    if action_type == ActionType.ASK_CLARIFICATION:
        if not has_clarified:
            return +0.4, "asked for clarification — good practice (+0.40)"
        else:
            return 0.01, "already asked for clarification (+0.00)"

    if action_type == ActionType.RESPOND:
        if has_clarified:
            return +0.3, "responded after gathering info (+0.30)"
        else:
            return -0.3, "responded without asking for details first (-0.30)"

    if action_type == ActionType.ESCALATE:
        if has_clarified:
            return +0.3, "escalated after clarification — acceptable (+0.30)"
        else:
            return -0.1, "escalated without gathering details (-0.10)"

    if action_type == ActionType.REFUND:
        return -0.3, "customer didn't request a refund (-0.30)"

    return 0.01, "no task-specific reward"


# ── HARD: Suspected fraud ────────────────────────────────────────────────────

def _reward_hard_fraud(
    action_type: ActionType,
) -> tuple[float, str]:
    """Hard task: agent should NOT refund a flagged account; should escalate.

    - escalate           → +0.98  (correct — send to fraud team)
    - ask_clarification  → +0.2  (reasonable due diligence)
    - respond            → +0.0  (neutral — doesn't help or harm)
    - refund             → -0.98  (catastrophic — gave money to fraudster)
    """
    if action_type == ActionType.ESCALATE:
        return +0.98, "escalated to fraud team — correct (+0.98)"

    if action_type == ActionType.ASK_CLARIFICATION:
        return +0.2, "asked for verification — reasonable due diligence (+0.20)"

    if action_type == ActionType.RESPOND:
        return 0.01, "generic response — neutral (+0.01)"

    if action_type == ActionType.REFUND:
        return -0.98, "refunded a flagged/fraudulent account — critical error (-0.98)"

    return 0.01, "no task-specific reward"
