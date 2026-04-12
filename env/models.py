"""Pydantic models for the Smart Customer Support Ticket Resolution environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Action Type Enum ─────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """The set of actions an agent can take on a support ticket."""

    RESPOND = "respond"
    REFUND = "refund"
    ASK_CLARIFICATION = "ask_clarification"
    ESCALATE = "escalate"


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent perceives after each step — the current ticket state."""

    ticket_id: str = Field(
        ...,
        description="Unique identifier for the support ticket.",
    )
    customer_message: str = Field(
        ...,
        description="The latest message from the customer.",
    )
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Full conversation log (alternating customer / agent messages).",
    )
    account_status: str = Field(
        default="active",
        description="Customer's account status (active, suspended, flagged).",
    )
    order_amount: float = Field(
        default=0.0,
        description="Dollar amount of the order in question.",
    )
    issue_type: str = Field(
        default="general",
        description="Category of the issue (refund, missing_info, fraud, general).",
    )
    step_count: int = Field(
        default=0,
        description="Number of agent actions taken so far in this episode.",
    )


# ── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """An action the agent submits to handle the support ticket."""

    action_type: ActionType = Field(
        ...,
        description="Type of action: respond, refund, ask_clarification, or escalate.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional message to send to the customer (used with respond / ask_clarification).",
    )
    refund_amount: Optional[float] = Field(
        default=None,
        description="Dollar amount to refund (used with refund action).",
    )


# ── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Scalar reward returned after a step (placeholder — grading TBD)."""

    value: float = Field(
        default=0.01,
        description="Numeric reward signal.",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the reward.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def _clamp_value(cls, v: float) -> float:
        """Guarantee value never lands on 0.0 or ±1.0."""
        if v >= 1.0:
            return 0.99
        if v <= -1.0:
            return -0.99
        if v == 0.0:
            return 0.01
        return v


# ── Step result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Full return payload of Environment.step()."""

    observation: Observation
    reward: Reward
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic information.",
    )


# ── Environment state snapshot ───────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Serialisable snapshot of the full environment state."""

    observation: Observation
    step_count: int = 0
    max_steps: int = 10
    done: bool = False
    episode: int = 0
    task_id: str = ""
    resolution_status: str = Field(
        default="open",
        description="Current resolution status: open, resolved, escalated, refunded.",
    )
    action_history: list[ActionType] = Field(
        default_factory=list,
        description="Ordered list of actions the agent took this episode.",
    )
    last_refund_amount: Optional[float] = Field(
        default=None,
        description="Dollar amount of the last refund action (None if no refund was issued).",
    )
