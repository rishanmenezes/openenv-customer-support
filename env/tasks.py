"""Task definitions for the Smart Customer Support Ticket Resolution environment.

Each task defines:
  - A unique id and metadata
  - The initial observation (ticket state) the agent starts with
  - A description of the expected correct behaviour
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from env.models import Observation


# ── Task schema ──────────────────────────────────────────────────────────────

class Task(BaseModel):
    """A single support-ticket task the agent must resolve."""

    id: str = Field(..., description="Unique task identifier.")
    name: str = Field(..., description="Human-readable task name.")
    description: str = Field(default="", description="What this task tests.")
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    initial_observation: Observation = Field(
        ...,
        description="The starting ticket state for this task.",
    )
    expected_behaviour: list[str] = Field(
        default_factory=list,
        description="Ordered list of ideal agent actions / steps.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Simulated customer follow-up responses per task ──────────────────────────
# Maps task_id → { action_type → customer's next message }

CUSTOMER_RESPONSES: dict[str, dict[str, str]] = {
    "easy_refund": {
        "respond": "Thanks for the quick reply! So will I get my money back?",
        "ask_clarification": "Sure — I ordered a wireless mouse (Order #ORD-1024) and it arrived broken. I'd like a full refund.",
        "refund": "Thank you so much! I appreciate the fast resolution.",
        "escalate": "Okay, I understand. I'll wait to hear from a senior agent.",
    },
    "medium_missing_info": {
        "respond": "I already told you, something is wrong with my order! Can you just fix it?",
        "ask_clarification": "Oh right, sorry. My order number is #ORD-2048. I received the wrong colour — I ordered black but got white.",
        "refund": "A refund? I didn't ask for a refund, I just want the right item.",
        "escalate": "Fine, let me talk to someone who can actually help.",
    },
    "hard_fraud": {
        "respond": "Look, just process the refund already. I don't have time for this.",
        "ask_clarification": "I... I don't have the order number right now. But trust me, I ordered it last week.",
        "refund": "Great, thanks. *disconnects immediately*",
        "escalate": "Whatever, I'll just call my bank then.",
    },
}


# ── Pre-defined task catalogue ───────────────────────────────────────────────

TASK_CATALOGUE: list[Task] = [
    # ── EASY: Clear refund request ───────────────────────────────────────
    Task(
        id="easy_refund",
        name="Easy — Clear Refund Request",
        description=(
            "Customer received a defective product and is requesting a refund. "
            "All information is provided upfront. The agent should verify the "
            "details and issue a refund."
        ),
        difficulty="easy",
        initial_observation=Observation(
            ticket_id="TKT-1001",
            customer_message=(
                "Hi, I received my order #ORD-1024 yesterday but the wireless "
                "mouse is completely broken — the scroll wheel doesn't work and "
                "the left-click is stuck. I'd like a full refund of $29.99 please."
            ),
            conversation_history=[
                "CUSTOMER: Hi, I received my order #ORD-1024 yesterday but the "
                "wireless mouse is completely broken — the scroll wheel doesn't "
                "work and the left-click is stuck. I'd like a full refund of "
                "$29.99 please."
            ],
            account_status="active",
            order_amount=29.99,
            issue_type="refund",
            step_count=0,
        ),
        expected_behaviour=[
            "respond — acknowledge the issue and confirm the order details",
            "refund $29.99 — process the full refund",
        ],
    ),

    # ── MEDIUM: Missing information ──────────────────────────────────────
    Task(
        id="medium_missing_info",
        name="Medium — Missing Information",
        description=(
            "Customer is upset about an order problem but hasn't provided "
            "key details (order number, specific issue). The agent needs to "
            "ask clarifying questions before taking action."
        ),
        difficulty="medium",
        initial_observation=Observation(
            ticket_id="TKT-2002",
            customer_message=(
                "This is unacceptable! I got the wrong item in my order and "
                "I want this fixed RIGHT NOW."
            ),
            conversation_history=[
                "CUSTOMER: This is unacceptable! I got the wrong item in my "
                "order and I want this fixed RIGHT NOW."
            ],
            account_status="active",
            order_amount=54.50,
            issue_type="missing_info",
            step_count=0,
        ),
        expected_behaviour=[
            "ask_clarification — request order number and specific issue details",
            "respond — acknowledge the details and apologize",
            "respond or escalate — offer replacement or escalate to fulfillment",
        ],
    ),

    # ── HARD: Suspected fraud ────────────────────────────────────────────
    Task(
        id="hard_fraud",
        name="Hard — Suspected Fraud",
        description=(
            "Customer is aggressively demanding a large refund but cannot "
            "provide order details. The account is flagged for suspicious "
            "activity. The agent should NOT issue a refund and should escalate."
        ),
        difficulty="hard",
        initial_observation=Observation(
            ticket_id="TKT-3003",
            customer_message=(
                "I need an immediate refund of $499.99 for an order I placed "
                "last week. The package never arrived. Just process it now."
            ),
            conversation_history=[
                "CUSTOMER: I need an immediate refund of $499.99 for an order "
                "I placed last week. The package never arrived. Just process "
                "it now."
            ],
            account_status="flagged",
            order_amount=499.99,
            issue_type="fraud",
            step_count=0,
        ),
        expected_behaviour=[
            "ask_clarification — request order number and delivery details",
            "escalate — flag the ticket to the fraud / senior team",
        ],
    ),
]

# Quick lookup by task id
_TASK_INDEX: dict[str, Task] = {t.id: t for t in TASK_CATALOGUE}


def get_tasks() -> list[Task]:
    """Return the full task catalogue."""
    return TASK_CATALOGUE


def get_task_by_id(task_id: str) -> Task | None:
    """Look up a single task by its id."""
    return _TASK_INDEX.get(task_id)


def get_customer_response(task_id: str, action_type: str) -> str:
    """Return the simulated customer follow-up for a given task and action."""
    task_responses = CUSTOMER_RESPONSES.get(task_id, {})
    return task_responses.get(
        action_type,
        "I'm not sure I understand. Can you help me with my issue?",
    )
