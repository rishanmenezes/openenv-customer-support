"""Core environment class for the Smart Customer Support Ticket Resolution system.

Implements reset / step / state with deterministic task-based scenarios.
"""

from __future__ import annotations

from typing import Any

from env.models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
)
from env.reward import compute_reward
from env.tasks import get_customer_response, get_task_by_id, Task

# Cycle order used when no explicit task_id is given
_TASK_CYCLE: list[str] = ["easy_refund", "medium_missing_info", "hard_fraud"]


class Environment:
    """Smart Customer Support Ticket Resolution environment.

    Each episode presents a support ticket. The agent takes actions
    (respond, refund, ask_clarification, escalate) and the environment
    simulates deterministic customer follow-ups and state transitions.
    """

    def __init__(self, max_steps: int = 10) -> None:
        self._max_steps: int = max_steps
        self._step_count: int = 0
        self._done: bool = False
        self._episode: int = 0
        self._task_id: str = ""
        self._task: Task | None = None
        self._resolution_status: str = "open"
        self._action_history: list[ActionType] = []
        self._last_refund_amount: float | None = None
        self._observation: Observation = Observation(
            ticket_id="",
            customer_message="",
        )

    # ── public interface ─────────────────────────────────────────────────

    def reset(self, task_id: str | None = None) -> Observation:
        """Reset the environment and load a task.

        Args:
            task_id: Explicit task to load.  If *None*, cycles through tasks
                     in a deterministic round-robin order.
        """
        # Pick task
        if task_id is None:
            task_id = _TASK_CYCLE[self._episode % len(_TASK_CYCLE)]

        task = get_task_by_id(task_id)
        if task is None:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {[t for t in _TASK_CYCLE]}"
            )

        self._task = task
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._episode += 1
        self._resolution_status = "open"
        self._action_history = []
        self._last_refund_amount = None

        # Deep-copy the initial observation so the task template stays clean
        self._observation = task.initial_observation.model_copy(deep=True)

        return self._observation

    def step(self, action: Action) -> StepResult:
        """Apply an agent action and return the updated state.

        State transitions:
        - **respond**: Agent message is logged; simulated customer reply added.
        - **ask_clarification**: Agent question logged; customer provides info.
        - **refund**: Refund is processed; episode ends.
        - **escalate**: Ticket is escalated; episode ends.
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )

        self._step_count += 1
        info: dict[str, Any] = {"task_id": self._task_id, "episode": self._episode}

        # ── Compute reward BEFORE state mutation ────────────────────
        # (action_history contains only *previous* actions at this point)
        reward = compute_reward(
            task_id=self._task_id,
            action_type=action.action_type,
            refund_amount=action.refund_amount,
            order_amount=self._observation.order_amount,
            step_count=self._step_count,
            action_history=list(self._action_history),
        )

        # Record action in history
        self._action_history.append(action.action_type)

        # ── Handle each action type ─────────────────────────────────
        if action.action_type == ActionType.RESPOND:
            self._handle_respond(action, info)

        elif action.action_type == ActionType.ASK_CLARIFICATION:
            self._handle_ask_clarification(action, info)

        elif action.action_type == ActionType.REFUND:
            self._handle_refund(action, info)

        elif action.action_type == ActionType.ESCALATE:
            self._handle_escalate(action, info)

        # ── Check max-step termination ──────────────────────────────
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            self._resolution_status = "timeout"
            info["termination_reason"] = "max_steps_reached"

        # ── Build observation snapshot ──────────────────────────────
        self._observation.step_count = self._step_count
        info["resolution_status"] = self._resolution_status
        info["reward_breakdown"] = reward.reason

        return StepResult(
            observation=self._observation.model_copy(deep=True),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return a serialisable snapshot of the environment's current state."""
        return EnvironmentState(
            observation=self._observation.model_copy(deep=True),
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            episode=self._episode,
            task_id=self._task_id,
            resolution_status=self._resolution_status,
            action_history=list(self._action_history),
            last_refund_amount=self._last_refund_amount,
        )

    # ── private action handlers ──────────────────────────────────────────

    def _handle_respond(self, action: Action, info: dict[str, Any]) -> None:
        """Log the agent's response and simulate a customer reply."""
        agent_msg = action.message or "Thank you for reaching out. Let me help you with that."
        self._observation.conversation_history.append(f"AGENT: {agent_msg}")
        info["agent_action"] = "respond"

        # Simulated customer follow-up
        customer_reply = get_customer_response(self._task_id, "respond")
        self._observation.conversation_history.append(f"CUSTOMER: {customer_reply}")
        self._observation.customer_message = customer_reply

    def _handle_ask_clarification(self, action: Action, info: dict[str, Any]) -> None:
        """Log the clarification request and simulate customer providing info."""
        agent_msg = action.message or "Could you please provide more details about your issue?"
        self._observation.conversation_history.append(f"AGENT: {agent_msg}")
        info["agent_action"] = "ask_clarification"

        # Customer provides clarifying information
        customer_reply = get_customer_response(self._task_id, "ask_clarification")
        self._observation.conversation_history.append(f"CUSTOMER: {customer_reply}")
        self._observation.customer_message = customer_reply

    def _handle_refund(self, action: Action, info: dict[str, Any]) -> None:
        """Process a refund and end the episode."""
        refund_amt = action.refund_amount or 0.0
        self._last_refund_amount = refund_amt
        agent_msg = (
            action.message
            or f"I've processed a refund of ${refund_amt:.2f} to your original payment method."
        )
        self._observation.conversation_history.append(f"AGENT: {agent_msg}")
        info["agent_action"] = "refund"
        info["refund_amount"] = refund_amt

        # Customer acknowledgement
        customer_reply = get_customer_response(self._task_id, "refund")
        self._observation.conversation_history.append(f"CUSTOMER: {customer_reply}")
        self._observation.customer_message = customer_reply

        self._resolution_status = "refunded"
        self._done = True

    def _handle_escalate(self, action: Action, info: dict[str, Any]) -> None:
        """Escalate the ticket to a senior agent and end the episode."""
        agent_msg = (
            action.message
            or "I'm escalating your ticket to a senior support specialist who can assist further."
        )
        self._observation.conversation_history.append(f"AGENT: {agent_msg}")
        info["agent_action"] = "escalate"

        # Customer acknowledgement
        customer_reply = get_customer_response(self._task_id, "escalate")
        self._observation.conversation_history.append(f"CUSTOMER: {customer_reply}")
        self._observation.customer_message = customer_reply

        self._resolution_status = "escalated"
        self._done = True
