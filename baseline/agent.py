"""Core LLM agent for the Smart Customer Support environment.

This module provides the reusable agent logic that can be driven by:
  - The CLI script (baseline/run_agent.py) via HTTP
  - The /baseline API endpoint directly against the Environment class
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from env.models import ActionType

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a customer support agent. You will receive a support ticket observation \
and must decide the best action to take.

## Available Actions

| action_type         | When to use                                            |
|---------------------|--------------------------------------------------------|
| respond             | Reply to the customer with a helpful message           |
| ask_clarification   | Ask the customer for missing information               |
| refund              | Issue a refund (include refund_amount)                 |
| escalate            | Escalate the ticket to a senior agent or fraud team    |

## Rules

1. If the customer provides all necessary info and requests a refund, issue it with the correct amount.
2. If key details are missing (order number, issue description), ask for clarification FIRST.
3. If the account is flagged or the request seems fraudulent, ESCALATE — do NOT refund.
4. Be efficient: resolve tickets in as few steps as possible.
5. Always include a helpful message when responding or asking for clarification.

## Response Format

You MUST respond with a single JSON object — no extra text, no markdown fences:

{
  "action_type": "respond" | "refund" | "ask_clarification" | "escalate",
  "message": "Your message to the customer (optional for refund/escalate)",
  "refund_amount": 29.99  // ONLY include when action_type is "refund"
}
"""

# Valid action types for validation
_VALID_ACTIONS = {a.value for a in ActionType}


# ── Agent class ──────────────────────────────────────────────────────────────

class BaselineAgent:
    """LLM-powered baseline agent using OpenAI chat completions.

    Args:
        api_key:     OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model:       Model name (default: gpt-4o-mini for cost efficiency).
        temperature: Sampling temperature (0 for deterministic output).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key provided. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to BaselineAgent()."
            )
        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._temperature = temperature

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Given an observation dict, return a parsed action dict.

        Returns:
            Dict with keys: action_type, message (optional), refund_amount (optional).
        """
        user_prompt = self._build_user_prompt(observation)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            return self._parse_action(raw)
        except Exception as exc:
            logger.warning("LLM API call failed: %s — using fallback action", exc)
            return self._fallback_action(observation)

    # ── Fallback (deterministic heuristic when LLM is unavailable) ────

    @staticmethod
    def _fallback_action(obs: dict[str, Any]) -> dict[str, Any]:
        """Return a safe deterministic action when the LLM call fails."""
        account_status = obs.get("account_status", "active")
        issue_type = obs.get("issue_type", "general")

        # Flagged accounts → always escalate (safest action)
        if account_status == "flagged":
            return {
                "action_type": "escalate",
                "message": "Escalating your ticket to a specialist for review.",
            }

        # Refund requests with a known amount → issue the refund
        if issue_type == "refund":
            order_amount = obs.get("order_amount", 0.0)
            if order_amount > 0:
                return {
                    "action_type": "refund",
                    "message": "Processing your refund.",
                    "refund_amount": order_amount,
                }

        # Default → ask for clarification (safe, non-destructive)
        return {
            "action_type": "ask_clarification",
            "message": "Could you please provide more details about your issue?",
        }

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(obs: dict[str, Any]) -> str:
        """Format the observation into a clear prompt for the LLM."""
        lines = [
            "## Current Ticket",
            f"- Ticket ID: {obs.get('ticket_id', 'N/A')}",
            f"- Issue Type: {obs.get('issue_type', 'N/A')}",
            f"- Account Status: {obs.get('account_status', 'N/A')}",
            f"- Order Amount: ${obs.get('order_amount', 0):.2f}",
            f"- Step: {obs.get('step_count', 0)}",
            "",
            "## Latest Customer Message",
            obs.get("customer_message", "(none)"),
            "",
            "## Conversation History",
        ]
        history = obs.get("conversation_history", [])
        if history:
            for entry in history:
                lines.append(f"  {entry}")
        else:
            lines.append("  (no prior messages)")

        lines.append("")
        lines.append("Decide your next action. Respond with JSON only.")
        return "\n".join(lines)

    @staticmethod
    def _parse_action(raw: str) -> dict[str, Any]:
        """Parse and validate the LLM's JSON output into a clean action dict."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON: %s", raw[:200])
            return {"action_type": "respond", "message": "I'm looking into your issue."}

        # Validate action_type
        action_type = data.get("action_type", "respond")
        if action_type not in _VALID_ACTIONS:
            logger.warning("Invalid action_type '%s', falling back to respond", action_type)
            action_type = "respond"

        result: dict[str, Any] = {"action_type": action_type}

        # Optional message
        msg = data.get("message")
        if msg and isinstance(msg, str):
            result["message"] = msg

        # Optional refund_amount (only meaningful for refund action)
        if action_type == "refund":
            amt = data.get("refund_amount")
            if amt is not None:
                try:
                    result["refund_amount"] = float(amt)
                except (TypeError, ValueError):
                    logger.warning("Invalid refund_amount: %s", amt)
                    result["refund_amount"] = 0.0

        return result


# ── Run agent on a single task (direct, no HTTP) ────────────────────────────

def run_single_task_direct(
    agent: BaselineAgent,
    env: Any,
    task_id: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Run the agent on a single task using direct Environment calls.

    Used by the /baseline API endpoint to avoid self-referential HTTP.

    Returns:
        Dict with task_id, score, passed, steps, actions, grade_details.
    """
    from env.grader import grade

    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()
    actions_taken: list[str] = []

    for _ in range(max_steps):
        action_dict = agent.decide(obs_dict)
        actions_taken.append(action_dict.get("action_type", "respond"))

        from env.models import Action
        action = Action(**action_dict)
        result = env.step(action)

        obs_dict = result.observation.model_dump()

        if result.done:
            break

    grade_result = grade(env.state())
    raw = grade_result.score
    if raw >= 1.0:
        clamped_score = 0.99
    elif raw <= 0.0:
        clamped_score = 0.01
    else:
        clamped_score = raw

    return {
        "task_id": task_id,
        "score": clamped_score,
        "passed": grade_result.passed,
        "steps": len(actions_taken),
        "actions": actions_taken,
        "grade_details": grade_result.details,
    }
