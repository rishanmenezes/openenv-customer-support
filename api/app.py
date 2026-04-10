"""FastAPI application -- exposes the OpenEnv Customer Support environment over HTTP."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import Environment
from env.grader import GradeResult, grade
from env.models import Action, EnvironmentState, Observation, StepResult
from env.tasks import Task, get_tasks

logger = logging.getLogger(__name__)

# ── App & Environment ────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv -- Smart Customer Support Ticket Resolution",
    description=(
        "An RL environment simulating customer support ticket resolution. "
        "Agents take actions (respond, refund, ask_clarification, escalate) "
        "to resolve support tickets across easy, medium, and hard scenarios."
    ),
    version="1.0.0",
)

env = Environment(max_steps=10)

TASK_IDS = ["easy_refund", "medium_missing_info", "hard_fraud"]


def _reset_environment(task_id: str | None) -> Observation:
    """Shared reset logic for GET and POST /reset."""
    try:
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Request schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Optional body for the /reset endpoint."""

    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Task to load (easy_refund, medium_missing_info, hard_fraud). "
            "If omitted, cycles through tasks in round-robin order."
        ),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/reset", response_model=Observation, summary="Reset the environment (GET)")
def reset_env_get(task_id: str | None = None) -> Observation:
    """Reset the environment. Optional query: ?task_id=easy_refund (round-robin if omitted)."""
    return _reset_environment(task_id)


@app.post("/reset", response_model=Observation, summary="Reset the environment")
def reset_env_post(body: ResetRequest | None = None) -> Observation:
    """Reset the environment; optional JSON body with task_id (e.g. easy_refund)."""
    tid = body.task_id if body else None
    return _reset_environment(tid)


@app.post("/step", response_model=StepResult, summary="Take a step")
def step_env(action: Action) -> StepResult:
    """Submit an action and receive (observation, reward, done, info)."""
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvironmentState, summary="Current state snapshot")
def get_state() -> EnvironmentState:
    """Return a serialisable snapshot of the environment's current state."""
    return env.state()


@app.get("/tasks", response_model=list[Task], summary="List available tasks")
def list_tasks() -> list[Task]:
    """Return the full task catalogue with initial observations and expected behaviour."""
    return get_tasks()


@app.get("/grader", response_model=GradeResult, summary="Grade current episode")
def grader_endpoint() -> GradeResult:
    """Grade the agent's performance in the current episode."""
    result = grade(env.state())
    # FORCE clamp: validator must never see 0.0 or 1.0
    clamped = min(max(result.score, 0.01), 0.99)
    result.score = clamped
    result.passed = clamped >= 0.5
    return result


@app.get("/baseline", summary="Run baseline LLM agent on all tasks")
def baseline_endpoint() -> dict[str, Any]:
    """Run the baseline LLM agent against all 3 tasks and return scores.

    Requires the OPENAI_API_KEY environment variable to be set.
    Uses a separate Environment instance so it does not disturb the main env.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "status": "error",
            "error": "OPENAI_API_KEY not set",
            "message": (
                "Set the OPENAI_API_KEY environment variable "
                "(or HF Space secret) to enable the baseline agent."
            ),
        }

    try:
        from baseline.agent import BaselineAgent, run_single_task_direct
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import baseline agent: {exc}",
        ) from exc

    # Use a SEPARATE environment so the main /step /reset state is untouched
    baseline_env = Environment(max_steps=10)

    try:
        agent = BaselineAgent(api_key=api_key, model="gpt-4o-mini", temperature=0.0)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task_results: dict[str, Any] = {}

    for task_id in TASK_IDS:
        try:
            result = run_single_task_direct(agent, baseline_env, task_id)
            task_results[task_id] = result
        except Exception as exc:
            logger.exception("Baseline task %s failed", task_id)
            task_results[task_id] = {
                "task_id": task_id,
                "score": 0.01,
                "passed": False,
                "steps": 0,
                "actions": [],
                "error": str(exc),
            }

    # Clamp every task score at API level
    task_scores = {
        tid: min(max(res["score"], 0.01), 0.99)
        for tid, res in task_results.items()
    }
    scores = list(task_scores.values())
    average_score = round(sum(scores) / len(scores), 4) if scores else 0.01

    return {
        "status": "completed",
        "model": "gpt-4o-mini",
        "task_scores": task_scores,
        "average_score": average_score,
        "all_passed": all(res.get("passed", False) for res in task_results.values()),
        "task_details": task_results,
    }


# ── Root & Health ────────────────────────────────────────────────────────────

@app.get("/", summary="Root")
def root() -> dict[str, str]:
    """Root endpoint confirming the service is live."""
    return {"message": "OpenEnv Customer Support Environment Running"}


@app.get("/health", summary="Health check")
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}
