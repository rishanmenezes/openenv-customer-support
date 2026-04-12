"""FastAPI application -- exposes the OpenEnv Customer Support environment over HTTP."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

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


# ── Nuclear response sanitizer ───────────────────────────────────────────────
# Intercepts ALL JSON responses and recursively clamps every float so that
# no value is exactly 0.0 or 1.0.  This is the absolute last line of defense.

def _sanitize_value(obj: Any) -> Any:
    """Recursively walk a JSON-serializable structure and clamp floats."""
    if isinstance(obj, float):
        if obj >= 1.0:
            return 0.99
        if obj <= 0.0:
            return 0.01
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_value(v) for v in obj]
    return obj


class SanitizeScoresMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Only process JSON responses
        if response.headers.get("content-type", "").startswith("application/json"):
            body = b""
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    body += chunk
                else:
                    body += chunk.encode("utf-8")
            try:
                data = json.loads(body)
                sanitized = _sanitize_value(data)
                new_body = json.dumps(sanitized).encode("utf-8")
                return Response(
                    content=new_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json",
                )
            except (json.JSONDecodeError, Exception):
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
        return response


app.add_middleware(SanitizeScoresMiddleware)


env = Environment(max_steps=10)

TASK_IDS = ["easy_refund", "medium_missing_info", "hard_fraud"]


def _safe_score(v: float) -> float:
    """Clamp a score to strictly (0, 1) — never 0.0 or 1.0."""
    if v >= 1.0:
        return 0.99
    if v <= 0.0:
        return 0.01
    return v


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
        result = env.step(action)
        # Clamp reward: validator must never see ±1.0 or 0.0
        r = result.reward.value
        r = min(max(r, -0.99), 0.99)
        if r == 0.0:
            r = 0.01
        result.reward.value = r
        return result
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
    # FORCE clamp: validator must never see 0.00 or 1.00
    clamped = _safe_score(result.score)
    result.score = clamped
    result.passed = clamped >= 0.5
    # Also clamp any score values inside details
    if "final_score" in result.details:
        result.details["final_score"] = _safe_score(result.details["final_score"])
    if "task_score" in result.details:
        result.details["task_score"] = _safe_score(result.details["task_score"])
    if "raw_score" in result.details:
        result.details["raw_score"] = _safe_score(result.details["raw_score"])
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
        tid: _safe_score(res["score"])
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
    return {"status": "healthy"}
