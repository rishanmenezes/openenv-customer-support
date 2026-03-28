"""Baseline inference script — runs the LLM agent against the OpenEnv API.

Usage:
    # Make sure the server is running first:
    #   uvicorn api.app:app --host 127.0.0.1 --port 8000
    #
    # Then run:
    #   python -m baseline.run_agent
    #   python -m baseline.run_agent --base-url http://localhost:8000
    #   python -m baseline.run_agent --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import httpx

from baseline.agent import BaselineAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASK_IDS = ["easy_refund", "medium_missing_info", "hard_fraud"]


# ── Single-task runner (HTTP mode) ───────────────────────────────────────────

def run_task_http(
    agent: BaselineAgent,
    client: httpx.Client,
    task_id: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Run the agent on a single task via HTTP endpoints.

    Returns:
        Dict with task_id, score, passed, steps, actions, grade_details.
    """
    logger.info("--- Task: %s ---", task_id)

    # Reset
    r = client.post("/reset", json={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()
    actions_taken: list[str] = []
    step_rewards: list[float] = []

    for step_num in range(1, max_steps + 1):
        # Agent decides
        action_dict = agent.decide(obs)
        action_type = action_dict.get("action_type", "respond")
        actions_taken.append(action_type)

        logger.info(
            "  Step %d: %s%s",
            step_num,
            action_type,
            f" (${action_dict['refund_amount']:.2f})"
            if action_dict.get("refund_amount") is not None
            else "",
        )
        if action_dict.get("message"):
            logger.info("    Message: %s", action_dict["message"][:100])

        # Step
        r = client.post("/step", json=action_dict)
        r.raise_for_status()
        step_result = r.json()

        step_rewards.append(step_result["reward"]["value"])
        obs = step_result["observation"]

        if step_result["done"]:
            logger.info("  Episode ended: %s", step_result["info"].get("resolution_status", "?"))
            break

    # Grade
    r = client.get("/grader")
    r.raise_for_status()
    grade_result = r.json()

    logger.info(
        "  Score: %.2f | Passed: %s | Steps: %d",
        grade_result["score"],
        grade_result["passed"],
        len(actions_taken),
    )

    return {
        "task_id": task_id,
        "score": grade_result["score"],
        "passed": grade_result["passed"],
        "steps": len(actions_taken),
        "actions": actions_taken,
        "cumulative_reward": round(sum(step_rewards), 4),
        "grade_details": grade_result["details"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the baseline LLM agent on all tasks.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running OpenEnv API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: reads from OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    # ── Validate API key ─────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
            "or pass --api-key <key>."
        )
        sys.exit(1)

    # ── Check server is reachable ────────────────────────────────────
    client = httpx.Client(base_url=args.base_url, timeout=30)
    try:
        r = client.get("/health")
        r.raise_for_status()
        logger.info("Server is healthy at %s", args.base_url)
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        logger.error("Cannot reach server at %s: %s", args.base_url, e)
        logger.error("Make sure the server is running: uvicorn api.app:app --port 8000")
        sys.exit(1)

    # ── Initialise agent ─────────────────────────────────────────────
    try:
        agent = BaselineAgent(api_key=api_key, model=args.model, temperature=0.0)
    except ValueError as e:
        logger.error("Agent init failed: %s", e)
        sys.exit(1)

    logger.info("Model: %s | Temperature: 0.0", args.model)
    print()
    print("=" * 60)
    print("BASELINE AGENT RUN")
    print("=" * 60)

    # ── Run all tasks ────────────────────────────────────────────────
    task_results: dict[str, Any] = {}
    start_time = time.time()

    for task_id in TASK_IDS:
        print()
        try:
            result = run_task_http(agent, client, task_id)
            task_results[task_id] = result
        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e)
            task_results[task_id] = {
                "task_id": task_id,
                "score": 0.0,
                "passed": False,
                "steps": 0,
                "actions": [],
                "error": str(e),
            }

    elapsed = round(time.time() - start_time, 2)

    # ── Summary ──────────────────────────────────────────────────────
    task_scores = {tid: res["score"] for tid, res in task_results.items()}
    scores = list(task_scores.values())
    average_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    summary = {
        "task_scores": task_scores,
        "average_score": average_score,
        "all_passed": all(res.get("passed", False) for res in task_results.values()),
        "total_steps": sum(res.get("steps", 0) for res in task_results.values()),
        "elapsed_seconds": elapsed,
        "model": args.model,
    }

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print()

    # Per-task detail
    for tid, res in task_results.items():
        status = "PASS" if res.get("passed") else "FAIL"
        print(f"  [{status}] {tid}: score={res['score']:.2f}, steps={res.get('steps', 0)}, actions={res.get('actions', [])}")

    print()
    print(f"  Average Score: {average_score:.4f}")
    print(f"  Time: {elapsed}s")
    print()

    client.close()

    # Exit with failure if any task didn't pass
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
