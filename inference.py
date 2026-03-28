"""Hackathon inference script -- runs the baseline LLM agent on all tasks.

This script drives the agent DIRECTLY against the Environment (no HTTP server
needed).  It reuses the existing baseline agent logic and prints the required
output format.

Required environment variables:
    OPENAI_API_KEY   - Your OpenAI API key

Optional environment variables:
    MODEL_NAME       - Model to use (default: gpt-4o-mini)
    API_BASE_URL     - Custom OpenAI-compatible base URL (default: OpenAI)

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASK_IDS = ["easy_refund", "medium_missing_info", "hard_fraud"]

# Runtime safety: abort remaining tasks if we approach the 20-minute limit.
# 18 minutes leaves a 2-minute buffer for cleanup / output.
MAX_TOTAL_SECONDS = 18 * 60  # 1080 seconds


def main() -> None:
    # ── Read configuration from environment ──────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_base_url = os.environ.get("API_BASE_URL", None)

    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set. "
            "Please set it before running: "
            "$env:OPENAI_API_KEY='sk-...' (PowerShell) or "
            "export OPENAI_API_KEY='sk-...' (bash)"
        )
        sys.exit(1)

    # ── Import project modules ───────────────────────────────────────
    from baseline.agent import BaselineAgent, run_single_task_direct
    from env.environment import Environment

    # ── Initialise agent and environment ─────────────────────────────
    logger.info("Model: %s | Temperature: 0.0", model_name)
    if api_base_url:
        logger.info("Base URL: %s", api_base_url)

    try:
        agent = BaselineAgent(
            api_key=api_key,
            model=model_name,
            temperature=0.0,
            base_url=api_base_url,
        )
    except Exception as exc:
        logger.error("Failed to initialise agent: %s", exc)
        sys.exit(1)

    env = Environment(max_steps=10)

    # ── Run all tasks ────────────────────────────────────────────────
    print("=" * 60)
    print("INFERENCE RUN")
    print("=" * 60)

    task_results: dict[str, Any] = {}
    start_time = time.time()

    for task_id in TASK_IDS:
        # Runtime safety: check elapsed time before starting next task
        elapsed_so_far = time.time() - start_time
        if elapsed_so_far >= MAX_TOTAL_SECONDS:
            logger.warning(
                "Runtime limit approaching (%.0fs elapsed). "
                "Skipping remaining tasks.",
                elapsed_so_far,
            )
            task_results[task_id] = {
                "task_id": task_id,
                "score": 0.0,
                "passed": False,
                "steps": 0,
                "actions": [],
                "error": "skipped — runtime limit reached",
            }
            continue

        print()
        logger.info("--- Task: %s ---", task_id)

        try:
            result = run_single_task_direct(agent, env, task_id)
            task_results[task_id] = result

            logger.info(
                "  Score: %.2f | Passed: %s | Steps: %d | Actions: %s",
                result["score"],
                result["passed"],
                result["steps"],
                result["actions"],
            )

        except Exception as exc:
            logger.error("  Task %s failed: %s", task_id, exc)
            task_results[task_id] = {
                "task_id": task_id,
                "score": 0.0,
                "passed": False,
                "steps": 0,
                "actions": [],
                "error": str(exc),
            }

    elapsed = round(time.time() - start_time, 2)

    # ── Build and print final output ─────────────────────────────────
    task_scores = {tid: res["score"] for tid, res in task_results.items()}
    scores = list(task_scores.values())
    average_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    final_output = {
        "task_scores": task_scores,
        "average_score": average_score,
    }

    print()
    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(json.dumps(final_output, indent=2))
    print()

    # ── Extended summary (informational) ─────────────────────────────
    for tid, res in task_results.items():
        status = "PASS" if res.get("passed") else "FAIL"
        print(f"  [{status}] {tid}: score={res['score']:.2f}, steps={res.get('steps', 0)}, actions={res.get('actions', [])}")

    print()
    print(f"  Average Score : {average_score:.4f}")
    print(f"  Total Time    : {elapsed}s")
    print(f"  Model         : {model_name}")
    print()


if __name__ == "__main__":
    main()
