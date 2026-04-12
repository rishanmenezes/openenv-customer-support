"""Verify that NO execution path produces a score of 0.0 or 1.0."""

from env.grader import grade, GradeResult
from env.models import Reward, Action
from env.environment import Environment

print("=== Reward validator ===")
for v in [0.0, 1.0, -1.0, 0.5, -0.5, 0.99, 0.01]:
    r = Reward(value=v)
    assert r.value != 0.0, f"FAIL: Reward({v}) -> 0.0"
    assert r.value != 1.0, f"FAIL: Reward({v}) -> 1.0"
    assert r.value != -1.0, f"FAIL: Reward({v}) -> -1.0"
    print(f"  Reward({v:+.2f}) -> {r.value:+.4f}  OK")

print()
print("=== GradeResult validator ===")
for v in [0.0, 1.0, 0.5, 0.01, 0.99, -0.5, 2.0]:
    g = GradeResult(score=v)
    assert 0 < g.score < 1, f"FAIL: GradeResult({v}) -> {g.score}"
    print(f"  GradeResult({v:+.2f}) -> {g.score:.4f}  OK")

print()
print("=== Full grading: optimal agent ===")

# Optimal actions per task
OPTIMAL = {
    "easy_refund": [
        Action(action_type="refund", refund_amount=29.99),
    ],
    "medium_missing_info": [
        Action(action_type="ask_clarification"),
        Action(action_type="respond"),
    ],
    "hard_fraud": [
        Action(action_type="escalate"),
    ],
}

env = Environment()
for tid, actions in OPTIMAL.items():
    env.reset(tid)
    for a in actions:
        result = env.step(a)
        # Check step reward
        assert result.reward.value != 0.0, f"FAIL: {tid} step reward = 0.0"
        assert abs(result.reward.value) != 1.0, f"FAIL: {tid} step reward = +-1.0"

    gr = grade(env.state())
    score = gr.score
    ts = gr.details.get("task_score", -1)
    fs = gr.details.get("final_score", -1)
    rs = gr.details.get("raw_score", -1)

    print(f"  {tid}:")
    print(f"    score={score:.4f}  task_score={ts}  final_score={fs}  raw_score={rs}")
    assert 0 < score < 1, f"FAIL: {tid} score={score}"
    assert 0 < ts < 1, f"FAIL: {tid} task_score={ts}"
    assert 0 < fs < 1, f"FAIL: {tid} final_score={fs}"
    print(f"    ALL OK")

print()
print("=== Full grading: worst-case agent (no actions / timeout) ===")
env2 = Environment(max_steps=1)
for tid in ["easy_refund", "medium_missing_info", "hard_fraud"]:
    env2.reset(tid)
    # Take a neutral action that won't score well
    env2.step(Action(action_type="respond", message="Hello"))
    gr = grade(env2.state())
    score = gr.score
    ts = gr.details.get("task_score", -1)
    print(f"  {tid}: score={score:.4f}  task_score={ts}")
    assert 0 < score < 1, f"FAIL: {tid} score={score}"
    assert 0 < ts < 1, f"FAIL: {tid} task_score={ts}"
    print(f"    OK")

print()
print("=== Format safety check ===")

# Inline copies to avoid importing inference.py (which needs OpenAI key)
def safe_reward(r):
    try:
        r = float(r)
    except (TypeError, ValueError):
        return 0.01
    if r >= 1.0:
        return 0.98
    if r <= 0.0:
        return 0.01
    if round(r, 2) <= 0.0:
        return 0.01
    if round(r, 2) >= 1.0:
        return 0.98
    return r

def _safe_fmt(r):
    s = f"{r:.2f}"
    if s in ("0.00", "-0.00", "1.00", "-1.00"):
        return "0.01" if float(s) <= 0.0 else "0.98"
    return s

for v in [0.0, 0.001, 0.004, 0.005, 0.999, 1.0, -0.001, -1.0]:
    sr = safe_reward(v)
    fmt = _safe_fmt(sr)
    assert fmt not in ("0.00", "1.00", "-0.00", "-1.00"), f"FAIL: safe_reward({v})={sr} -> fmt={fmt}"
    print(f"  safe_reward({v:+.3f}) = {sr:+.4f} -> fmt={fmt}  OK")

print()
print("=" * 50)
print("ALL CHECKS PASSED - No 0.0 or 1.0 anywhere!")
print("=" * 50)
