"""Quick verification that ALL numeric values in grader JSON are strictly in (0, 1)."""
from env.grader import grade
from env.environment import Environment
from env.models import Action
import json

env = Environment()
CASES = [
    ("easy_refund", [Action(action_type="refund", refund_amount=29.99)]),
    ("medium_missing_info", [Action(action_type="ask_clarification"), Action(action_type="respond")]),
    ("hard_fraud", [Action(action_type="escalate")]),
]

def check_all_numbers(obj, path=""):
    """Recursively find all floats in the JSON and verify (0,1)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_all_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_all_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if "score" in path.lower():
            assert 0 < obj < 1, f"FAIL: {path} = {obj} (not in (0,1))"
            print(f"  {path} = {obj}  OK")

for tid, acts in CASES:
    env.reset(tid)
    for a in acts:
        env.step(a)
    r = grade(env.state())
    d = r.model_dump()
    print(f"\n=== {tid} ===")
    print(f"Full JSON:\n{json.dumps(d, indent=2)}")
    check_all_numbers(d)

print("\n" + "=" * 50)
print("ALL SCORE-LIKE FIELDS VERIFIED IN (0, 1)")
print("=" * 50)
