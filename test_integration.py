"""Quick deployment readiness test."""
import httpx, sys

c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=10)
ok = 0
fail = 0

def t(name, r, checks=None):
    global ok, fail
    good = r.status_code == 200
    d = r.json()
    if checks:
        for ch in checks:
            if not ch(d): good = False
    if good: ok += 1
    else: fail += 1
    print(f"  [{'PASS' if good else 'FAIL'}] {name} -> {r.status_code}")

# Root
t("GET /", c.get("/"), [lambda d: d["message"] == "OpenEnv Customer Support Environment Running"])

# Health
t("GET /health", c.get("/health"), [lambda d: d["status"] == "ok"])

# Tasks
t("GET /tasks", c.get("/tasks"), [lambda d: len(d) == 3])

# Reset + step + grader
c.post("/reset", json={"task_id": "easy_refund"})
r = c.post("/step", json={"action_type": "refund", "refund_amount": 29.99})
t("POST /step", r, [lambda d: d["done"] is True])
t("GET /grader", c.get("/grader"), [lambda d: d["score"] == 1.0])
t("GET /state", c.get("/state"), [lambda d: d["resolution_status"] == "refunded"])

# Baseline (no API key = graceful error, not crash)
r = c.get("/baseline")
t("GET /baseline (no key)", r, [
    lambda d: d.get("error") == "OPENAI_API_KEY not set"
])

print(f"\n  {ok} passed, {fail} failed")
c.close()
sys.exit(1 if fail else 0)
