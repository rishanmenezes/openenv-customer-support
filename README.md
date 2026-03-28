---
title: OpenEnv Customer Support
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv: Smart Customer Support Ticket Resolution

A deterministic reinforcement learning environment for training and evaluating AI agents on **customer support ticket resolution**. Built to the OpenEnv specification with a FastAPI-based HTTP interface, dense reward shaping, and a deterministic grading system.

> Customer support automation is one of the highest-impact applications of AI in production today. This environment distills the core decision-making challenges — information gathering, policy compliance, fraud detection, and efficient resolution — into a structured RL task that any agent (rule-based, LLM, or RL) can interact with through a clean API.

---

## Table of Contents

- [Environment Design](#environment-design)
- [Tasks](#tasks)
- [Reward Function](#reward-function)
- [Grader](#grader)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running Inference Script](#running-inference-script)
- [Baseline Agent](#baseline-agent)
- [Design Highlights](#design-highlights)
- [License](#license)

---

## Environment Design

### Episode Flow

```
reset(task_id) ──> Observation
       │
       ▼
  ┌─────────┐     ┌──────────┐     ┌──────────┐
  │  Agent   │────>│  step()  │────>│  Result   │
  │ decides  │     │          │     │ obs,reward│
  │ action   │<────│ env      │<────│ done,info │
  └─────────┘     └──────────┘     └──────────┘
       │                                │
       └──── loop until done ───────────┘
       │
       ▼
  grader(state) ──> Score [0.0 – 1.0]
```

Each episode represents a single support ticket. The agent reads the ticket, takes actions (respond, refund, ask for more info, or escalate), and the environment simulates deterministic customer follow-ups until the ticket is resolved or the step limit is reached.

### Observation Space

The agent receives a structured observation at every step:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket identifier (e.g., `TKT-1001`) |
| `customer_message` | `str` | The customer's latest message |
| `conversation_history` | `list[str]` | Full chat log (`AGENT: ...` / `CUSTOMER: ...`) |
| `account_status` | `str` | Customer account status: `active`, `suspended`, or `flagged` |
| `order_amount` | `float` | Dollar amount of the order in question |
| `issue_type` | `str` | Category: `refund`, `missing_info`, `fraud`, `general` |
| `step_count` | `int` | Number of actions taken so far this episode |

### Action Space

The agent submits one of four action types per step:

| Action | Purpose | Optional Fields |
|--------|---------|-----------------|
| `respond` | Send a reply to the customer | `message` |
| `ask_clarification` | Request missing information | `message` |
| `refund` | Issue a refund and close the ticket | `message`, `refund_amount` |
| `escalate` | Escalate to a senior agent / fraud team | `message` |

**Terminal actions**: `refund` and `escalate` end the episode immediately.

```json
{
  "action_type": "refund",
  "message": "I've processed your refund.",
  "refund_amount": 29.99
}
```

---

## Tasks

Three deterministic scenarios of increasing difficulty:

### Easy: Clear Refund (`easy_refund`)

> Customer received a defective product and requests a full refund of $29.99. All information is provided upfront.

- **Expected behavior**: Acknowledge the issue, then issue a refund for the correct amount.
- **Trap**: Don't escalate — it's a straightforward case.

### Medium: Missing Information (`medium_missing_info`)

> Customer is upset about a wrong item but hasn't provided key details (order number, specific issue).

- **Expected behavior**: Ask for clarification *first*, then resolve after gathering info.
- **Trap**: Responding or resolving without asking for details is penalized.

### Hard: Suspected Fraud (`hard_fraud`)

> Customer aggressively demands a $499.99 refund but can't provide order details. Account is flagged.

- **Expected behavior**: Ask probing questions, then escalate to the fraud team. **Do NOT refund.**
- **Trap**: Issuing a refund to a flagged account is catastrophic (`-1.0` reward, `0.0` grade).

---

## Reward Function

Rewards are computed **per step** with dense shaping to guide learning. All values are deterministic and clamped to `[-1.0, +1.0]`.

### General Shaping (all tasks)

| Signal | Value | Purpose |
|--------|-------|---------|
| Step penalty | `-0.05` | Encourages efficiency |
| Repeated action | `-0.10` | Discourages loops |
| Invalid refund (no amount) | `-0.20` | Ensures action validity |

### Task-Specific Rewards

**Easy (easy_refund)**

| Action | Reward |
|--------|--------|
| Correct refund ($29.99) | `+1.00` |
| Wrong refund amount | `+0.30` |
| Respond (acknowledgement) | `+0.10` |
| Unnecessary clarification | `-0.10` |
| Unnecessary escalation | `-0.50` |

**Medium (medium_missing_info)**

| Action | Reward |
|--------|--------|
| Ask clarification (first time) | `+0.40` |
| Respond after clarification | `+0.30` |
| Respond without clarification | `-0.30` |
| Refund (not requested) | `-0.30` |

**Hard (hard_fraud)**

| Action | Reward |
|--------|--------|
| Escalate to fraud team | `+1.00` |
| Ask for verification | `+0.20` |
| Refund a flagged account | `-1.00` |

---

## Grader

A deterministic **post-episode evaluator** producing a score in `[0.0, 1.0]`:

### Scoring Criteria

| Task | Criteria | Points |
|------|----------|--------|
| **Easy** | Refund action taken | +0.70 |
| | Correct refund amount | +0.30 |
| **Medium** | Asked for clarification | +0.40 |
| | Resolved after clarification | +0.60 |
| | *No clarification = capped at 0.50* | |
| **Hard** | Escalated | +0.70 |
| | Did NOT refund | +0.30 |
| | *Refund issued = forced 0.0* | |

### Efficiency Bonus

- Resolved in **2 steps or fewer**: `+0.10`
- Final score is capped at `1.0`
- **Pass threshold**: `score >= 0.5`

### Response Format

```json
{
  "score": 1.0,
  "passed": true,
  "details": {
    "task_id": "easy_refund",
    "task_score": 1.0,
    "task_breakdown": {
      "refund_action": "+0.70 (correctly issued a refund)",
      "refund_amount": "+0.30 (correct amount $29.99)"
    },
    "efficiency_bonus": "+0.10 (resolved in 2 steps)",
    "final_score": 1.0
  }
}
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root — confirms service is running |
| `GET` | `/health` | Liveness probe |
| `POST` | `/reset` | Reset environment (optional `task_id` in body) |
| `POST` | `/step` | Submit an action, receive `{observation, reward, done, info}` |
| `GET` | `/state` | Full environment state snapshot |
| `GET` | `/tasks` | List all available tasks with metadata |
| `GET` | `/grader` | Grade the current episode |
| `GET` | `/baseline` | Run baseline LLM agent on all tasks |

Interactive Swagger documentation is available at `/docs` when the server is running.

---

## Project Structure

```
Meta/
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI endpoints
├── baseline/
│   ├── __init__.py
│   ├── agent.py               # LLM-powered baseline agent
│   └── run_agent.py           # CLI script for running the agent via HTTP
├── env/
│   ├── __init__.py
│   ├── environment.py         # Environment class (reset / step / state)
│   ├── models.py              # Pydantic models (Observation, Action, Reward, ...)
│   ├── tasks.py               # Task catalogue & customer response simulation
│   ├── grader.py              # Deterministic episode grading
│   └── reward.py              # Dense per-step reward function
├── inference.py               # Hackathon inference entry point
├── openenv.yaml               # OpenEnv specification metadata
├── requirements.txt
├── Dockerfile
├── test_integration.py
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/rishanmenezes/openenv-customer-support.git
cd openenv-customer-support

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn api.app:app --reload --host 0.0.0.0 --port 7860
```

The API is now live at **http://localhost:7860**
Swagger docs at **http://localhost:7860/docs**

### Quick Test

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_refund"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "refund", "refund_amount": 29.99}'

# Check the grade
curl http://localhost:7860/grader
```

### Docker

```bash
# Build
docker build -t openenv-env .

# Run
docker run -p 7860:7860 openenv-env

# With baseline agent (requires OpenAI key)
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-env
```

### Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Push this repository to the Space
3. (Optional) Add `OPENAI_API_KEY` as a Space secret for the baseline agent
4. The app boots on port `7860` with zero required environment variables

---

## Running Inference Script

The `inference.py` script is the **primary entry point** for hackathon evaluation. It runs the baseline LLM agent directly against the environment (no server needed) and prints the required output format.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | - | Your OpenAI API key |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model to use for inference |
| `API_BASE_URL` | No | OpenAI default | Custom OpenAI-compatible endpoint |

### Running

```bash
# Set your API key
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Run inference
python inference.py
```

### Expected Output

```json
{
  "task_scores": {
    "easy_refund": 1.0,
    "medium_missing_info": 1.0,
    "hard_fraud": 1.0
  },
  "average_score": 1.0
}
```

### How It Works

1. Reads `OPENAI_API_KEY`, `MODEL_NAME`, and `API_BASE_URL` from environment
2. Creates a `BaselineAgent` (temperature=0, deterministic)
3. For each task (`easy_refund`, `medium_missing_info`, `hard_fraud`):
   - Calls `env.reset(task_id)` to load the scenario
   - Loops: sends observation to LLM, parses JSON action, calls `env.step()`
   - Continues until `done=True` or max steps reached
4. Grades each episode via `grader.grade()`
5. Prints `task_scores` and `average_score` as JSON

---

## Baseline Agent

An LLM-powered baseline agent is included to benchmark performance. It uses OpenAI's chat completions API with structured JSON output.

### Running via CLI

```bash
# Set your API key
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Start the server (terminal 1)
uvicorn api.app:app --port 7860

# Run the agent (terminal 2)
python -m baseline.run_agent --base-url http://127.0.0.1:7860
```

### Running via API

```bash
curl http://localhost:7860/baseline
```

### Example Output

```json
{
  "task_scores": {
    "easy_refund": 1.0,
    "medium_missing_info": 1.0,
    "hard_fraud": 1.0
  },
  "average_score": 1.0,
  "all_passed": true,
  "total_steps": 5,
  "model": "gpt-4o-mini"
}
```

### Agent Design

- **Model**: `gpt-4o-mini` (configurable via `--model`)
- **Temperature**: `0.0` (deterministic)
- **Output format**: Structured JSON with `response_format: json_object`
- **Fallback handling**: Invalid JSON or unknown action types fall back to `respond`

---

## Design Highlights

| Principle | Implementation |
|-----------|---------------|
| **Deterministic** | No randomness — tasks, customer responses, rewards, and grades are all reproducible |
| **Dense rewards** | Per-step reward shaping with task-specific signals, not just a final score |
| **Real-world fidelity** | Simulates actual support scenarios: refunds, missing info, fraud |
| **Typed everything** | Pydantic v2 models with full type annotations throughout |
| **Clean separation** | Environment logic, reward, grader, tasks, and API are all independent modules |
| **Graceful degradation** | Server boots without `OPENAI_API_KEY`; baseline endpoint returns a clear error |
| **OpenEnv compliant** | Standard `reset()` / `step()` / `state()` interface with `openenv.yaml` metadata |

---

## License

MIT
