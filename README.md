# OpenEnv: Smart Customer Support Ticket Resolution

A deterministic reinforcement learning environment for training and evaluating AI agents on **customer support ticket resolution**. Built to the OpenEnv specification with a FastAPI-based HTTP interface, dense reward shaping, and a deterministic grading system.

> Customer support automation is one of the highest-impact applications of AI in production today. This environment distills the core decision-making challenges — information gathering, policy compliance, fraud detection, and efficient resolution — into a structured RL task that any agent (rule-based, LLM, or RL) can interact with through a clean API.

## Project Overview

This project implements a complete OpenEnv-compliant reinforcement learning environment specifically designed for customer support ticket resolution. It provides:

- **Deterministic RL Environment**: Three carefully crafted tasks of increasing difficulty (easy, medium, hard)
- **FastAPI HTTP Interface**: RESTful API endpoints for environment interaction
- **Dense Reward Shaping**: Per-step rewards with task-specific feedback mechanisms
- **Deterministic Grading System**: Post-episode evaluation with detailed scoring breakdowns
- **Baseline LLM Agent**: Reference implementation using OpenAI GPT models
- **Validator-Safe Design**: All outputs clamped to avoid edge cases in automated evaluation
- **Docker Support**: Containerized deployment for consistent environments

### Technology Stack

- **Backend**: FastAPI with Uvicorn server
- **Data Models**: Pydantic for type-safe serialization
- **LLM Integration**: OpenAI API for baseline agent
- **Environment**: OpenEnv-core specification compliance
- **Packaging**: Python setuptools with pyproject.toml
- **Containerization**: Docker with multi-stage support

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

Reward is computed before state mutation to ensure causality and prevent future-state leakage.

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
Openenv Customer Support/
├── api/
│   ├── __init__.py           # API package initialization
│   └── app.py                # FastAPI application with HTTP endpoints
├── baseline/
│   ├── __init__.py           # Baseline agent package initialization
│   ├── agent.py              # LLM-powered baseline agent implementation
│   └── run_agent.py          # CLI script for running agent via HTTP
├── env/
│   ├── __init__.py           # Environment package with exports
│   ├── environment.py        # Core Environment class (reset/step/state)
│   ├── models.py             # Pydantic models (Observation, Action, Reward, etc.)
│   ├── tasks.py              # Task catalogue & customer response simulation
│   ├── grader.py             # Deterministic episode grading system
│   └── reward.py             # Dense reward function (per-step, task-specific)
├── server/
│   ├── __init__.py           # Server package initialization
│   └── app.py                # CLI entry point for server command
├── .dockerignore            # Docker build exclusions
├── .gitignore               # Git exclusions
├── Dockerfile               # Container build configuration
├── inference.py             # Validator-safe inference script for evaluation
├── openenv.yaml             # OpenEnv environment metadata specification
├── pyproject.toml           # Project metadata, dependencies, and scripts
├── requirements.txt         # Python dependencies
└── README.md                # This comprehensive documentation
```

### Key Components

**API Layer (`api/`)**
- `app.py`: FastAPI application exposing HTTP endpoints for environment interaction
  - `/reset`: Initialize environment with optional task selection
  - `/step`: Submit actions and receive observations/rewards
  - `/state`: Get current environment snapshot
  - `/tasks`: List available tasks with metadata
  - `/grader`: Grade current episode performance
  - `/baseline`: Run baseline LLM agent on all tasks

**Environment Core (`env/`)**
- `environment.py`: Main `Environment` class implementing RL loop
  - Manages episode state, action history, and step counting
  - Handles action types: respond, refund, ask_clarification, escalate
  - Simulates deterministic customer responses
- `models.py`: Pydantic data models for type safety
  - `Observation`: Agent's view of the ticket state
  - `Action`: Agent's action with optional message/refund_amount
  - `Reward`: Scalar reward with explanation
  - `StepResult`: Complete step return payload
  - `EnvironmentState`: Full environment snapshot
- `tasks.py`: Task definitions and customer response simulation
  - Three tasks: easy_refund, medium_missing_info, hard_fraud
  - Pre-defined initial observations for each task
  - Deterministic customer follow-up responses
- `reward.py`: Dense reward shaping function
  - General shaping: step penalty, repeat penalty, invalid action checks
  - Task-specific rewards for each action type
  - All rewards clamped to [-0.98, +0.98]
- `grader.py`: Post-episode deterministic grading
  - Task-specific scoring criteria
  - Efficiency bonus for quick resolution
  - Final scores in (0, 1) range

**Baseline Agent (`baseline/`)**
- `agent.py`: LLM-powered reference implementation
  - Uses OpenAI API for decision making
  - Includes deterministic fallback for API failures
  - System prompt engineering for customer support context
- `run_agent.py`: CLI script for HTTP-based evaluation
  - Connects to running API server
  - Runs agent on all tasks and reports scores
  - Configurable model and API key settings

**Evaluation Scripts**
- `inference.py`: Validator-safe inference for automated evaluation
  - Includes both LLM and deterministic fallback agents
  - Reward formatting to avoid edge cases
  - Progress logging for validator compatibility

---

## Configuration Files

### `pyproject.toml`
Project metadata and dependencies configuration:
- **Project Info**: Name, version, description, and author
- **Dependencies**: fastapi, uvicorn, pydantic, openai, httpx, openenv-core
- **Python Version**: Requires Python >=3.10
- **Scripts**: Defines `server` command entry point
- **Build System**: Uses setuptools with wheel backend

### `openenv.yaml`
OpenEnv environment specification:
- **Environment Metadata**: Name, version, description, author, license
- **Observation Space**: Structured dict with ticket fields
- **Action Space**: Dict with action_type, message, and refund_amount
- **Reward Configuration**: Scalar rewards in [-1.0, 1.0] with dense shaping
- **API Specification**: FastAPI on port 7860 with defined endpoints
- **Task Definitions**: Three tasks with difficulty levels and descriptions

### `requirements.txt`
Core Python dependencies:
- fastapi: Web framework for API
- uvicorn: ASGI server
- pydantic: Data validation
- openai: LLM API client
- httpx: HTTP client for baseline agent
- openenv-core>=0.2.0: OpenEnv specification compliance

### `Dockerfile`
Container build configuration:
- **Base Image**: Python 3.10 slim
- **Working Directory**: /app
- **Dependencies**: Installs from requirements.txt
- **Port**: Exposes 7860
- **Command**: Runs uvicorn server with FastAPI app

## Installation Methods

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Openenv Customer Support
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or using uv (if available):
   uv sync
   ```

4. **Verify installation**:
   ```bash
   python -c "from env import Environment; print('Installation successful')"
   ```

### Docker Installation

1. **Build the Docker image**:
   ```bash
   docker build -t openenv-customer-support .
   ```

2. **Run the container**:
   ```bash
   docker run -p 7860:7860 openenv-customer-support
   ```

3. **Access the API**: The server will be available at `http://localhost:7860`

## Running the Server

### Starting the API Server

**Using uvicorn directly**:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 7860
```

**Using the server command**:
```bash
python -m server.app
```

**With Docker**:
```bash
docker run -p 7860:7860 openenv-customer-support
```

The API will be available at **http://localhost:7860** with interactive Swagger documentation at **http://localhost:7860/docs**

### Quick API Test

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

## Running the Baseline Agent

### Via HTTP (Requires Running Server)

```bash
# Make sure the server is running first
uvicorn api.app:app --host 127.0.0.1 --port 8000

# Run the baseline agent
python -m baseline.run_agent \
  --base-url http://localhost:8000 \
  --model gpt-4o-mini \
  --api-key YOUR_OPENAI_API_KEY
```

**Options**:
- `--base-url`: API server URL (default: http://127.0.0.1:8000)
- `--model`: OpenAI model to use (default: gpt-4o-mini)
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY environment variable)

### Via API Endpoint

The `/baseline` endpoint runs the baseline agent directly:

```bash
curl http://localhost:7860/baseline
```

This requires the `OPENAI_API_KEY` environment variable to be set.

### Docker with Baseline Agent

```bash
# Build
docker build -t openenv-customer-support .

# Run with OpenAI API key
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-customer-support
```

### Hugging Face Spaces Deployment

1. Create a new Space with **Docker** SDK
2. Push this repository to the Space
3. Add `OPENAI_API_KEY` as a Space secret for the baseline agent
4. The app boots on port `7860` with zero required environment variables

---

## Running Inference Script

The `inference.py` script is the **primary entry point** for automated evaluation. It runs agents directly against the environment (no server needed) and prints validator-safe progress output.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | Your OpenAI API key (uses fallback agent if not provided) |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model to use for inference |
| `API_BASE_URL` | No | OpenAI default | Custom OpenAI-compatible endpoint |
| `HF_TOKEN` | No | - | Hugging Face token for validator proxy |

### Running

```bash
# With OpenAI API key (LLM agent)
export OPENAI_API_KEY="sk-..."
python inference.py

# Without API key (uses deterministic fallback agent)
python inference.py
```

### Expected Output Format

The script prints validator-safe progress lines:

```
[START] task=easy_refund env=customer_support model=gpt-4o-mini
[STEP] step=1 action=refund reward=0.98 done=true error=null
[END] success=true steps=1 rewards=0.98
[START] task=medium_missing_info env=customer_support model=gpt-4o-mini
...
```

### Agent Implementation

The inference script includes two agent implementations:

**ProxyAgent**: LLM-powered agent using OpenAI API
- Uses structured JSON output with response_format
- Implements fallback to deterministic agent on API failures
- Parses and validates LLM responses

**FallbackAgent**: Deterministic rule-based agent
- Pre-defined playbooks for each task type
- Handles easy_refund, medium_missing_info, hard_fraud scenarios
- No external API dependencies
- Used when LLM is unavailable or fails

---

## Baseline Agent Details

The baseline agent (`baseline/agent.py`) provides a reference implementation for customer support automation using LLMs.

### System Prompt Engineering

The agent uses a carefully crafted system prompt that:

1. **Defines available actions**: respond, ask_clarification, refund, escalate
2. **Establishes rules**: When to use each action type based on context
3. **Emphasizes efficiency**: Resolve tickets in minimal steps
4. **Specifies output format**: Structured JSON without markdown

### Decision Process

```
Observation → Build User Prompt → LLM API Call → Parse JSON → Validate Action
                                    ↓
                            Fallback to Deterministic Rules
```

### Fallback Strategy

When the LLM API fails or returns invalid output:

1. **Flagged accounts**: Always escalate (safest action)
2. **Refund requests with known amount**: Issue the refund
3. **Default**: Ask for clarification (safe, non-destructive)

### Configuration Options

- **api_key**: OpenAI API key (or OPENAI_API_KEY environment variable)
- **model**: Model name (default: gpt-4o-mini)
- **temperature**: Sampling temperature (default: 0.0 for deterministic output)
- **base_url**: Custom API endpoint URL

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
| **Adversarial safety** | Hard task evaluates safe decision-making under adversarial conditions (fraud detection) |
| **Validator-safe** | All numeric outputs clamped to avoid 0.0/1.0 edge cases in automated evaluation |
| **Modular architecture** | Clear separation between environment core, API layer, and agent implementations |

## Development

### Project Structure Philosophy

The project follows a modular architecture with clear separation of concerns:

1. **Environment Layer (`env/`)**: Core RL logic independent of any interface
2. **API Layer (`api/`)**: HTTP interface using FastAPI
3. **Agent Layer (`baseline/`)**: Reference agent implementations
4. **Server Layer (`server/`)**: CLI entry points and deployment

### Adding New Tasks

To add a new task to the environment:

1. **Define the task in `env/tasks.py`**:
   ```python
   Task(
       id="new_task",
       name="New Task",
       description="What this task tests",
       difficulty="medium",
       initial_observation=Observation(...),
       expected_behaviour=[...],
   )
   ```

2. **Add customer responses in `CUSTOMER_RESPONSES`**:
   ```python
   "new_task": {
       "respond": "Customer reply to respond",
       "ask_clarification": "Customer reply to clarification",
       # ... other action types
   }
   ```

3. **Implement reward logic in `env/reward.py`**:
   ```python
   def _reward_new_task(action_type, ...):
       # Return (reward_value, reason)
   ```

4. **Implement grading logic in `env/grader.py`**:
   ```python
   def _grade_new_task(state):
       # Return score in (0, 1)
   ```

### Testing

Run the baseline agent to verify the environment:

```bash
# Start the server
uvicorn api.app:app --reload --port 7860

# Run baseline agent
python -m baseline.run_agent --base-url http://localhost:7860
```

### Code Quality

- **Type Safety**: Full type annotations with Pydantic validation
- **Error Handling**: Comprehensive exception handling with fallback strategies
- **Logging**: Structured logging for debugging and monitoring
- **Documentation**: Detailed docstrings for all modules and functions

## Deployment

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | OpenAI API key for LLM agent |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model name for inference |
| `API_BASE_URL` | No | OpenAI default | Custom API endpoint |

### Production Considerations

- **Security**: Never commit API keys; use environment variables or secrets
- **Scaling**: The FastAPI server can be scaled behind a load balancer
- **Monitoring**: Use the `/health` endpoint for liveness probes
- **Rate Limiting**: Implement rate limiting for production deployments

## Troubleshooting

### Common Issues

**Server won't start**:
- Check if port 7860 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Baseline agent fails**:
- Ensure `OPENAI_API_KEY` is set
- Verify API key has sufficient credits
- Check network connectivity to OpenAI API

**Import errors**:
- Make sure you're in the project directory
- Activate the virtual environment
- Verify Python version >= 3.10

**Docker build fails**:
- Check Docker daemon is running
- Verify no conflicts with port 7860
- Ensure sufficient disk space for image

## Contributing

Contributions are welcome! Areas for improvement:

- Additional task scenarios
- Alternative agent implementations (RL, rule-based)
- Enhanced reward shaping
- Additional language support
- Performance optimizations
- Extended documentation

## License

MIT License - See LICENSE file for details

## Author

**Rishan Menezes**

## Acknowledgments

- Built with [OpenEnv](https://openenv.ai) specification
- Uses [FastAPI](https://fastapi.tiangolo.com/) for HTTP interface
- Powered by [OpenAI](https://openai.com/) for baseline agent
