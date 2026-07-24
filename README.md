<h1 align="center">Agentic Nash Marketing</h1>
<p align="center"><b>Neuro-Symbolic Multi-Agent Competitive Ad Auction with Nash Equilibrium</b></p>

<p align="center"><sub>FastAPI · SQLAlchemy · NumPy · SciPy · Ollama · vLLM · OpenTelemetry · Docker</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Production Ready">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tests-92%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/LLM-Mock%20%7C%20Ollama%20%7C%20vLLM%20%7C%20Transformers-yellow" alt="LLM Backends">
  <img src="https://img.shields.io/badge/Tracing-OpenTelemetry-blueviolet" alt="OTel">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Autonomous AI brand agents compete in real-time ad auctions using a **neuro-symbolic architecture**: a neural LLM proposes strategies, a symbolic planner reasons about market trends, a guardrail enforces budget rules, and a vectorized Nash equilibrium solver computes optimal mixed strategies — all with full OpenTelemetry distributed tracing.

---

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Architecture Overview](#architecture-overview)
  - [Neuro-Symbolic Pipeline](#neuro-symbolic-pipeline)
  - [Execution Flow](#execution-flow)
  - [Async Parallel Execution](#async-parallel-execution)
  - [Nash Equilibrium Solver](#nash-equilibrium-solver)
  - [Observability](#observability)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [LLM Backend Configuration](#llm-backend-configuration)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

---

## Why This Matters

| Problem | Impact | Our Solution |
|:--------|:-------|:-------------|
| Advertisers waste 30%+ of spend on suboptimal bidding | Proven optimal strategies exist via Nash equilibrium | Post-hoc Nash solver with vectorized Monte Carlo |
| Auction platforms lose revenue from unstable bidding wars | Equilibrium stabilizes clearing prices | Game-theoretic mixed-strategy convergence |
| Campaign managers rely on rules-of-thumb, not game theory | Data-driven strategy replaces intuition | LLM-powered agents with adaptive planning |
| Multi-agent systems are slow when agents think sequentially | 5 agents x 2s = 10s per round | `asyncio.gather` parallel bid computation |
| Debugging concurrent LLM calls is impossible with logs alone | Can't see who decided what and why | OpenTelemetry distributed tracing |

---

## Architecture Overview

### Neuro-Symbolic Pipeline

The system implements a three-layer neuro-symbolic architecture where neural and symbolic components collaborate at each decision point:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUCTION ROUND (async)                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: Parallel Neural Execution (asyncio.gather)     │   │
│  │                                                          │   │
│  │  Agent A ──► StrategyPlanner ──► LLM Synthesizer ──► Bid │   │
│  │  Agent B ──► StrategyPlanner ──► LLM Synthesizer ──► Bid │   │
│  │  Agent C ──► StrategyPlanner ──► LLM Synthesizer ──► Bid │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: Symbolic Enforcement (sequential, CPU-bound)    │   │
│  │                                                          │   │
│  │  BudgetGuardrail ──► VCG Auction Resolution              │   │
│  │  (soft/hard/emergency caps)  (second-price allocation)   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Layer 1 — Neural Planning (`core/planner.py`):** Queries the LLM with budget %, recent win rate, and clearing prices to select a tactical strategy (`aggressive`, `balanced`, or `conserve`). The agent adapts its persona based on market conditions — e.g., shifting to `conserve` when budget drops below 30%.

**Layer 2 — Neural Synthesis (`core/agents.py` → `core/llm_engine.py`):** The LLM receives the tactical strategy as its role and generates a structured JSON bid with justification, max daily spend, and target CPA.

**Layer 3 — Symbolic Enforcement (`core/auction.py` → `core/guardrails.py`):** The `BudgetGuardrail` intercepts every raw bid and applies multi-layer rules:
- **Soft warning** at 20% remaining budget
- **Hard cap** (5% of remaining) at 10%
- **Emergency floor** (1% of total) at 5%

The VCG second-price mechanism then allocates impressions and determines payments.

### Execution Flow

```
POST /simulation/run
    │
    ├─► SimulationRunner (core/simulation_runner.py)
    │       │
    │       └─► for round in num_rounds:
    │               await engine.run_round(agents)
    │                   │
    │                   ├─► Phase 1: asyncio.gather(
    │                   │       agent.decide_bid(),  ← per agent
    │                   │       agent.decide_bid(),
    │                   │       ...
    │                   │   )
    │                   │
    │                   └─► Phase 2: guardrail.check() → VCG resolution
    │
    ├─► NashEquilibriumSolver (post-hoc)
    │       vectorized Monte Carlo (5000 samples)
    │
    └─► Persist results to database
```

### Async Parallel Execution

All LLM inferences within a single auction round execute concurrently:

```python
# core/auction.py — Phase 1
bid_tasks = [
    agent.decide_bid(market_price=..., recent_history=...)
    for agent in active_agents
]
raw_results = await asyncio.gather(*bid_tasks)  # All agents think simultaneously
```

With real LLM latency (e.g., 2s per call), a 5-agent round drops from ~10s sequential to ~2s parallel.

### Nash Equilibrium Solver

The post-hoc Nash solver (`core/nash_solver.py`) computes optimal mixed strategies using:

1. **Vectorized Monte Carlo** — `np.random.choice` samples opponent bids in bulk; broadcasting compares all candidate bids simultaneously (no Python loops over bid levels)
2. **Iterative best-response** with softmax temperature annealing
3. **Per-agent bid levels** derived from CPA × role range for differentiated equilibria

```text
for iteration in range(max_iterations):
    for each agent:
        utilities = vectorized_expected_utility(my_bids, opponent_strategies)
        new_strategy = softmax(utilities / temperature)  # annealing
    if max_strategy_change < tolerance:
        break  # Nash equilibrium found
```

**Performance:** 3-agent solve in ~0.2s, 5-agent in ~1s (5000 MC samples).

### Observability

OpenTelemetry tracing is woven through the entire execution chain:

```text
auction_round                              (core/auction.py)
├── agent_decide_bid                       (core/agents.py)
│   ├── planner_evaluate                   (core/planner.py)
│   │   └── llm_inference                  (core/llm_engine.py)
│   └── llm_inference                      (core/llm_engine.py)
├── agent_decide_bid                       (parallel, per agent)
│   └── ...
└── [guardrail enforcement + VCG resolution]
```

**Span attributes** capture: agent name/strategy/bid, planner budget/win-rate, LLM model/provider/latency/tokens, auction impressions/revenue/clearing price, and guardrail interventions.

Setup:
```python
from core.telemetry import setup_telemetry
setup_telemetry("my-service", export_to_console=True)
```

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/aragit/nash-marketing-agents.git
cd nash-marketing-agents
docker-compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard.

### Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### With Ollama

```bash
# Start Ollama with a model
ollama pull llama3

# Set environment variable
export LLM_BACKEND=ollama
export LLM_MODEL=llama3

uvicorn api.main:app --reload
```

### With vLLM

```bash
# Start vLLM server (separate process)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

# Set environment variable
export LLM_BACKEND=vllm
export LLM_MODEL=meta-llama/Llama-3-8B

uvicorn api.main:app --reload
```

---

## Project Structure

```
nash-marketing-agents/
├── api/
│   ├── main.py              # FastAPI routes + background task orchestration
│   └── schemas.py           # Pydantic request/response models
├── core/
│   ├── agents.py            # BrandAgent — autonomous bidding agent
│   ├── auction.py           # AuctionEngine — VCG second-price mechanism
│   ├── guardrails.py        # BudgetGuardrail — multi-layer budget enforcement
│   ├── llm_engine.py        # LLM backends (Mock, Ollama, vLLM, Transformers)
│   ├── market.py            # MarketSimulator — stochastic impression supply
│   ├── nash_solver.py       # NashEquilibriumSolver — vectorized Monte Carlo
│   ├── planner.py           # StrategyPlanner — neural tactical reasoning
│   ├── prompts.py           # BrandPrompt — role-specific LLM prompts
│   ├── simulation_runner.py # SimulationRunner — decoupled execution loop
│   └── telemetry.py         # OpenTelemetry tracer setup
├── configs/
│   └── settings.py          # Pydantic Settings (env-based config)
├── database/
│   ├── connection.py        # Database init utilities
│   └── models.py            # SQLAlchemy models (Simulation, Agent, Round)
├── scripts/
│   └── run_simulation.py    # CLI simulation runner
├── static/
│   └── index.html           # Dashboard UI (Chart.js)
├── tests/
│   ├── conftest.py          # Shared fixtures (MockLLM, agent factories)
│   ├── test_agents.py       # Agent state, bid generation, async execution
│   ├── test_api.py          # Endpoint health, Nash compute, error handling
│   ├── test_auction.py      # VCG mechanics, budget depletion, history
│   ├── test_e2e.py          # Full simulation lifecycle, role differentiation
│   ├── test_guardrails.py   # Soft/hard/emergency caps, system status
│   ├── test_llm_engine.py   # Ollama, vLLM, factory, contract tests
│   ├── test_nash.py         # Convergence, distributions, performance benchmarks
│   ├── test_planner.py      # Neural planner, fallback, win-rate calculator
│   └── test_properties.py   # Economic invariants (monotonicity, IR, Nash bounds)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## LLM Backend Configuration

The system supports four LLM backends, swappable via environment variable or factory function:

| Backend | Provider String | Use Case | Requires |
|:--------|:---------------|:---------|:---------|
| **MockLLM** | `mock` | CI/CD, fast local dev | Nothing |
| **Ollama** | `ollama` | Local GPU inference | `ollama` package + running server |
| **vLLM** | `vllm` | High-throughput serving | `openai` package + running server |
| **Transformers** | `transformers` | CPU/GPU inference | `torch`, `transformers` |

### Factory Pattern

```python
from core.llm_engine import create_llm_engine

# Mock (default)
llm = create_llm_engine("mock")

# Ollama
llm = create_llm_engine("ollama", model="llama3", host="http://localhost:11434")

# vLLM (OpenAI-compatible)
llm = create_llm_engine("vllm", model="meta-llama/Llama-3-8B",
                         base_url="http://localhost:8000/v1")

# Legacy factory still works
from core.llm_engine import LLMEngineFactory
llm = LLMEngineFactory.create(use_mock=True)
```

### Configuration via Environment

```bash
# .env
LLM_BACKEND=ollama          # mock | ollama | vllm | transformers
LLM_MODEL=llama3
DATABASE_URL=sqlite:///./nash_marketing.db
LOG_LEVEL=INFO
```

---

## How It Works

1. **Configure** — Set brand names, roles, budgets, and target CPAs via the dashboard or API.

2. **Plan** — Each agent's `StrategyPlanner` queries the LLM with budget %, recent win rate, and clearing prices to select a tactical strategy (`aggressive`, `balanced`, or `conserve`).

3. **Synthesize** — The LLM receives the tactical strategy as its role and generates a structured bid (amount, max spend, justification) as JSON.

4. **Enforce** — The `BudgetGuardrail` intercepts every raw bid and applies multi-layer caps (soft warning → hard cap → emergency floor).

5. **Auction** — A second-price VCG auction allocates impressions to highest bidders. Winners pay the next-highest bid. Guardrail interventions are logged for observability.

6. **Iterate** — Steps 2-5 repeat for each round. Early termination if fewer than 2 agents have remaining budget.

7. **Equilibrium** — After all rounds, the Nash solver computes optimal mixed strategies using vectorized Monte Carlo with 5000 samples per bid level.

8. **Analyze** — The dashboard renders budget depletion, agent performance radar, clearing price history, bid ranges, cumulative spend, and Nash equilibrium distribution.

---

## Testing

```bash
pytest tests/ -v
```

**92 tests** covering:

| Module | Tests | What's Verified |
|:-------|:------|:----------------|
| `test_agents.py` | 9 | Agent initialization, async bid generation, state updates, CPA calculation |
| `test_auction.py` | 6 | Empty auction, scarce supply, clearing price, revenue matching, budget depletion |
| `test_properties.py` | 7 | **Monotonicity** (higher CPA → higher win rate), **Individual rationality** (no overpay), **Nash bounds** (convergence, expected bid ≤ valuation) |
| `test_nash.py` | 12 | Convergence, distributions, clearing price bounds, per-agent levels, degenerate strategies, **performance benchmarks** |
| `test_guardrails.py` | 6 | Soft warning, hard cap, emergency mode, system status aggregation |
| `test_planner.py` | 14 | Neural strategy selection, fallback on invalid/empty/exception, win-rate calculator, history injection |
| `test_llm_engine.py` | 23 | Ollama response parsing, vLLM response parsing, factory pattern, contract tests, edge cases |
| `test_api.py` | 7 | Health endpoint, simulation lifecycle, Nash compute, error handling |
| `test_e2e.py` | 6 | Full simulation lifecycle, role differentiation, budget guardrails, clearing price history, Nash equilibrium |

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **LLM Backends** | MockLLM (instant) · Ollama (local GPU) · vLLM (high-throughput) · Transformers (CPU/GPU) |
| **Planning** | Neural StrategyPlanner (LLM-powered tactical reasoning) |
| **Guardrails** | BudgetGuardrail (3-tier symbolic enforcement) |
| **Auction** | VCG second-price mechanism |
| **Nash Solver** | Vectorized Monte Carlo (NumPy), iterative best-response with softmax |
| **Observability** | OpenTelemetry (distributed tracing) |
| **Async** | `asyncio.gather` for parallel agent inference |
| **API** | FastAPI + Pydantic v2 |
| **ORM** | SQLAlchemy 2.0 |
| **Database** | SQLite (local) / PostgreSQL (production) |
| **Dashboard** | Vanilla JS + Chart.js |
| **Container** | Docker + docker-compose |
| **Testing** | pytest + pytest-asyncio + unittest.mock |

---

## API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | System health check (API status, LLM backend, database) |
| `POST` | `/simulation/run` | Start a new auction simulation (runs async, returns immediately) |
| `GET` | `/simulations` | List all past simulations (newest first) |
| `GET` | `/simulation/{id}` | Get full simulation detail (agents, rounds, Nash equilibrium) |
| `POST` | `/nash/compute` | Compute Nash equilibrium for arbitrary agent configurations |

---

## Dashboard

| Feature | Description |
|:--------|:------------|
| **System Health** | Real-time API, LLM backend, database status indicators |
| **Quick Stats** | Total simulations, last clearing price, cumulative revenue |
| **Run Simulation** | Configure agent count, strategies, budgets, CPA targets, rounds |
| **Budget Depletion Chart** | Grouped bar chart of amount spent vs remaining per agent |
| **Agent Performance (Radar)** | Multi-dimensional radar of win rate, budget utilization, CPA efficiency |
| **Clearing Price History** | Line chart of market dynamics across rounds |
| **Bid Range per Agent** | Min/avg/max bid range per agent |
| **Cumulative Spend** | Per-agent spend trajectory across rounds |
| **Nash Equilibrium (Polar Area)** | Expected bid distribution per agent at equilibrium |
| **Event Log** | Real-time stream of simulation events and agent decisions |

<p align="center">
  <img src="assets/run_simu.png" alt="Run Simulation Form" width="600px">
</p>

<p align="center">
  <img src="assets/run_charts.png" alt="Simulation Charts" width="900px">
</p>

---

## Refactoring History

This codebase underwent an 8-step architectural refactor:

| Step | What Changed | Key Files |
|:-----|:-------------|:----------|
| **1. SimulationRunner** | Extracted execution loop from API layer into isolated `SimulationRunner` class | `core/simulation_runner.py` |
| **2. Ghost Guardrail** | Removed hardcoded budget cap from agent; wired `BudgetGuardrail` into `AuctionEngine` | `core/auction.py`, `core/agents.py` |
| **3. Multi-Turn Planner** | Added `StrategyPlanner` for temporal awareness — agents adapt strategy based on market history | `core/planner.py` |
| **4. Neural Planner** | Upgraded `StrategyPlanner` from heuristic if/else to LLM-powered reasoning | `core/planner.py`, `core/llm_engine.py` |
| **5. Nash Solver Optimization** | Vectorized Monte Carlo with NumPy broadcasting — 5-agent solve in <1s | `core/nash_solver.py` |
| **6. Async Execution** | Converted entire chain to `async/await`; `asyncio.gather` for parallel agent inference | `core/auction.py`, `core/agents.py`, `core/planner.py` |
| **7. LLM Backend Support** | Added native async `OllamaEngine` and `VLLMEngine` with factory pattern | `core/llm_engine.py` |
| **8. OpenTelemetry** | Distributed tracing across `auction_round` → `agent_decide_bid` → `planner_evaluate` → `llm_inference` | `core/telemetry.py` |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes and run tests: `pytest tests/ -v`
4. Commit: `git commit -m "feat: describe your change"`
5. Push: `git push origin feat/your-feature`
6. Open a Pull Request against `main`

Please ensure all 92 tests pass before submitting.

---

## License

MIT — see [LICENSE](LICENSE) for details.
