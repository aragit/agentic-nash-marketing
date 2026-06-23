# 🏛️ Agentic Nash Marketing
<p align="center"><b>Multi-Agent Competitive Ad Auction with Nash Equilibrium</b></p>

<p align="center"><sub>FastAPI · SQLAlchemy · SciPy · Docker · pytest · Chart.js</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-🚀%20Production%20Ready-brightgreen" alt="Production Ready">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0+-orange?logo=sqlalchemy" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/SciPy-1.10+-blueviolet?logo=scipy" alt="SciPy">
  <img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Tests-36%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/LLM-Mock%20%7C%20Transformers-yellow" alt="LLM">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Autonomous AI brand agents compete in real-time ad auctions. Each agent uses an LLM to formulate bidding strategy, then a **game-theoretic Nash equilibrium** solver **computes optimal mixed strategies**. Budget guardrails prevent catastrophic depletion.

---

## 📋 Table of Contents

- [Why This Matters](#-why-this-matters)
- [Architecture](#-architecture)
  - [Agentic AI Criteria](#1-agentic-ai-criteria)
  - [Neuro-Symbolic Paradigm](#2-neuro-symbolic-paradigm)
  - [Nash Algorithm](#3-the-nash-algorithm)
- [Quick Start](#-quick-start)
  - [Docker](#docker-recommended)
  - [Local Development](#local-development)
  - [Dashboard Features](#-dashboard-features)
- [How It Works](#-how-it-works)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)
- [API Endpoints](#-api-endpoints)
- [Future Integration](#-future-integration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 💡 Why This Matters

| Problem | Impact |
|:--------|:-------|
| **Advertisers** waste 30%+ of spend on suboptimal bidding | Nash equilibrium proves optimal strategies exist |
| **Auction platforms** lose revenue from unstable bidding wars | Equilibrium stabilizes clearing prices |
| **Campaign managers** rely on rules-of-thumb, not game theory | Data-driven strategy replaces intuition |

This project replaces guesswork with mathematical guarantees. It simulates how rational agents *should* bid, then validates against real auction outcomes.

### Use Cases

- **Ad tech R&D** — Test bidding algorithms before production deployment
- **Market design** — Analyze how impression supply affects advertiser behavior
- **Education** — Interactive demonstration of Nash equilibrium in a concrete domain
- **Procurement integration** — Bridge to [autonomous procurement swarm](https://github.com/aragit/autonomous-procurement-swarm)

---

## 🏗️ Architecture

### 1. Agentic AI Criteria

An agentic AI system is defined by autonomous entities that perceive, decide, and act in an environment with persistent goals. Our system satisfies all four criteria:

| Criterion | Implementation | Evidence |
|:---|:---|:---|
| **Perception** | Agents observe market state (clearing price, competitor count, win rate) | `BrandAgent.decide_bid()` receives market context |
| **Decision** | LLM-powered strategic reasoning with structured JSON output | `LLMEngine.chat_completion()` generates bid strategy |
| **Action** | Agents submit bids to auction engine | `AuctionEngine.run_round()` executes bids |
| **Persistent goals** | Budget preservation, CPA targets, win rate optimization | `AgentState` tracks cumulative performance across rounds |

Unlike simple API wrappers, these agents maintain state across rounds, adapt strategy based on outcomes, and operate without human intervention — the definition of autonomy.

### 2. Neuro-Symbolic Paradigm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURO-SYMBOLIC ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐        ┌─────────────────────┐                     │
│  │   NEURAL (LLM)      │        │   SYMBOLIC (Math)   │                     │
│  │                     │        │                     │                     │
│  │  • Pattern matching │◄──────►│  • Nash equilibrium │                     │
│  │  • Strategy gen     │        │  • Linear programs  │                     │
│  │  • Natural lang     │        │  • Constraint opt   │                     │
│  │  • Context aware    │        │  • Probabilistic    │                     │
│  │                     │        │    inference        │                     │
│  └─────────────────────┘        └─────────────────────┘                     │
│           │                              │                                  │
│           └──────────────┬───────────────┘                                  │
│                          │                                                  │
│                          ▼                                                  │
│               ┌─────────────────────┐                                       │
│               │  HYBRID REASONING   │                                       │
│               │  Layer              │                                       │
│               │                     │                                       │
│               │  LLM proposes →     │                                       │
│               │  Symbolic validates │                                       │
│               │  Guardrail enforces │                                       │
│               └─────────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> The LLM is the **intuition** — it generates creative bidding strategies. The Nash solver is the **logic** — it proves no agent can improve by deviating. The guardrail is the **conscience** — it prevents self-destructive behavior.

### 3. The Nash Algorithm

#### The Problem: The Tragedy of the Commons in Ad Auctions

Without equilibrium analysis, agents engage in destructive bidding wars:

| Scenario | Without Nash | With Nash Equilibrium |
|:---------|:-------------|:---------------------|
| Bidding dynamics | Nike $10 → Adidas $11 → Puma $12 → Nike $13… (escalation) | Nike $3.20, Adidas $2.80, Puma $2.50 (stable) |
| CPA trajectory | Explodes every round | Predictable, bounded |
| Budget depletion | Days | Campaign-long |
| Market stability | Volatile clearing prices | Predictable clearing prices |

#### How It Works: Iterative Best-Response with Softmax

```text
for iteration in range(max_iterations):
    for each agent:
        # Compute expected utility for every bid level
        # given opponents' current mixed strategies
        utilities = [expected_profit(bid, opponent_strategies)
                     for bid in bid_levels]

        # Softmax best response (temperature annealing)
        # High temp early = exploration. Low temp late = convergence.
        new_strategy = softmax(utilities / temperature)

    # Check convergence: did any agent's strategy change significantly?
    if max_strategy_change < tolerance:
        break  # Nash equilibrium found!
```

> **Mathematical guarantee:** At convergence, no agent can improve their expected utility by changing their strategy alone. This is the definition of Nash equilibrium.

#### Runtime Flow

```text
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Configure │───▶│ Simulate │───▶│  Auction │───▶│   Nash   │───▶│ Analyze  │
│  Agents   │    │  Rounds  │    │ (VCG)    │    │  Solver  │    │Dashboard │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     │ Brand names   │ LLM decides   │ 2nd-price     │ Mixed-strategy│ Chart.js   │
     │ Budgets, CPAs │ bids per round│ allocation    │ equilibrium   │ visuals    │
     └───────────────┴───────────────┴───────────────┴───────────────┴───────────┘
```

---

## 🚀 Quick Start

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

---

## 📊 Dashboard Features

| Feature | Description |
|:--------|:------------|
| **System Health** | Real-time API, LLM backend, database status indicators |
| **Quick Stats** | Total simulations, last clearing price, cumulative revenue |
| **Run Simulation** | Configure agent count, strategies, budgets, CPA targets, rounds |
| **Budget Depletion Chart** | Grouped bar chart of amount spent vs remaining per agent |
| **Win Rate Chart** | Doughnut chart of auction success percentage per brand |
| **Clearing Price History** | Line chart of market dynamics across rounds |
| **Nash Equilibrium Chart** | Expected bid distribution per agent at equilibrium |
| **Event Log** | Real-time stream of simulation events and agent decisions |

---

## 🎮 How It Works

1. **Configure** — Set brand names, strategies (aggressive / balanced / conservative), budgets, and target CPAs via the dashboard form.
2. **Simulate** — Each round, every agent queries its LLM with current market context (clearing price, competitor count, win rate, remaining budget) and receives a structured bid decision in JSON.
3. **Auction** — A second-price VCG auction allocates impressions to the highest bidders. Winners pay the next-highest bid. Budget guardrails cap per-round spend at 20% of remaining budget.
4. **Equilibrium** — After all rounds complete, the Nash solver iteratively computes optimal mixed strategies using softmax best-response dynamics with temperature annealing.
5. **Analyze** — The dashboard renders four charts (budget, win rate, clearing price, Nash equilibrium) and an event log for post-hoc analysis.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

36 tests covering:

| Module | Tests | What's Verified |
|:-------|:------|:----------------|
| `tests/test_agents.py` | 9 | Agent initialization, bid generation, state updates, CPA calculation |
| `tests/test_auction.py` | 6 | Empty auction, scarce supply, clearing price, revenue matching, budget depletion |
| `tests/test_nash.py` | 6 | Convergence, strategy validity, expected bid ranges, clearing price bounds |
| `tests/test_guardrails.py` | 8 | Soft warning, hard cap, emergency mode, system status aggregation |
| `tests/test_api.py` | 7 | Health endpoint, simulation lifecycle, Nash compute, error handling |

---

## 📦 Tech Stack

| Layer | Technology |
|:---|:---|
| **LLM** | MockLLM (default, instant) / Transformers CPU (optional, real inference) |
| **Math** | NumPy + SciPy (Nash equilibrium, optimization) |
| **Database** | SQLite (local) / PostgreSQL (production) |
| **API** | FastAPI + Pydantic v2 |
| **ORM** | SQLAlchemy 2.0 |
| **Dashboard** | Vanilla JS + Chart.js |
| **Container** | Docker + docker-compose |
| **Testing** | pytest + pytest-asyncio |

---

## 📝 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | System health check (API status, LLM backend, database) |
| `POST` | `/simulation/run` | Start a new auction simulation (runs async, returns immediately) |
| `GET` | `/simulations` | List all past simulations (newest first) |
| `GET` | `/simulation/{id}` | Get full simulation detail (agents, rounds, Nash equilibrium) |
| `POST` | `/nash/compute` | Compute Nash equilibrium for arbitrary agent configurations |

---

## 🔮 Future Integration

This project is designed to integrate with [autonomous-procurement-swarm](https://github.com/aragit/autonomous-procurement-swarm):

| Procurement Swarm | Nash Marketing Agents | Integration Point |
|:---|:---|:---|
| Bilateral negotiation | N-player competitive auction | Shared LLM engine |
| Buyer vs. Seller | Brand vs. Brand | Shared PostgreSQL ledger |
| Pareto efficiency | Nash equilibrium | Unified dashboard |
| Cost minimization | Budget preservation | Cross-domain analytics |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes and run tests: `pytest tests/ -v`
4. Commit: `git commit -m "feat: describe your change"`
5. Push: `git push origin feat/your-feature`
6. Open a Pull Request against `main`

Please ensure all 36 tests pass before submitting.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
