<h1 align="center">Agentic Nash Marketing</h1>
<p align="center"><b>Neuro-Symbolic Multi-Agent Competitive Ad Auction with Nash Equilibrium</b></p>

<p align="center"><sub>FastAPI · Server-Sent Events (SSE) · TailwindCSS · Ollama · vLLM · NumPy · OpenTelemetry</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Production Ready">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streaming-SSE%20Enabled-orange" alt="SSE">
  <img src="https://img.shields.io/badge/LLM-Mock%20%7C%20Ollama%20%7C%20vLLM-yellow" alt="LLM Backends">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Autonomous AI brand agents compete in real-time ad auctions using a **neuro-symbolic architecture**: a neural LLM proposes strategies, a symbolic planner reasons about market trends, a guardrail enforces budget rules, and a vectorized Nash equilibrium solver computes optimal mixed strategies. 

**New:** The engine now features per-agent **Server-Sent Events (SSE) streaming**, allowing you to watch the LLMs think, shift strategies, and bid in real-time via a sleek Tailwind dashboard.

---

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Architecture Overview](#architecture-overview)
  - [Real-Time SSE Streaming](#real-time-sse-streaming)
  - [Neuro-Symbolic Pipeline](#neuro-symbolic-pipeline)
  - [Async Parallel Execution](#async-parallel-execution)
- [Quick Start](#quick-start)
- [LLM Backend Configuration](#llm-backend-configuration)
- [Live Dashboard](#live-dashboard)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Why This Matters

| Problem | Impact | Our Solution |
|:--------|:-------|:-------------|
| Advertisers waste 30%+ of spend on suboptimal bidding | Proven optimal strategies exist via Nash equilibrium | Post-hoc Nash solver with vectorized Monte Carlo |
| Auction platforms lose revenue from unstable bidding wars | Equilibrium stabilizes clearing prices | Game-theoretic mixed-strategy convergence |
| Multi-agent systems are slow when agents think sequentially | 5 agents x 100s = 500s per round | `asyncio.wait` parallel inference streaming |
| Standard APIs timeout during heavy LLM inference | Silent failures and broken frontends | Per-agent SSE yielding keeping channels alive |

---

## Architecture Overview

### Real-Time SSE Streaming

The simulation engine is decoupled from standard request/response blocking. Using `asyncio.wait(FIRST_COMPLETED)`, the `SimulationRunner` yields events exactly when an individual agent's LLM finishes inference, ensuring the network socket stays alive and the UI updates organically, even during heavy CPU-bound Ollama workloads.

```
Browser (EventSource)
    ↓ GET /api/v1/simulate/stream
FastAPI (SSE endpoint)
    ↓ creates OllamaEngine
SimulationRunner.stream_run()
    ↓ per-round, per-agent streaming
    ↓ asyncio.wait(FIRST_COMPLETED)
AuctionEngine (guardrail + VCG resolution)
    ↓ yields events
SSE Response → Browser UI live updates
```

### Neuro-Symbolic Pipeline

The system implements a three-layer neuro-symbolic architecture:
1. **Neural Planning:** Queries the LLM with budget % and win rate to select a tactical strategy (`aggressive`, `balanced`, `conserve`).
2. **Neural Synthesis:** Generates a structured JSON bid with justification based on the selected strategy.
3. **Symbolic Enforcement:** The `BudgetGuardrail` intercepts every raw bid and applies caps. The VCG second-price mechanism then resolves the auction, mathematically guaranteeing no agent overpays.

---

## Quick Start

### 1. Start the API Server
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.server:app --reload
```

### 2. Open the Live Dashboard
No web server required. Open `dashboard.html` directly in your browser.

### 3. Run with Real LLMs (Ollama)
Ensure Ollama is running locally with a model pulled (e.g., `qwen2.5:1.5b`). Select the Ollama backend directly from the UI dropdown to watch real inference in action.

---

## LLM Backend Configuration

The system supports multiple LLM backends via a unified factory pattern:

| Backend | Provider String | Use Case |
|:--------|:----------------|:---------|
| MockLLM | `mock` | CI/CD, UI testing, instant simulation |
| Ollama | `ollama` | Local CPU/GPU inference (`ollama` package required) |
| vLLM | `vllm` | High-throughput serving (`openai` package required) |

---

## Live Dashboard

The repository includes a single-file, zero-dependency (vanilla JS + Tailwind CDN) real-time dashboard.

- Monitor live market stats (clearing prices, total revenue).
- Watch dynamic strategy shifts (e.g., an agent shifts from balanced to conserve as budgets deplete).
- Track VCG auction mechanics validating that second-price rules are enforced.

---

## Future Work

We are actively expanding the agentic capabilities of this engine. Contributions are welcome for the following roadmap items:

- **[Option 2] GPU Throughput Benchmarking:** Wrapping the environment in a reverse-proxy tunnel to run live Tokens-Per-Second (TPS) load tests on Kaggle T4x2 / cloud instances using vLLM.
- **[Option 3] Advanced Market Dynamics:** Injecting chaotic market shocks mid-simulation (e.g., a sudden 50% budget slash in Round 3, or dynamic floor prices).
- **[Option 3] Agent Collusion Detection:** Implementing symbolic rules to detect if LLM agents are intentionally colluding to artificially lower the VCG clearing price over extended time horizons.

---

## Refactoring History

| Step | What Changed |
|:-----|:-------------|
| 1-7 | Monolithic separation, Multi-turn reasoning, Nash Solver Vectorization, Async execution, Factory patterns. |
| 8. Observability | Distributed OpenTelemetry tracing across the entire inference pipeline. |
| 9. Live Streaming | Added FastAPI SSE endpoint, real-time Tailwind dashboard, and Ollama integration with semantic strategy aliases. |

---

## License

MIT — see LICENSE for details.
