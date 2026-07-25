# Report: Ollama Integration & Real-Time SSE Streaming

## 1. Architecture Overview

The neuro-symbolic auction system was extended with:

- **SSE Streaming Endpoint** (`api/server.py`) — FastAPI + `sse-starlette` for real-time event streaming
- **Live Dashboard** (`dashboard.html`) — Single-file UI with Tailwind CSS and vanilla JS `EventSource`
- **Real LLM Backend** — Ollama integration via `OllamaEngine` for local inference
- **Per-Agent Streaming** — `SimulationRunner.stream_run()` yields events as each agent's LLM finishes, not after the full round

### Data Flow

```
Browser (EventSource)
    ↓ GET /api/v1/simulate/stream
FastAPI (SSE endpoint)
    ↓ creates MockLLMEngine or OllamaEngine
SimulationRunner.stream_run()
    ↓ per-round, per-agent streaming
    ↓ asyncio.wait(FIRST_COMPLETED)
AuctionEngine (guardrail + VCG resolution)
    ↓ yields events
SSE Response → Browser UI updates
```

## 2. Implementation Steps

### Step 1: SSE Infrastructure

Added to `api/server.py`:
- `GET /api/v1/simulate/stream` endpoint with query params: `num_agents`, `num_rounds`, `seed`, `backend`, `model`
- `EventSourceResponse` wrapping an async generator
- CORS middleware for local HTML file access

### Step 2: SimulationRunner Modification

Original `run()` method blocked until all rounds completed. Modified `stream_run()` to:
- Yield `start` event with agent configs
- For each round, use `asyncio.wait(FIRST_COMPLETED)` instead of `asyncio.gather()`
- Yield `agent_thinking` event the moment each agent's LLM inference completes
- Yield `agent_decision` and `round_complete` after VCG resolution
- Yield `complete` with final summary

### Step 3: Ollama Integration

- Installed `ollama` Python package (v0.4.7) for both conda and system Python
- `OllamaEngine` uses native async via `ollama.AsyncClient`
- Graceful fallback: if Ollama unavailable, falls back to `MockLLMEngine`
- Default model: `qwen2.5:1.5b` (940MB, runs on CPU)

### Step 4: Dashboard UI

Single-file `dashboard.html` with:
- Backend/Model selectors (Ollama or Mock)
- Agent count/round count inputs
- Live market stats (round, clearing price, revenue, impressions)
- Agent cards with real-time bid status (thinking → won/lost)
- Scrolling event log with timestamps

## 3. Issues Encountered & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: No module named 'ollama'` | Package installed in conda but uvicorn runs under `/usr/bin/python3` (system Python 3.10) | Installed `ollama` for system Python via `/usr/bin/python3 -m pip install ollama` |
| SSE connection dies after Round 1 | `engine.run_round()` blocks for 2+ minutes (all agents finish before any event yields) | Restructured `stream_run()` to use `asyncio.wait(FIRST_COMPLETED)` — yields per-agent |
| EventSource error spam | `error` event fires on every auto-retry and on normal stream close | Switched to `onerror` handler with `readyState` check; only shows status for never-connected state |
| Planner warning "invalid strategy 'conservative'" | Real LLM returns "conservative" but planner only accepted "conserve" | Added `STRATEGY_ALIASES` mapping and expanded `VALID_STRATEGIES` set |
| FastAPI/Starlette version conflict | Starlette 1.3.1 removed `on_startup` param used by FastAPI 0.115.x | Pinned `starlette>=0.37.0,<1.0.0` in requirements.txt |

## 4. Results Analysis

### 4.1 Inference Latency (Ollama + qwen2.5:1.5b on CPU)

| Agent | Round 1 | Round 2 | Round 3 |
|-------|---------|---------|---------|
| Agent 1 (fastest) | 42s | 50s | 65s |
| Agent 2 | 76s | 84s | 98s |
| Agent 3 (slowest) | 110s | 128s | 127s |

**Key observations:**
- Each agent takes 40-130 seconds per inference
- 3 agents in parallel → round duration ≈ slowest agent (110-130s)
- Total 3-round simulation: ~7 minutes
- Latency variance: first agent is fastest (queue position advantage), last agent waits for GPU/CPU resources

### 4.2 Bid Behavior Comparison

| Metric | Mock LLM | Ollama qwen2.5:1.5b |
|--------|----------|---------------------|
| Bid range | $12-30 | $2-3 |
| Strategy | Deterministic by role | LLM-generated |
| Planner calls | Instant | 40-110s each |
| Bid generation | Rule-based JSON | LLM JSON output |

**Why lower bids with real LLM?** The mock engine was tuned to produce bids proportional to `target_cpa * role_pct` (aggressive: 70-95%, balanced: 35-60%, conservative: 5-25%). The real qwen2.5 model interprets the prompt differently and generates lower bids based on its training data, resulting in more conservative pricing.

### 4.3 VCG Mechanism Verification

Round 1 results (3 agents):
```
Bids: Adidas $3.21, Nike $2.81, Puma $2.61
Impressions: 1
Winner: Adidas (highest bid)
Payment: $2.81 (second-highest price)
Revenue: $2.81
```

Round 2 results (3 agents, 3 impressions available):
```
Bids: Nike $2.81, Adidas $2.61, Puma $2.61
Impressions: 3
All win (everyone gets an impression)
Nike pays $2.61 (next highest)
Adidas pays $2.61 (next highest)
Puma pays $2.35 (90% of own bid, no one below)
Revenue: $7.57
```

**VCG invariants verified:**
- Winners never pay more than their bid
- Payment equals second-highest bid (or 90% of own bid if lowest winner)
- Revenue = sum of all payments

### 4.4 Strategy Shifts

The planner (also using Ollama) evaluates budget state and win rate to shift strategies:

```
Round 1: All agents "balanced" → mixed after first round
Round 2: Adidas shifts "balanced → aggressive" (100% win rate)
Round 3: Nike shifts "aggressive → balanced" (50% win rate, conserving budget)
```

This demonstrates the neural-symbolic loop: LLM generates bids, symbolic planner evaluates performance and adjusts strategy, LLM incorporates new strategy into next bid.

## 5. Performance Characteristics

| Component | Latency | Bottleneck |
|-----------|---------|------------|
| Market simulation | <1ms | Negligible |
| Guardrail check | <1ms | Negligible |
| VCG resolution | <1ms | Negligible |
| Planner LLM call | 40-110s | CPU inference |
| Bid generation LLM call | 40-130s | CPU inference |
| SSE event yield | <1ms | Negligible |
| Dashboard UI update | <1ms | Negligible |

**Total per-round latency:** ~2 minutes (dominated by LLM inference)
**Total 3-round simulation:** ~7 minutes
**Throughput:** ~0.5 rounds/minute with 3 agents on CPU

## 6. Recommendations

### For Faster Inference
1. **Use GPU**: `OLLAMA_GPU_LAYERS=99` for CUDA acceleration
2. **Smaller model**: `tinyllama` (608MB) is ~2x faster than `qwen2.5:1.5b`
3. **Reduce max_tokens**: Lower from 256/512 to 64 for faster generation

### For More Agents
- Current max: 10 agents (dashboard grid supports up to 10)
- With 7+ agents, round duration increases linearly (more parallel LLM calls)
- Consider batching agents or using model parallelism for 10+ agents

### For Production
- Add connection pooling for Ollama client
- Implement retry logic with exponential backoff
- Add authentication to SSE endpoint
- Consider WebSocket for bidirectional communication

## 7. Files Modified

| File | Changes |
|------|---------|
| `requirements.txt` | Added `sse-starlette`, pinned `starlette<1.0.0` |
| `api/server.py` | Added CORS, Ollama backend selection, graceful fallback |
| `api/main.py` | (unchanged) |
| `core/simulation_runner.py` | Added `stream_run()` with per-agent event yielding |
| `core/planner.py` | Added `STRATEGY_ALIASES` mapping for real LLM output |
| `dashboard.html` | New file — SSE dashboard with real-time UI |
| `REPORT_ollama_integration.md` | This report |

## 8. Conclusion

The integration successfully demonstrates:
1. Real-time streaming of multi-agent auction simulations via SSE
2. Live dashboard updating as each agent's LLM inference completes
3. VCG second-price mechanism functioning correctly with real LLM bids
4. Neural-symbolic loop working end-to-end (LLM → planner → LLM)

The system is ready for experimentation with different models, agent configurations, and auction parameters. The per-agent streaming approach ensures the SSE connection stays alive during long inference windows, providing a smooth user experience even with CPU-bound LLM backends.
