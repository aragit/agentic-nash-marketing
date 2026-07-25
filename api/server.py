"""FastAPI SSE server for streaming auction simulations.

Run with:
    uvicorn api.server:app --reload --reload-exclude="*.log" --reload-exclude="*.db" --reload-exclude="*.json" --reload-exclude=".git/*"
"""

import json
import asyncio
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from core.llm_engine import MockLLMEngine, OllamaEngine, create_llm_engine
from core.market import MarketSimulator
from core.auction import AuctionEngine
from core.agents import BrandAgent
from core.simulation_runner import SimulationRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Auction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default agent configurations (10 brands, varied roles & budgets)
DEFAULT_AGENTS = [
    {"name": "Nike",       "role": "aggressive",    "budget": 8000.0, "target_cpa": 35.0},
    {"name": "Adidas",     "role": "balanced",      "budget": 6500.0, "target_cpa": 40.0},
    {"name": "Puma",       "role": "conservative",  "budget": 3000.0, "target_cpa": 30.0},
    {"name": "Lotto",      "role": "aggressive",    "budget": 5500.0, "target_cpa": 38.0},
    {"name": "Diadora",    "role": "balanced",      "budget": 4000.0, "target_cpa": 32.0},
    {"name": "Fila",       "role": "conservative",  "budget": 2500.0, "target_cpa": 28.0},
    {"name": "Mizuno",     "role": "balanced",      "budget": 3500.0, "target_cpa": 25.0},
    {"name": "Saucony",    "role": "conservative",  "budget": 2000.0, "target_cpa": 22.0},
    {"name": "Salomon",    "role": "aggressive",    "budget": 4500.0, "target_cpa": 45.0},
    {"name": "Merrell",    "role": "balanced",      "budget": 5000.0, "target_cpa": 42.0},
]


@app.get("/api/v1/simulate/stream")
async def stream_simulation(
    num_agents: int = Query(default=3, ge=2, le=10, description="Number of brand agents"),
    num_rounds: int = Query(default=1, ge=1, le=100, description="Number of auction rounds"),
    seed: int = Query(default=42, description="Random seed for reproducibility"),
    backend: str = Query(default="ollama", description="LLM backend: 'ollama' or 'mock'"),
    model: str = Query(default="qwen2.5:1.5b", description="Model name for Ollama"),
):
    """Stream a multi-agent auction simulation via Server-Sent Events.

    Events:
      - ``start``           — simulation began, includes agent configs
      - ``agent_decision``  — one agent's bid was resolved (won/lost/skipped)
      - ``round_complete``  — full auction round cleared
      - ``complete``        — simulation finished, includes final summary
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        engine = AuctionEngine(market=MarketSimulator(seed=seed))

        if backend == "mock":
            llm = MockLLMEngine(seed=seed)
        else:
            try:
                llm = create_llm_engine(provider="ollama", model=model)
                logger.info(f"Using Ollama model: {model}")
            except Exception as e:
                logger.warning(f"Ollama unavailable ({e}), falling back to MockLLM")
                llm = MockLLMEngine(seed=seed)

        # Build agent list — use defaults or trim to requested count
        agent_configs = DEFAULT_AGENTS[:num_agents]
        agents = [
            BrandAgent(
                name=cfg["name"],
                role=cfg["role"],
                budget=cfg["budget"],
                target_cpa=cfg["target_cpa"],
                llm=llm,
            )
            for cfg in agent_configs
        ]

        runner = SimulationRunner(engine=engine, agents=agents, num_rounds=num_rounds)

        try:
            async for event in runner.stream_run():
                yield event
                # Yield control so the event loop can send data to clients
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Simulation stream failed: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }
        finally:
            llm.shutdown()

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    return {"status": "healthy"}
