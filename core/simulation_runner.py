"""Decoupled simulation runner for the neuro-symbolic auction loop.

Executes the core auction rounds without any dependency on FastAPI,
SQLAlchemy, or background task infrastructure. Accepts pre-built
engine and agent instances via dependency injection.
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, AsyncGenerator, Tuple

from core.auction import AuctionEngine, AuctionResult
from core.agents import BrandAgent
from core.market import MarketState
from core.telemetry import tracer

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Runs a multi-round auction simulation in isolation.

    This class is the single source of truth for the execution loop.
    It holds no references to database sessions, background tasks, or
    HTTP-layer concerns — all side-effects are the caller's responsibility.
    """

    def __init__(self, engine: AuctionEngine, agents: List[BrandAgent], num_rounds: int):
        self.engine = engine
        self.agents = agents
        self.num_rounds = num_rounds

    async def run(self) -> Dict[str, Any]:
        """Execute the full simulation loop and return the final state."""
        for round_idx in range(self.num_rounds):
            await self.engine.run_round(self.agents)
            active = [a for a in self.agents if a.state.remaining_budget > 0]
            if len(active) < 2:
                logger.info(
                    f"Simulation ended early at round {round_idx + 1}: "
                    f"only {len(active)} agents active"
                )
                break
        return self._build_final_state()

    async def stream_run(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the simulation, yielding SSE events as each agent finishes.

        Unlike ``run()``, this drives the auction round step-by-step so that
        each agent's LLM inference produces an event immediately upon completion,
        keeping the SSE connection alive during long inference windows.
        """
        agent_states = [
            {
                "name": a.name,
                "role": a.state.role,
                "total_budget": a.state.total_budget,
                "target_cpa": a.state.target_cpa,
            }
            for a in self.agents
        ]
        yield {
            "event": "start",
            "data": json.dumps({
                "message": "Simulation started",
                "num_rounds": self.num_rounds,
                "agents": agent_states,
            }),
        }

        for round_idx in range(self.num_rounds):
            async for event in self._stream_round(round_idx + 1):
                yield event

            active = [a for a in self.agents if a.state.remaining_budget > 0]
            if len(active) < 2:
                logger.info(
                    f"Simulation ended early at round {round_idx + 1}: "
                    f"only {len(active)} agents active"
                )
                break

        final_state = self._build_final_state()
        yield {
            "event": "complete",
            "data": json.dumps(final_state, default=str),
        }

    async def _stream_round(self, round_number: int) -> AsyncGenerator[Dict[str, Any], None]:
        """Drive one auction round, yielding events as agents complete inference.

        Replicates the AuctionEngine round logic but yields an ``agent_decision``
        event the moment each agent's LLM returns, rather than waiting for all
        agents to finish before emitting anything.
        """
        with tracer.start_as_current_span("auction_round") as span:
            span.set_attribute("auction.agent_count", len(self.agents))
            span.set_attribute("auction.round_number", round_number)

            market = self.engine.market.step(len(self.agents))
            span.set_attribute("auction.impressions", market.available_impressions)
            span.set_attribute("auction.base_cpm", market.base_cpm)

            active_agents = [a for a in self.agents if a.state.remaining_budget > 0]
            skipped = [a for a in self.agents if a.state.remaining_budget <= 0]

            for agent in skipped:
                yield {
                    "event": "agent_decision",
                    "data": json.dumps({
                        "round": round_number,
                        "agent": agent.name,
                        "status": "skipped",
                        "reason": "budget_depleted",
                    }),
                }

            # --- Phase 1: Gather bids, yielding as each agent finishes ---
            recent_hist = [r.__dict__ for r in self.engine.history]
            bid_tasks = [
                asyncio.ensure_future(
                    agent.decide_bid(
                        market_price=market.base_cpm,
                        competitor_count=len(self.agents) - 1,
                        available_impressions=market.available_impressions,
                        recent_history=recent_hist,
                    )
                )
                for agent in active_agents
            ]

            # Collect results as they arrive (not all-at-once)
            raw_results: List[Tuple[BrandAgent, Dict[str, Any]]] = []
            pending = list(zip(active_agents, bid_tasks))
            while pending:
                done, _ = await asyncio.wait(
                    [task for _, task in pending],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                finished = [(a, t) for a, t in pending if t in done]
                pending = [(a, t) for a, t in pending if t not in done]

                for agent, task in finished:
                    result = task.result()
                    raw_results.append((agent, result))
                    # Yield immediately — this keeps the SSE pipe alive
                    yield {
                        "event": "agent_thinking",
                        "data": json.dumps({
                            "round": round_number,
                            "agent": agent.name,
                            "bid": result["bid"],
                            "strategy": result.get("strategy", agent.state.role),
                            "latency_ms": result.get("latency_ms", 0),
                        }),
                    }

            span.set_attribute("auction.active_agents", len(active_agents))

            # --- Phase 2: Guardrail + VCG resolution ---
            bids: List[Tuple[BrandAgent, float]] = []
            for agent, result in raw_results:
                action = self.engine.guardrail.check(
                    agent_name=agent.name,
                    bid=result["bid"],
                    remaining=agent.state.remaining_budget,
                    total=agent.state.total_budget,
                )
                bids.append((agent, action.adjusted_bid))

            if not bids:
                empty = AuctionResult(
                    round_number=round_number,
                    clearing_price=0.0, winners=[], losers=[],
                    total_revenue=0.0, market_state=market,
                )
                self.engine.history.append(empty)
                yield {
                    "event": "round_complete",
                    "data": json.dumps({
                        "round": round_number, "clearing_price": 0.0,
                        "total_revenue": 0.0, "winner_count": 0,
                        "loser_count": 0, "available_impressions": market.available_impressions,
                        "audience_quality": market.audience_quality,
                        "seasonality": market.seasonality_factor,
                    }),
                }
                return

            bids.sort(key=lambda x: x[1], reverse=True)

            winners_data: List[Dict[str, Any]] = []
            losers_data: List[Dict[str, Any]] = []

            for i, (agent, bid) in enumerate(bids):
                if i < market.available_impressions:
                    pay_price = bids[i + 1][1] if i + 1 < len(bids) else bid * 0.9
                    pay_price = round(pay_price, 2)
                    conversions = self.engine.market.estimate_conversions(
                        impressions=1, audience_quality=market.audience_quality, bid=bid,
                    )
                    agent.observe_result(won=True, bid=pay_price, conversions=conversions)
                    winners_data.append({
                        "agent_name": agent.name, "bid": bid, "paid": pay_price,
                        "conversions": conversions,
                        "remaining_budget": agent.state.remaining_budget,
                        "strategy": agent.state.role,
                    })
                else:
                    agent.observe_result(won=False, bid=bid)
                    losers_data.append({
                        "agent_name": agent.name, "bid": bid,
                        "remaining_budget": agent.state.remaining_budget,
                        "strategy": agent.state.role,
                    })

            total_revenue = sum(w["paid"] for w in winners_data)
            result = AuctionResult(
                round_number=round_number,
                clearing_price=winners_data[-1]["paid"] if winners_data else 0.0,
                winners=winners_data, losers=losers_data,
                total_revenue=round(total_revenue, 2), market_state=market,
            )
            self.engine.history.append(result)

            # Yield final per-agent outcomes
            for entry in winners_data:
                yield {
                    "event": "agent_decision",
                    "data": json.dumps({
                        "round": round_number, "agent": entry["agent_name"],
                        "status": "won", "bid": entry["bid"], "paid": entry["paid"],
                        "conversions": entry["conversions"],
                        "remaining_budget": entry["remaining_budget"],
                        "strategy": entry["strategy"],
                    }),
                }
            for entry in losers_data:
                yield {
                    "event": "agent_decision",
                    "data": json.dumps({
                        "round": round_number, "agent": entry["agent_name"],
                        "status": "lost", "bid": entry["bid"],
                        "remaining_budget": entry["remaining_budget"],
                        "strategy": entry["strategy"],
                    }),
                }

            yield {
                "event": "round_complete",
                "data": json.dumps({
                    "round": round_number,
                    "clearing_price": result.clearing_price,
                    "total_revenue": result.total_revenue,
                    "winner_count": len(winners_data),
                    "loser_count": len(losers_data),
                    "available_impressions": market.available_impressions,
                    "audience_quality": market.audience_quality,
                    "seasonality": market.seasonality_factor,
                }),
            }

            logger.info(
                f"[Auction Round {round_number}] "
                f"Clearing: ${result.clearing_price:.2f} | "
                f"Winners: {len(winners_data)} | "
                f"Revenue: ${total_revenue:.2f}"
            )

    def _build_final_state(self) -> Dict[str, Any]:
        """Assemble a serialisable snapshot of the simulation outcome."""
        history = self.engine.history
        return {
            "history": [
                {
                    "round_number": r.round_number,
                    "clearing_price": r.clearing_price,
                    "total_revenue": r.total_revenue,
                    "winners": r.winners,
                    "losers": r.losers,
                    "available_impressions": r.market_state.available_impressions,
                    "audience_quality": r.market_state.audience_quality,
                    "seasonality": r.market_state.seasonality_factor,
                }
                for r in history
            ],
            "total_rounds": len(history),
            "total_revenue": sum(r.total_revenue for r in history),
            "final_clearing_price": history[-1].clearing_price if history else 0.0,
            "agents": [
                {
                    "name": a.name,
                    "role": a.state.role,
                    "remaining_budget": a.state.remaining_budget,
                    "impressions_won": a.state.impressions_won,
                    "total_spent": a.state.total_spent,
                    "total_conversions": a.state.total_conversions,
                    "win_rate": a.state.win_rate,
                    "target_cpa": a.state.target_cpa,
                    "total_budget": a.state.total_budget,
                }
                for a in self.agents
            ],
        }
