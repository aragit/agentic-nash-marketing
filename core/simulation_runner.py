"""Decoupled simulation runner for the neuro-symbolic auction loop.

Executes the core auction rounds without any dependency on FastAPI,
SQLAlchemy, or background task infrastructure. Accepts pre-built
engine and agent instances via dependency injection.
"""

import logging
from typing import List, Dict, Any

from core.auction import AuctionEngine, AuctionResult
from core.agents import BrandAgent

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
        """Execute the full simulation loop and return the final state.

        Returns:
            Dictionary containing:
              - history: list of AuctionResult per completed round
              - total_rounds: how many rounds actually executed
              - total_revenue: cumulative revenue across all rounds
              - final_clearing_price: clearing price of the last round
              - agents: list of per-agent state snapshots
        """
        for round_idx in range(self.num_rounds):
            await self.engine.run_round(self.agents)

            # Early termination when fewer than 2 agents can still bid
            active = [a for a in self.agents if a.state.remaining_budget > 0]
            if len(active) < 2:
                logger.info(
                    f"Simulation ended early at round {round_idx + 1}: "
                    f"only {len(active)} agents active"
                )
                break

        return self._build_final_state()

    def _build_final_state(self) -> Dict[str, Any]:
        """Assemble a serialisable snapshot of the simulation outcome."""
        history = self.engine.history
        return {
            "history": history,
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
