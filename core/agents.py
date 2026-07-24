"""Brand agent definitions for the Nash marketing auction."""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from core.telemetry import tracer
from core.llm_engine import BaseLLMEngine, LLMResponse
from core.prompts import BrandPrompt
from core.planner import StrategyPlanner

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Mutable state for a brand agent."""
    name: str
    role: str  # "aggressive", "conservative", "balanced"
    total_budget: float
    remaining_budget: float
    target_cpa: float
    impressions_won: int = 0
    total_spent: float = 0.0
    total_conversions: int = 0
    win_rate: float = 0.0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    bid_history: List[float] = field(default_factory=list)

    @property
    def effective_cpa(self) -> float:
        """Calculate actual CPA."""
        if self.total_conversions == 0:
            return float('inf')
        return self.total_spent / self.total_conversions

    def update_after_auction(self, won: bool, bid: float, conversions: int = 0):
        """Update state after an auction round."""
        if won:
            self.remaining_budget -= bid
            self.total_spent += bid
            self.impressions_won += 1
            self.total_conversions += conversions
        total_auctions = len(self.bid_history) + 1
        self.win_rate = self.impressions_won / total_auctions if total_auctions > 0 else 0.0
        self.bid_history.append(bid)


class BrandAgent:
    """Autonomous brand agent that bids in ad auctions using LLM strategy."""

    def __init__(self, name: str, role: str, budget: float, target_cpa: float, llm: BaseLLMEngine):
        self.name = name
        self.llm = llm
        self.planner = StrategyPlanner(llm_client=llm)
        self.current_strategy = role
        self.state = AgentState(
            name=name,
            role=role,
            total_budget=budget,
            remaining_budget=budget,
            target_cpa=target_cpa,
        )

    async def decide_bid(
        self,
        market_price: float,
        competitor_count: int,
        available_impressions: int,
        recent_history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Use LLM to decide bidding strategy for this round.

        Before querying the LLM, the symbolic planner evaluates recent
        performance and budget state to select a tactical strategy that
        is injected into the prompt.
        """
        with tracer.start_as_current_span("agent_decide_bid") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("agent.role", self.state.role)
            span.set_attribute("agent.budget_pct",
                self.state.remaining_budget / self.state.total_budget
                if self.state.total_budget > 0 else 0.0)

            # --- Planning Layer: tactical strategy override ---
            budget_pct = self.state.remaining_budget / self.state.total_budget if self.state.total_budget > 0 else 0.0
            recent_win_rate = StrategyPlanner.compute_recent_win_rate(
                agent_name=self.name,
                history=recent_history or [],
            )
            new_strategy = await self.planner.evaluate_strategy(
                current_budget_pct=budget_pct,
                recent_win_rate=recent_win_rate,
                recent_history=recent_history,
            )
            if new_strategy != self.current_strategy:
                logger.info(
                    f"[{self.name}] Strategy shift: {self.current_strategy} → {new_strategy} "
                    f"(budget {budget_pct:.0%}, recent win rate {recent_win_rate:.0%})"
                )
                self.current_strategy = new_strategy

            span.set_attribute("agent.strategy", self.current_strategy)

            history_str = self._format_history()

            prompt = BrandPrompt.render(
                brand_name=self.name,
                role=self.current_strategy,
                budget=self.state.total_budget,
                remaining_budget=self.state.remaining_budget,
                target_cpa=self.state.target_cpa,
                market_price=market_price,
                win_rate=self.state.win_rate,
                competitor_count=competitor_count,
                history=history_str,
            )

            messages = [
                {"role": "system", "content": prompt},
            ]

            try:
                response: LLMResponse = await self.llm.async_chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=256,
                )
                strategy = json.loads(response.content)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[{self.name}] LLM parse failed: {e}. Using fallback.")
                strategy = self._fallback_strategy(market_price)

            bid = float(strategy.get("bid", market_price))
            span.set_attribute("agent.bid", round(bid, 2))

            return {
                "bid": round(bid, 2),
                "max_daily_spend": float(strategy.get("max_daily_spend", self.state.remaining_budget * 0.15)),
                "target_cpa": float(strategy.get("target_cpa", self.state.target_cpa)),
                "strategy": strategy.get("strategy", self.state.role),
                "justification": strategy.get("justification", "Fallback strategy"),
                "latency_ms": getattr(response, 'latency_ms', 0),
            }

    def _format_history(self) -> str:
        """Format recent bid history for prompt context."""
        if not self.state.bid_history:
            return "No previous bids."
        recent = self.state.bid_history[-5:]
        return f"Recent bids: {recent}. Win rate: {self.state.win_rate:.1%}."

    def _fallback_strategy(self, market_price: float) -> Dict[str, Any]:
        """Fallback if LLM fails."""
        return {
            "bid": round(market_price * 0.95, 2),
            "max_daily_spend": self.state.remaining_budget * 0.1,
            "target_cpa": self.state.target_cpa,
            "strategy": self.current_strategy,
            "justification": "Fallback: conservative bid at 95% of market price",
        }

    def observe_result(self, won: bool, bid: float, conversions: int = 0):
        """Observe auction result and update internal state."""
        self.state.update_after_auction(won, bid, conversions)
        logger.info(
            f"[{self.name}] Auction result: {'WON' if won else 'LOST'} | "
            f"Bid: ${bid:.2f} | Remaining: ${self.state.remaining_budget:.2f} | "
            f"Win rate: {self.state.win_rate:.1%}"
        )