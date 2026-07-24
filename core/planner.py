"""Neural strategy planner for brand agents.

Uses the LLM as a strategic brain: feeds budget state, win rates, and
recent market history into a planning prompt, and the model reasons about
the best tactical posture before the agent synthesizes a numerical bid.
"""

import logging
from typing import List, Dict, Any

from core.telemetry import tracer
from core.llm_engine import BaseLLMEngine

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"aggressive", "balanced", "conserve"}

PLANNER_PROMPT = """You are the strategic planner for an AI bidding agent in a real-time ad auction.

Current Budget Remaining: {budget_pct:.0%}
Recent Win Rate: {win_rate:.0%}
Number of recent rounds analyzed: {round_count}
Recent clearing prices: {clearing_prices}

Based on this data, choose the best tactical posture for the next auction round.
- "aggressive": bid high to capture impressions when underperforming or market is favorable.
- "balanced": maintain steady competitive bids with moderate risk.
- "conserve": protect remaining budget when running low or market is overpriced.

Respond with EXACTLY ONE word: aggressive, balanced, or conserve.
"""


class StrategyPlanner:
    """Neural planner that uses the LLM to select a tactical strategy.

    Runs before every bid synthesis.  The returned strategy string is
    injected into the bid-generation prompt so the neural synthesizer
    produces a bid aligned with the current strategic plan.
    """

    def __init__(self, llm_client: BaseLLMEngine):
        self.llm = llm_client

    async def evaluate_strategy(
        self,
        current_budget_pct: float,
        recent_win_rate: float,
        recent_history: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Query the LLM to select a tactical strategy.

        Args:
            current_budget_pct: remaining / total budget (0.0 – 1.0).
            recent_win_rate:    win rate over recent rounds.
            recent_history:     list of AuctionResult dicts from the engine.

        Returns:
            One of "aggressive", "balanced", or "conserve".
        """
        with tracer.start_as_current_span("planner_evaluate") as span:
            span.set_attribute("planner.budget_pct", current_budget_pct)
            span.set_attribute("planner.win_rate", recent_win_rate)

            history = recent_history or []
            clearing_prices = [
                r.get("clearing_price", 0.0) for r in history[-5:]
            ]
            clearing_prices_str = (
                ", ".join(f"${p:.2f}" for p in clearing_prices)
                if clearing_prices else "N/A (first round)"
            )

            prompt = PLANNER_PROMPT.format(
                budget_pct=current_budget_pct,
                win_rate=recent_win_rate,
                round_count=len(history),
                clearing_prices=clearing_prices_str,
            )

            messages = [{"role": "system", "content": prompt}]

            try:
                response = await self.llm.async_chat_completion(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=10,
                )
                strategy = response.content.strip().lower()
            except Exception as e:
                logger.warning(f"Planner LLM call failed: {e}. Defaulting to balanced.")
                span.set_attribute("planner.strategy", "balanced")
                span.set_attribute("planner.fallback", True)
                return "balanced"

            if strategy in VALID_STRATEGIES:
                span.set_attribute("planner.strategy", strategy)
                return strategy

            logger.warning(
                f"Planner returned invalid strategy '{strategy}'. Defaulting to balanced."
            )
            span.set_attribute("planner.strategy", "balanced")
            span.set_attribute("planner.fallback", True)
            return "balanced"

    @staticmethod
    def compute_recent_win_rate(
        agent_name: str,
        history: List[Dict[str, Any]],
        window: int = 5,
    ) -> float:
        """Derive a windowed win rate from engine history for a single agent.

        Args:
            agent_name: the agent to evaluate.
            history:    list of AuctionResult dicts (must contain "winners"
                        and "losers" keys, each a list of dicts with
                        "agent_name").
            window:     how many recent rounds to consider.

        Returns:
            Win rate (0.0 – 1.0) over the last *window* rounds the agent
            participated in.  Returns 0.0 if no history is available.
        """
        if not history:
            return 0.0

        recent = history[-window:]
        wins = 0
        total = 0
        for r in recent:
            winner_names = [w["agent_name"] for w in r.get("winners", [])]
            loser_names = [l["agent_name"] for l in r.get("losers", [])]
            if agent_name in winner_names:
                wins += 1
                total += 1
            elif agent_name in loser_names:
                total += 1

        return wins / total if total > 0 else 0.0
