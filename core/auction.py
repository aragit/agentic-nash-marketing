"""Ad auction engine with second-price VCG mechanism."""

import asyncio
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from core.telemetry import tracer
from core.agents import BrandAgent
from core.market import MarketSimulator, MarketState
from core.guardrails import BudgetGuardrail

logger = logging.getLogger(__name__)


@dataclass
class AuctionResult:
    """Result of a single auction round."""
    round_number: int
    clearing_price: float
    winners: List[Dict[str, Any]]
    losers: List[Dict[str, Any]]
    total_revenue: float
    market_state: MarketState


class AuctionEngine:
    """Second-price VCG auction engine."""

    def __init__(self, market: MarketSimulator):
        self.market = market
        self.history: List[AuctionResult] = []
        self.guardrail = BudgetGuardrail()

    async def run_round(self, agents: List[BrandAgent]) -> AuctionResult:
        """Execute one auction round.

        Phase 1 (parallel): All agents compute strategy + bid concurrently.
        Phase 2 (sequential): Guardrail enforcement + VCG resolution.
        """
        with tracer.start_as_current_span("auction_round") as span:
            span.set_attribute("auction.agent_count", len(agents))
            span.set_attribute("auction.round_number", self.market.round + 1)

            market = self.market.step(len(agents))
            span.set_attribute("auction.impressions", market.available_impressions)
            span.set_attribute("auction.base_cpm", market.base_cpm)

            # --- Phase 1: Parallel neural execution ---
            active_agents = [a for a in agents if a.state.remaining_budget > 0]
            skipped = [a for a in agents if a.state.remaining_budget <= 0]
            for agent in skipped:
                logger.info(f"[{agent.name}] SKIPPED — budget depleted")

            span.set_attribute("auction.active_agents", len(active_agents))

            recent_hist = [r.__dict__ for r in self.history]
            bid_tasks = [
                agent.decide_bid(
                    market_price=market.base_cpm,
                    competitor_count=len(agents) - 1,
                    available_impressions=market.available_impressions,
                    recent_history=recent_hist,
                )
                for agent in active_agents
            ]
            raw_results = await asyncio.gather(*bid_tasks)

            # --- Phase 2: Symbolic guardrail + VCG resolution ---
            bids: List[Tuple[BrandAgent, float]] = []
            guardrail_interventions = 0
            for agent, result in zip(active_agents, raw_results):
                action = self.guardrail.check(
                    agent_name=agent.name,
                    bid=result["bid"],
                    remaining=agent.state.remaining_budget,
                    total=agent.state.total_budget,
                )
                if action.action != "allow":
                    guardrail_interventions += 1
                    logger.info(
                        f"[Guardrail] {agent.name}: ${result['bid']:.2f} → "
                        f"${action.adjusted_bid:.2f} ({action.action}: {action.reason})"
                    )
                bids.append((agent, action.adjusted_bid))

            span.set_attribute("auction.guardrail_interventions", guardrail_interventions)

            if not bids:
                return AuctionResult(
                    round_number=market.round_number,
                    clearing_price=0.0,
                    winners=[],
                    losers=[],
                    total_revenue=0.0,
                    market_state=market,
                )

            # Sort by bid (highest first)
            bids.sort(key=lambda x: x[1], reverse=True)

            # Determine winners (second-price mechanism)
            winners_data: List[Dict[str, Any]] = []
            losers_data: List[Dict[str, Any]] = []

            for i, (agent, bid) in enumerate(bids):
                if i < market.available_impressions:
                    if i + 1 < len(bids):
                        pay_price = bids[i + 1][1]
                    else:
                        pay_price = bid * 0.9

                    pay_price = round(pay_price, 2)

                    conversions = self.market.estimate_conversions(
                        impressions=1,
                        audience_quality=market.audience_quality,
                        bid=bid,
                    )

                    agent.observe_result(won=True, bid=pay_price, conversions=conversions)

                    winners_data.append({
                        "agent_name": agent.name,
                        "bid": bid,
                        "paid": pay_price,
                        "conversions": conversions,
                        "remaining_budget": agent.state.remaining_budget,
                        "strategy": agent.state.role,
                    })
                else:
                    agent.observe_result(won=False, bid=bid)
                    losers_data.append({
                        "agent_name": agent.name,
                        "bid": bid,
                        "remaining_budget": agent.state.remaining_budget,
                        "strategy": agent.state.role,
                    })

            total_revenue = sum(w["paid"] for w in winners_data)

            result = AuctionResult(
                round_number=market.round_number,
                clearing_price=winners_data[-1]["paid"] if winners_data else 0.0,
                winners=winners_data,
                losers=losers_data,
                total_revenue=round(total_revenue, 2),
                market_state=market,
            )

            self.history.append(result)

            span.set_attribute("auction.winner_count", len(winners_data))
            span.set_attribute("auction.total_revenue", total_revenue)
            span.set_attribute("auction.clearing_price", result.clearing_price)

            # Verify VCG math
            for w in winners_data:
                assert w["paid"] <= w["bid"] + 0.01, (
                    f"VCG VIOLATION: {w['agent_name']} paid ${w['paid']:.2f} > bid ${w['bid']:.2f}"
                )

            logger.info(
                f"[Auction Round {market.round_number}] "
                f"Clearing: ${result.clearing_price:.2f} | "
                f"Winners: {len(winners_data)} | "
                f"Revenue: ${total_revenue:.2f}"
            )

            return result