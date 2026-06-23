"""Ad auction engine with second-price VCG mechanism."""

import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from core.agents import BrandAgent
from core.market import MarketSimulator, MarketState

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

    def run_round(self, agents: List[BrandAgent]) -> AuctionResult:
        """Execute one auction round."""
        market = self.market.step(len(agents))

        # Collect bids from all agents
        bids: List[Tuple[BrandAgent, float]] = []
        for agent in agents:
            if agent.state.remaining_budget <= 0:
                logger.info(f"[{agent.name}] SKIPPED — budget depleted")
                continue

            result = agent.decide_bid(
                market_price=market.base_cpm,
                competitor_count=len(agents) - 1,
                available_impressions=market.available_impressions,
            )
            bids.append((agent, result["bid"]))

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

        # Trace: log all bids before clearing
        bid_trace = ", ".join(f"{a.name}=${b:.2f}" for a, b in bids)
        logger.debug(
            f"[Round {market.round_number}] Bids: {bid_trace} | "
            f"Impressions: {market.available_impressions} | "
            f"Clearing price candidate: ${bids[market.available_impressions][1]:.2f}"
            if len(bids) > market.available_impressions
            else f"Impressions: {market.available_impressions} | All win (no losing bid)"
        )

        # Determine winners (second-price mechanism)
        winners_data: List[Dict[str, Any]] = []
        losers_data: List[Dict[str, Any]] = []

        for i, (agent, bid) in enumerate(bids):
            if i < market.available_impressions:
                # Winner pays second-highest price (or their own bid if last)
                if i + 1 < len(bids):
                    pay_price = bids[i + 1][1]
                else:
                    pay_price = bid * 0.9  # Slight discount if no competition below

                pay_price = round(pay_price, 2)

                # Estimate conversions
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

        # Trace: verify VCG math — winner pays ≤ own bid
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
        logger.debug(
            f"[Round {market.round_number} payments] " +
            " | ".join(
                f"{w['agent_name']}: bid=${w['bid']:.2f} paid=${w['paid']:.2f}"
                for w in winners_data
            )
        )

        return result