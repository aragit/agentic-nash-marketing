"""Stochastic market simulation for ad auction environment."""

import random
import logging
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current state of the ad market."""
    round_number: int
    available_impressions: int
    base_cpm: float  # Cost per mille (per 1000 impressions)
    audience_quality: float  # 0.0 to 1.0, affects conversion rate
    seasonality_factor: float  # 0.5 to 1.5, demand multiplier
    competitor_intensity: int  # Number of active competitors


class MarketSimulator:
    """Simulates ad market dynamics with stochastic impression supply."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.round = 0
        self.base_cpm = 2.50

    def step(self, active_agents: int) -> MarketState:
        """Advance one auction round."""
        self.round += 1

        # Stochastic impression supply — create scarcity vs active_agents for competitive auctions
        available = int(self.rng.gauss(max(2, active_agents * 0.7), 1))
        available = max(1, available)

        # Seasonality: sine wave + random noise
        seasonality = 1.0 + 0.3 * self.rng.gauss(0, 1)
        seasonality = max(0.5, min(1.5, seasonality))

        # Audience quality varies by segment
        audience_quality = self.rng.uniform(0.3, 0.9)

        # Competitor intensity affects clearing price
        competitor_intensity = active_agents

        # Base CPM drifts slightly each round
        self.base_cpm *= (1 + self.rng.gauss(0, 0.02))
        self.base_cpm = max(1.0, min(10.0, self.base_cpm))

        return MarketState(
            round_number=self.round,
            available_impressions=available,
            base_cpm=round(self.base_cpm, 2),
            audience_quality=round(audience_quality, 2),
            seasonality_factor=round(seasonality, 2),
            competitor_intensity=competitor_intensity,
        )

    def compute_clearing_price(self, bids: List[float], available: int) -> float:
        """Second-price auction: highest bids win, pay second-highest price."""
        if not bids or available <= 0:
            return self.base_cpm

        sorted_bids = sorted(bids, reverse=True)
        winners = min(available, len(sorted_bids))

        if winners < len(sorted_bids):
            # Last winner pays the bid just below them (second-price)
            clearing = sorted_bids[winners]
        else:
            # Everyone wins, pay minimum bid
            clearing = min(sorted_bids) if sorted_bids else self.base_cpm

        return round(max(clearing, self.base_cpm * 0.5), 2)

    def estimate_conversions(self, impressions: int, audience_quality: float, bid: float) -> int:
        """Estimate conversions based on impressions, quality, and bid."""
        # Higher bid = better placement = higher CTR
        ctr = 0.01 + (bid / 10.0) * 0.05  # 1% to 6% CTR
        ctr *= audience_quality
        conversions = int(impressions * ctr * self.rng.uniform(0.8, 1.2))
        return max(0, conversions)