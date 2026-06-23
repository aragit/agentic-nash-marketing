"""Tests for AuctionEngine."""

import pytest
from core.market import MarketSimulator
from core.auction import AuctionEngine
from core.agents import BrandAgent
from core.llm_engine import MockLLMEngine


class TestAuctionEngine:
    """Test suite for auction mechanics."""

    @pytest.fixture
    def engine(self):
        return AuctionEngine(MarketSimulator(seed=42))

    @pytest.fixture
    def three_agents(self):
        llm = MockLLMEngine(seed=42)
        return [
            BrandAgent("Nike", "aggressive", 5000, 30, llm),
            BrandAgent("Adidas", "balanced", 5000, 35, llm),
            BrandAgent("Puma", "conservative", 5000, 40, llm),
        ]

    def test_empty_auction(self, engine):
        """Auction with no agents should return empty result."""
        result = engine.run_round([])
        assert result.winners == []
        assert result.losers == []
        assert result.total_revenue == 0.0

    def test_some_agents_lose_when_supply_scarce(self, engine, three_agents):
        """With scarce impressions, not all agents win."""
        results = [engine.run_round(three_agents) for _ in range(10)]
        total_winners = sum(len(r.winners) for r in results)
        total_losers = sum(len(r.losers) for r in results)
        # Over 10 rounds, some rounds should have losers
        assert any(len(r.losers) > 0 for r in results), "No losers in any round — supply too high"
        assert total_winners + total_losers == 30  # 3 agents * 10 rounds

    def test_clearing_price_is_non_negative(self, engine, three_agents):
        """Clearing price should never be negative."""
        result = engine.run_round(three_agents)
        assert result.clearing_price >= 0

    def test_total_revenue_matches_winner_payments(self, engine, three_agents):
        """Revenue should equal sum of winner payments."""
        result = engine.run_round(three_agents)
        expected = sum(w["paid"] for w in result.winners)
        assert result.total_revenue == pytest.approx(expected, abs=0.01)

    def test_auction_history_grows(self, engine, three_agents):
        """History should accumulate after each round."""
        assert len(engine.history) == 0
        engine.run_round(three_agents)
        assert len(engine.history) == 1
        engine.run_round(three_agents)
        assert len(engine.history) == 2

    def test_budget_depletes_over_rounds(self, engine, three_agents):
        """Agents should spend budget over multiple rounds."""
        initial_budgets = [a.state.remaining_budget for a in three_agents]
        for _ in range(5):
            engine.run_round(three_agents)
        final_budgets = [a.state.remaining_budget for a in three_agents]
        for initial, final in zip(initial_budgets, final_budgets):
            assert final < initial