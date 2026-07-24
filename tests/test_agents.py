"""Tests for BrandAgent."""

import json
import pytest
from core.agents import BrandAgent


class TestBrandAgent:
    """Test suite for brand agent behavior."""

    def test_agent_initialization(self, nike_agent):
        """Agent should initialize with correct state."""
        assert nike_agent.name == "Nike"
        assert nike_agent.state.role == "aggressive"
        assert nike_agent.state.total_budget == 5000.0
        assert nike_agent.state.remaining_budget == 5000.0
        assert nike_agent.state.target_cpa == 30.0

    @pytest.mark.asyncio
    async def test_decide_bid_returns_valid_structure(self, nike_agent):
        """Bid decision should return required fields."""
        result = await nike_agent.decide_bid(
            market_price=3.50,
            competitor_count=2,
            available_impressions=100,
        )
        assert "bid" in result
        assert "max_daily_spend" in result
        assert "target_cpa" in result
        assert "strategy" in result
        assert "justification" in result

    @pytest.mark.asyncio
    async def test_bid_is_positive(self, nike_agent):
        """Bid should always be positive."""
        result = await nike_agent.decide_bid(3.50, 2, 100)
        assert result["bid"] > 0

    def test_agent_updates_state_after_win(self, nike_agent):
        """Winning should reduce remaining budget."""
        initial = nike_agent.state.remaining_budget
        nike_agent.observe_result(won=True, bid=10.0, conversions=1)
        assert nike_agent.state.remaining_budget == initial - 10.0
        assert nike_agent.state.impressions_won == 1

    def test_agent_updates_state_after_loss(self, nike_agent):
        """Losing should not reduce budget."""
        initial = nike_agent.state.remaining_budget
        nike_agent.observe_result(won=False, bid=10.0)
        assert nike_agent.state.remaining_budget == initial
        assert nike_agent.state.impressions_won == 0

    def test_win_rate_calculation(self, nike_agent):
        """Win rate should be accurate."""
        nike_agent.observe_result(won=True, bid=1.0)
        nike_agent.observe_result(won=False, bid=1.0)
        assert nike_agent.state.win_rate == 0.5

    def test_effective_cpa_calculation(self, nike_agent):
        """CPA should be total_spent / conversions."""
        nike_agent.observe_result(won=True, bid=100.0, conversions=2)
        assert nike_agent.state.effective_cpa == 50.0

    def test_effective_cpa_infinite_when_no_conversions(self, nike_agent):
        """CPA should be infinite with zero conversions."""
        nike_agent.observe_result(won=True, bid=100.0, conversions=0)
        assert nike_agent.state.effective_cpa == float('inf')

    @pytest.mark.asyncio
    async def test_bid_returns_raw_llm_proposal(self, nike_agent):
        """Agent should return the raw LLM bid without inline caps — guardrail is in the engine."""
        nike_agent.state.remaining_budget = 100.0
        result = await nike_agent.decide_bid(50.0, 2, 100)
        # The bid is whatever the LLM proposed (target_cpa × role_pct), not capped here
        assert result["bid"] > 0
