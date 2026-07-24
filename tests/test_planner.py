"""Tests for StrategyPlanner (neural planner)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from core.planner import StrategyPlanner, VALID_STRATEGIES
from core.llm_engine import LLMResponse


def _make_llm_mock(return_content: str):
    """Create a mock LLM that returns a fixed string via async_chat_completion."""
    llm = MagicMock()
    llm.async_chat_completion = AsyncMock(return_value=LLMResponse(
        content=return_content,
        tokens_in=100,
        tokens_out=1,
        latency_ms=1.0,
        model="mock",
    ))
    return llm


class TestStrategyPlanner:
    """Test suite for the neural StrategyPlanner."""

    @pytest.mark.asyncio
    async def test_returns_aggressive_when_valid(self):
        llm = _make_llm_mock("aggressive")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.10)
        assert result == "aggressive"

    @pytest.mark.asyncio
    async def test_returns_balanced_when_valid(self):
        llm = _make_llm_mock("balanced")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.50)
        assert result == "balanced"

    @pytest.mark.asyncio
    async def test_returns_conserve_when_valid(self):
        llm = _make_llm_mock("conserve")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.15, 0.60)
        assert result == "conserve"

    @pytest.mark.asyncio
    async def test_strips_whitespace_and_case(self):
        llm = _make_llm_mock("  Balanced  ")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.50)
        assert result == "balanced"

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_response(self):
        llm = _make_llm_mock("I think you should bid higher")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.50)
        assert result == "balanced"

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self):
        llm = _make_llm_mock("")
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.50)
        assert result == "balanced"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_exception(self):
        llm = MagicMock()
        llm.async_chat_completion = AsyncMock(side_effect=RuntimeError("LLM offline"))
        planner = StrategyPlanner(llm_client=llm)
        result = await planner.evaluate_strategy(0.50, 0.50)
        assert result == "balanced"

    @pytest.mark.asyncio
    async def test_always_returns_valid_strategy(self):
        """Every code path must return one of the three valid strategies."""
        for content in ["aggressive", "balanced", "conserve", "garbage", "", None]:
            llm = _make_llm_mock(content or "")
            planner = StrategyPlanner(llm_client=llm)
            if content is None:
                llm.async_chat_completion = AsyncMock(side_effect=Exception("fail"))
            result = await planner.evaluate_strategy(0.50, 0.50)
            assert result in VALID_STRATEGIES, f"Bad result for input {content!r}"

    @pytest.mark.asyncio
    async def test_planner_receives_history_in_prompt(self):
        """The planner prompt should include clearing prices from history."""
        llm = MagicMock()
        llm.async_chat_completion = AsyncMock(return_value=LLMResponse(
            content="balanced", tokens_in=100, tokens_out=1,
            latency_ms=1.0, model="mock",
        ))
        planner = StrategyPlanner(llm_client=llm)
        history = [
            {"clearing_price": 3.50, "winners": [], "losers": []},
            {"clearing_price": 4.20, "winners": [], "losers": []},
        ]
        await planner.evaluate_strategy(0.50, 0.50, recent_history=history)

        call_args = llm.async_chat_completion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "$3.50" in prompt
        assert "$4.20" in prompt


class TestComputeRecentWinRate:
    """Test suite for the static win-rate calculator."""

    def test_empty_history(self):
        assert StrategyPlanner.compute_recent_win_rate("Nike", []) == 0.0

    def test_all_wins(self):
        history = [
            {"winners": [{"agent_name": "Nike"}], "losers": [{"agent_name": "Adidas"}]},
            {"winners": [{"agent_name": "Nike"}], "losers": [{"agent_name": "Puma"}]},
        ]
        assert StrategyPlanner.compute_recent_win_rate("Nike", history) == 1.0

    def test_all_losses(self):
        history = [
            {"winners": [{"agent_name": "Adidas"}], "losers": [{"agent_name": "Nike"}]},
            {"winners": [{"agent_name": "Puma"}], "losers": [{"agent_name": "Nike"}]},
        ]
        assert StrategyPlanner.compute_recent_win_rate("Nike", history) == 0.0

    def test_mixed_results(self):
        history = [
            {"winners": [{"agent_name": "Nike"}], "losers": [{"agent_name": "Adidas"}]},
            {"winners": [{"agent_name": "Adidas"}], "losers": [{"agent_name": "Nike"}]},
            {"winners": [{"agent_name": "Nike"}], "losers": [{"agent_name": "Puma"}]},
        ]
        assert StrategyPlanner.compute_recent_win_rate("Nike", history) == pytest.approx(2 / 3)

    def test_window_limits_lookback(self):
        history = [
            {"winners": [{"agent_name": "Nike"}], "losers": []},
            {"winners": [{"agent_name": "Adidas"}], "losers": [{"agent_name": "Nike"}]},
        ]
        # Window of 1 should only see the last round (loss)
        assert StrategyPlanner.compute_recent_win_rate("Nike", history, window=1) == 0.0
        # Window of 2 sees both (1 win, 1 loss)
        assert StrategyPlanner.compute_recent_win_rate("Nike", history, window=2) == 0.5
