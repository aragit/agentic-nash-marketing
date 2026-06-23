"""Property-based tests for economic guarantees.

Tests assert invariant properties of the auction system:
- Monotonicity: Higher CPA → higher win rate
- Individual rationality: No agent overpays
- Nash convergence: Solver converges to valid bounds
"""

import pytest
from core.market import MarketSimulator
from core.auction import AuctionEngine
from core.agents import BrandAgent
from core.llm_engine import MockLLMEngine
from core.nash_solver import NashEquilibriumSolver


class TestMonotonicity:
    """Higher target CPA → higher bid → higher win rate."""

    @pytest.fixture
    def engine(self):
        return AuctionEngine(MarketSimulator(seed=42))

    def test_higher_cpa_higher_win_rate(self, engine):
        """Agents with higher CPA should win more (same role isolates CPA)."""
        llm = MockLLMEngine(seed=42)
        agents = [
            BrandAgent("Low", "balanced", 5000, 10, llm),
            BrandAgent("Mid", "balanced", 5000, 50, llm),
            BrandAgent("High", "balanced", 5000, 100, llm),
        ]
        for _ in range(20):
            engine.run_round(agents)
        rates = [a.state.win_rate for a in agents]
        assert rates[2] >= rates[1] >= rates[0], (
            f"Monotonicity violated: Low={rates[0]:.1%}, Mid={rates[1]:.1%}, High={rates[2]:.1%}"
        )

    def test_higher_cpa_higher_bid(self, engine):
        """Bid should increase monotonically with CPA."""
        llm = MockLLMEngine(seed=42)
        agents = [
            BrandAgent("Low", "balanced", 5000, 10, llm),
            BrandAgent("Mid", "balanced", 5000, 50, llm),
            BrandAgent("High", "balanced", 5000, 100, llm),
        ]
        all_bids = {a.name: [] for a in agents}
        for _ in range(5):
            for a in agents:
                result = a.decide_bid(market_price=2.50, competitor_count=2, available_impressions=5)
                all_bids[a.name].append(result["bid"])
        avg_bids = {name: sum(bids) / len(bids) for name, bids in all_bids.items()}
        assert avg_bids["High"] > avg_bids["Mid"] > avg_bids["Low"], (
            f"Bid monotonicity violated: Low=${avg_bids['Low']:.2f}, Mid=${avg_bids['Mid']:.2f}, High=${avg_bids['High']:.2f}"
        )


class TestIndividualRationality:
    """Agents should never pay more than their bid or exceed budget guardrails."""

    @pytest.fixture
    def engine(self):
        return AuctionEngine(MarketSimulator(seed=42))

    @pytest.fixture
    def agents(self):
        llm = MockLLMEngine(seed=42)
        return [
            BrandAgent("Nike", "aggressive", 5000, 15, llm),
            BrandAgent("Adidas", "balanced", 5000, 35, llm),
            BrandAgent("Puma", "conservative", 5000, 80, llm),
        ]

    def test_clearing_price_never_exceeds_winning_bid(self, engine, agents):
        """Second-price guarantee: winner pays ≤ own bid."""
        for _ in range(10):
            result = engine.run_round(agents)
            for w in result.winners:
                assert w["paid"] <= w["bid"] + 0.01, (
                    f"{w['agent_name']} paid ${w['paid']:.2f} > bid ${w['bid']:.2f}"
                )

    def test_bid_within_budget_guardrail(self, engine, agents):
        """Bid should never exceed 20% of pre-round remaining budget."""
        for _ in range(10):
            result = engine.run_round(agents)
            for entry in result.winners + result.losers:
                pre_remaining = entry["remaining_budget"] + entry.get("paid", 0)
                max_allowed = pre_remaining * 0.2 + 0.01
                assert entry["bid"] <= max_allowed, (
                    f"{entry['agent_name']} bid ${entry['bid']:.2f} > 20% of ${pre_remaining:.2f}"
                )


class TestNashBounds:
    """Nash solver should converge to economically valid bounds."""

    def test_solver_converges(self):
        """Solver should reach tolerance within max iterations."""
        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium(
            {"Nike": 5000, "Adidas": 5000, "Puma": 5000},
            {"Nike": 15, "Adidas": 35, "Puma": 80},
            impression_supply=1,
        )
        assert result["convergence"] < 0.01, f"Nash did not converge: {result['convergence']}"
        assert result["iterations"] < 100, f"Nash exceeded max iterations: {result['iterations']}"

    def test_expected_bid_within_valuation(self):
        """No agent should expect to bid above their valuation (IR in expectation)."""
        solver = NashEquilibriumSolver()
        valuations = {"Nike": 15, "Adidas": 35, "Puma": 80}
        result = solver.compute_equilibrium(
            {"Nike": 5000, "Adidas": 5000, "Puma": 5000},
            valuations,
            impression_supply=1,
        )
        for name, strategy in result["strategies"].items():
            assert strategy["expected_bid"] <= valuations[name] + 0.01, (
                f"{name} expected bid ${strategy['expected_bid']:.2f} > valuation ${valuations[name]:.2f}"
            )

    def test_expected_bid_positive(self):
        """Expected bids should be strictly positive."""
        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 10, "B": 50},
            impression_supply=1,
        )
        for name, strategy in result["strategies"].items():
            assert strategy["expected_bid"] > 0, f"{name} expected bid is zero"
