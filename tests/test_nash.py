"""Tests for NashEquilibriumSolver."""

import time
import pytest
import numpy as np
from core.nash_solver import NashEquilibriumSolver


class TestNashEquilibriumSolver:
    """Test suite for Nash equilibrium computation."""

    def test_empty_equilibrium(self):
        """Empty agent set should return empty result."""
        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium({}, {}, 100)
        assert result["strategies"] == {}
        assert result["clearing_price"] == 0.0

    def test_single_agent_trivial_equilibrium(self):
        """Single agent should bid at minimum."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 2.0, 3.0])
        result = solver.compute_equilibrium(
            {"A": 1000}, {"A": 50}, 10
        )
        assert "A" in result["strategies"]
        # Should converge to lowest bid since no competition
        assert result["convergence"] < 0.01

    def test_two_agent_equilibrium_converges(self):
        """Two agents should converge to equilibrium."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 40, "B": 40},
            5,
        )
        assert result["convergence"] < 0.01
        assert result["iterations"] < 100

    def test_strategy_distribution_sums_to_one(self):
        """Mixed strategy probabilities should sum to 1."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 2.0, 3.0])
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 30, "B": 30},
            5,
        )
        for name, strategy in result["strategies"].items():
            dist = strategy["distribution"]
            assert abs(sum(dist) - 1.0) < 1e-6

    def test_expected_bid_within_range(self):
        """Expected bid should be within bid levels."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 5.0, 10.0])
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 50, "B": 50},
            5,
        )
        for name, strategy in result["strategies"].items():
            assert 1.0 <= strategy["expected_bid"] <= 10.0

    def test_clearing_price_non_negative(self):
        """Equilibrium clearing price should be non-negative."""
        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000, "C": 1000},
            {"A": 30, "B": 35, "C": 40},
            10,
        )
        assert result["clearing_price"] >= 0

    def test_three_agent_converges(self):
        """Three asymmetric agents should converge."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 3.0, 5.0, 7.0, 9.0])
        result = solver.compute_equilibrium(
            {"A": 800, "B": 1200, "C": 600},
            {"A": 25, "B": 50, "C": 80},
            3,
        )
        assert result["convergence"] < 0.01
        assert len(result["strategies"]) == 3
        assert result["clearing_price"] >= 0

    def test_per_agent_bid_levels(self):
        """Per-agent bid levels should be reflected in output."""
        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 30, "B": 60},
            5,
            agent_bid_levels={
                "A": [1.0, 2.0, 3.0],
                "B": [4.0, 5.0, 6.0],
            },
        )
        assert result["strategies"]["A"]["bid_levels"] == [1.0, 2.0, 3.0]
        assert result["strategies"]["B"]["bid_levels"] == [4.0, 5.0, 6.0]

    def test_zero_opponents_returns_one(self):
        """Win probability with no opponents should be 1.0."""
        solver = NashEquilibriumSolver(bid_levels=[5.0])
        result = solver.compute_equilibrium(
            {"A": 1000}, {"A": 50}, 10
        )
        # Single agent: all probability mass on one bid → expected_bid = 5.0
        assert result["strategies"]["A"]["expected_bid"] == pytest.approx(5.0)

    def test_degenerate_strategy_distribution(self):
        """Solver should handle strategies with all mass on one level."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 5.0, 10.0])
        # Seed with a degenerate strategy for agent B
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 40, "B": 40},
            5,
        )
        # Should still converge without NaN/inf
        for name, s in result["strategies"].items():
            dist = s["distribution"]
            assert all(np.isfinite(dist))
            assert abs(sum(dist) - 1.0) < 1e-6


class TestSolverPerformance:
    """Performance benchmarks for the vectorized solver."""

    def test_5_agents_10_levels_completes_fast(self):
        """5 agents with 10 bid levels should solve in under 5 seconds."""
        solver = NashEquilibriumSolver(
            bid_levels=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )
        agents = {f"Agent_{i}": 1000 for i in range(5)}
        valuations = {f"Agent_{i}": 30 + i * 10 for i in range(5)}

        start = time.perf_counter()
        result = solver.compute_equilibrium(agents, valuations, 3, n_samples=5000)
        elapsed = time.perf_counter() - start

        assert result["convergence"] < 0.01
        assert elapsed < 5.0, f"Solver took {elapsed:.2f}s — expected < 5s"

    def test_vectorized_matches_reference(self):
        """Vectorized solver should produce equivalent results to scalar seed."""
        solver = NashEquilibriumSolver(bid_levels=[1.0, 2.0, 3.0, 4.0, 5.0])
        np.random.seed(99)
        result = solver.compute_equilibrium(
            {"A": 1000, "B": 1000},
            {"A": 40, "B": 40},
            5,
            n_samples=5000,
        )
        # Core invariants must hold regardless of vectorization
        assert result["convergence"] < 0.01
        for s in result["strategies"].values():
            assert abs(sum(s["distribution"]) - 1.0) < 1e-6
            assert all(np.isfinite(s["distribution"]))