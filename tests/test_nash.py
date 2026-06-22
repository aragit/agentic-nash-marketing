"""Tests for NashEquilibriumSolver."""

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