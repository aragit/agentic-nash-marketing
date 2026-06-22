"""Tests for BudgetGuardrail."""

import pytest
from core.guardrails import BudgetGuardrail, GuardrailAction


class TestBudgetGuardrail:
    """Test suite for budget guardrails."""

    def test_healthy_budget_allows_bid(self):
        """Healthy budget (>20%) should allow bid."""
        guard = BudgetGuardrail()
        action = guard.check("Nike", 100.0, remaining=4000.0, total=5000.0)
        assert action.action == "allow"
        assert action.adjusted_bid == 100.0

    def test_soft_warning_at_15_percent(self):
        """15% remaining should trigger soft warning."""
        guard = BudgetGuardrail()
        action = guard.check("Nike", 100.0, remaining=750.0, total=5000.0)
        assert action.action == "allow"
        assert "SOFT WARNING" in action.reason

    def test_hard_cap_at_8_percent(self):
        """8% remaining should cap bid."""
        guard = BudgetGuardrail()
        action = guard.check("Nike", 100.0, remaining=400.0, total=5000.0)
        assert action.action == "cap"
        assert action.adjusted_bid <= 20.0  # 5% of 400

    def test_emergency_at_3_percent(self):
        """3% remaining should force minimum bid."""
        guard = BudgetGuardrail()
        action = guard.check("Nike", 100.0, remaining=150.0, total=5000.0)
        assert action.action == "emergency"
        assert action.adjusted_bid <= 50.0  # 1% of 5000

    def test_system_status_all_healthy(self):
        """All healthy budgets should return HEALTHY."""
        guard = BudgetGuardrail()
        budgets = {
            "A": {"remaining": 4000, "total": 5000},
            "B": {"remaining": 4500, "total": 5000},
        }
        status = guard.get_system_status(budgets)
        assert status["A"]["status"] == "HEALTHY"
        assert status["B"]["status"] == "HEALTHY"

    def test_system_status_mixed(self):
        """Mixed budgets should reflect correct statuses."""
        guard = BudgetGuardrail()
        budgets = {
            "A": {"remaining": 4000, "total": 5000},   # 80% - healthy
            "B": {"remaining": 400, "total": 5000},    # 8% - warning
            "C": {"remaining": 100, "total": 5000},    # 2% - critical
        }
        status = guard.get_system_status(budgets)
        assert status["A"]["status"] == "HEALTHY"
        assert status["B"]["status"] == "WARNING"
        assert status["C"]["status"] == "CRITICAL"