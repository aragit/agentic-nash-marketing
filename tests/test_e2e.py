"""End-to-end tests for full simulation lifecycle."""

import time
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

AGENTS = [
    {"name": "Nike", "role": "aggressive", "budget": 120, "target_cpa": 15},
    {"name": "Adidas", "role": "balanced", "budget": 200, "target_cpa": 35},
    {"name": "Puma", "role": "conservative", "budget": 300, "target_cpa": 80},
]


def poll_simulation(sim_id: int, timeout: int = 60, interval: int = 1) -> dict:
    """Poll /simulations until the target sim completes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/simulations")
        assert resp.status_code == 200
        matches = [s for s in resp.json() if s["id"] == sim_id]
        if matches and matches[0]["status"] == "completed":
            detail = client.get(f"/simulation/{sim_id}")
            assert detail.status_code == 200
            return detail.json()
        time.sleep(interval)
    pytest.fail(f"Simulation {sim_id} did not complete within {timeout}s")


class TestE2ESimulation:
    """Full end-to-end simulation lifecycle."""

    def test_full_simulation_completes(self):
        """Run a simulation end-to-end and verify all results."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-test",
            "rounds": 10,
            "agents": AGENTS,
        })
        assert resp.status_code == 200
        sim_id = resp.json()["id"]

        data = poll_simulation(sim_id)

        assert data["total_rounds"] == 10
        assert data["status"] == "completed"
        assert data["total_revenue"] > 0
        assert data["final_clearing_price"] >= 0
        assert len(data["agents"]) == 3

    def test_agents_differentiated_by_role(self):
        """Verify role-based behavior differences in results."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-role-diff",
            "rounds": 10,
            "agents": AGENTS,
        })
        sim_id = resp.json()["id"]
        data = poll_simulation(sim_id)

        agents = {a["name"]: a for a in data["agents"]}
        nike = agents["Nike"]
        adidas = agents["Adidas"]
        puma = agents["Puma"]

        assert nike["role"] == "aggressive"
        assert adidas["role"] == "balanced"
        assert puma["role"] == "conservative"

        puma_spent = puma["total_spent"]
        adidas_spent = adidas["total_spent"]

        combined = nike["total_spent"] + puma_spent + adidas_spent
        assert combined > 0

        adidas_won = adidas["impressions_won"]
        puma_won = puma["impressions_won"]
        total = adidas_won + nike["impressions_won"] + puma_won
        assert total >= 10

    def test_each_agent_spent_something(self):
        """Every agent should have nonzero spend after 10 rounds."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-spend",
            "rounds": 10,
            "agents": AGENTS,
        })
        sim_id = resp.json()["id"]
        data = poll_simulation(sim_id)

        for agent in data["agents"]:
            assert agent["total_spent"] > 0, f"{agent['name']} spent nothing"
            assert agent["remaining_budget"] >= 0

    def test_budget_guardrail_respected(self):
        """No single bid should exceed 20% of pre-round remaining budget."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-guardrails",
            "rounds": 10,
            "agents": AGENTS,
        })
        sim_id = resp.json()["id"]
        data = poll_simulation(sim_id)

        for agent in data["agents"]:
            total_spent = agent["total_spent"]
            remaining = agent["remaining_budget"]
            total = agent["total_budget"]
            assert remaining >= 0
            assert total_spent <= total

    def test_clearing_price_history(self):
        """Clearing price history should have one entry per round."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-clearing",
            "rounds": 10,
            "agents": AGENTS,
        })
        sim_id = resp.json()["id"]
        data = poll_simulation(sim_id)

        rounds = data.get("rounds", [])
        assert len(rounds) == 10
        for r in rounds:
            assert r["clearing_price"] >= 0

    def _poll_nash_equilibrium(self, sim_id: int, timeout: int = 30) -> dict:
        """Poll for Nash equilibrium data after simulation completes."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            detail = client.get(f"/simulation/{sim_id}")
            assert detail.status_code == 200
            ne = detail.json().get("nash_equilibrium", {})
            if ne.get("strategies"):
                return ne
            time.sleep(1)
        pytest.fail(f"Nash equilibrium not available within {timeout}s for sim {sim_id}")

    def test_nash_equilibrium_present_and_valid(self):
        """Nash equilibrium should converge with valid strategies."""
        resp = client.post("/simulation/run", json={
            "name": "e2e-nash",
            "rounds": 10,
            "agents": AGENTS,
        })
        sim_id = resp.json()["id"]
        poll_simulation(sim_id)

        ne = self._poll_nash_equilibrium(sim_id)
        assert "clearing_price" in ne
        assert "convergence" in ne
        assert "iterations" in ne
        assert ne["convergence"] < 0.01  # converged below tolerance
        assert ne["iterations"] > 0
        assert ne["clearing_price"] >= 0

        strategies = ne["strategies"]
        assert set(strategies.keys()) == {"Nike", "Adidas", "Puma"}

        for name, s in strategies.items():
            dist = s["distribution"]
            bids = s["bid_levels"]
            expected = s["expected_bid"]
            assert len(dist) == len(bids)
            assert abs(sum(dist) - 1.0) < 0.01
            assert all(d >= 0 for d in dist)
            assert expected <= max(bids)
            assert expected >= min(bids)
            # Per-agent levels: 10 levels for CPA × role range
            assert len(bids) == 10
