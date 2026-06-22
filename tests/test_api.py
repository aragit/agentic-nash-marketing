"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for /health."""

    def test_health_returns_200(self):
        """Health check should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_structure(self):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "llm_backend" in data
        assert "database" in data

    def test_health_status_is_healthy(self):
        """Status should be 'healthy'."""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


class TestNashEndpoint:
    """Test suite for /nash/compute."""

    def test_nash_compute_returns_200(self):
        """Nash compute should return 200 for valid input."""
        response = client.post(
            "/nash/compute",
            json={
                "budgets": {"A": 1000, "B": 1000},
                "valuations": {"A": 30, "B": 30},
                "impression_supply": 10,
            },
        )
        assert response.status_code == 200

    def test_nash_compute_returns_strategies(self):
        """Response should contain strategies."""
        response = client.post(
            "/nash/compute",
            json={
                "budgets": {"A": 1000, "B": 1000},
                "valuations": {"A": 30, "B": 30},
                "impression_supply": 10,
            },
        )
        data = response.json()
        assert "strategies" in data
        assert "clearing_price" in data
        assert "convergence" in data
        assert "iterations" in data

    def test_nash_compute_invalid_input_returns_400(self):
        """Invalid input should return 400."""
        response = client.post("/nash/compute", json={})
        assert response.status_code == 400


class TestSimulationEndpoints:
    """Test suite for simulation endpoints."""

    def test_list_simulations_returns_200(self):
        """GET /simulations should return 200."""
        response = client.get("/simulations")
        assert response.status_code == 200

    def test_list_simulations_returns_list(self):
        """Response should be a list."""
        response = client.get("/simulations")
        assert isinstance(response.json(), list)

    def test_get_nonexistent_simulation_returns_404(self):
        """GET /simulation/99999 should return 404."""
        response = client.get("/simulation/99999")
        assert response.status_code == 404