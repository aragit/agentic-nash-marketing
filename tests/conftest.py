"""pytest fixtures and configuration."""

import pytest
from core.llm_engine import MockLLMEngine
from core.agents import BrandAgent


@pytest.fixture
def mock_llm():
    """Provide a MockLLM engine."""
    return MockLLMEngine(seed=42)


@pytest.fixture
def nike_agent(mock_llm):
    """Provide a Nike aggressive agent."""
    return BrandAgent("Nike", "aggressive", 5000.0, 30.0, mock_llm)


@pytest.fixture
def adidas_agent(mock_llm):
    """Provide an Adidas balanced agent."""
    return BrandAgent("Adidas", "balanced", 5000.0, 35.0, mock_llm)


@pytest.fixture
def puma_agent(mock_llm):
    """Provide a Puma conservative agent."""
    return BrandAgent("Puma", "conservative", 5000.0, 40.0, mock_llm)