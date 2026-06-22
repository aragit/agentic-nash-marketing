"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class AgentConfig(BaseModel):
    """Configuration for a single brand agent."""
    name: str = Field(..., description="Brand name")
    role: str = Field(default="balanced", description="Strategy: aggressive, balanced, or conservative")
    budget: float = Field(default=5000.0, gt=0)
    target_cpa: float = Field(default=35.0, gt=0)


class RunSimulationRequest(BaseModel):
    """Request to run a new auction simulation."""
    name: str = Field(default="unnamed", description="Simulation name")
    agents: List[AgentConfig] = Field(default_factory=list, min_length=2, max_length=10)
    rounds: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)


class SimulationSummary(BaseModel):
    """Summary of a completed simulation."""
    id: int
    name: str
    total_rounds: int
    total_agents: int
    final_clearing_price: float
    total_revenue: float
    status: str
    created_at: datetime


class SimulationDetail(BaseModel):
    """Detailed simulation results."""
    id: int
    name: str
    total_rounds: int
    total_agents: int
    final_clearing_price: float
    total_revenue: float
    status: str
    created_at: datetime
    agents: List[Dict[str, Any]]
    rounds: List[Dict[str, Any]]
    nash_equilibrium: Optional[Dict[str, Any]]


class AgentPerformance(BaseModel):
    """Performance metrics for a single agent."""
    name: str
    role: str
    total_budget: float
    remaining_budget: float
    impressions_won: int
    total_spent: float
    win_rate: float
    effective_cpa: float


class NashEquilibriumResponse(BaseModel):
    """Nash equilibrium computation result."""
    strategies: Dict[str, Any]
    clearing_price: float
    convergence: float
    iterations: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    llm_backend: str
    database: str