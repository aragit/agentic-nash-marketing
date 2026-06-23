"""FastAPI application for Nash Marketing Agents."""

import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from configs.settings import settings
from database.models import get_db, SessionLocal, Simulation, AgentRecord, AuctionRound
from database.connection import init_database
from core.llm_engine import LLMEngineFactory
from core.agents import BrandAgent
from core.market import MarketSimulator
from core.auction import AuctionEngine
from core.nash_solver import NashEquilibriumSolver
from core.guardrails import BudgetGuardrail
from api.schemas import (
    RunSimulationRequest, SimulationSummary, SimulationDetail,
    AgentPerformance, NashEquilibriumResponse, HealthResponse,
)

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nash Marketing Agents",
    description="Multi-Agent Competitive Ad Auction with Nash Equilibrium",
    version="1.0.0",
)

# Background task executor
executor = ThreadPoolExecutor(max_workers=2)
running_simulations = {}

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_database()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/static/index.html">
    </head>
    <body>
        <p>Redirecting to dashboard...</p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        llm_backend=settings.llm_backend,
        database="connected" if "sqlite" in settings.database_url else "unknown",
    )


@app.post("/simulation/run", response_model=SimulationSummary)
async def run_simulation(
    request: RunSimulationRequest,
    db: Session = Depends(get_db),
):
    """Run a new auction simulation asynchronously."""
    logger.info(f"Starting simulation: {request.name} with {len(request.agents)} agents")

    # Create simulation record
    sim = Simulation(
        name=request.name,
        total_agents=len(request.agents),
        status="running",
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    sim_id = sim.id  # Capture ID before closing request session

    def _run_simulation():
        """Background simulation runner with its own DB session."""
        # CRITICAL: Create new session for this thread — don't reuse FastAPI's
        thread_db = SessionLocal()
        try:
            llm = LLMEngineFactory.create(use_mock=(settings.llm_backend == "mock"))
            market = MarketSimulator(seed=request.seed)
            engine = AuctionEngine(market)

            # Create agents
            agents = []
            for cfg in request.agents:
                agent = BrandAgent(
                    name=cfg.name,
                    role=cfg.role,
                    budget=cfg.budget,
                    target_cpa=cfg.target_cpa,
                    llm=llm,
                )
                agents.append(agent)

                # Record agent in DB
                agent_record = AgentRecord(
                    simulation_id=sim_id,
                    name=cfg.name,
                    role=cfg.role,
                    total_budget=cfg.budget,
                    remaining_budget=cfg.budget,
                    target_cpa=cfg.target_cpa,
                )
                thread_db.add(agent_record)

            thread_db.commit()

            # Run rounds
            for round_num in range(request.rounds):
                result = engine.run_round(agents)

                # Record round
                round_record = AuctionRound(
                    simulation_id=sim_id,
                    round_number=result.round_number,
                    clearing_price=result.clearing_price,
                    total_revenue=result.total_revenue,
                    available_impressions=result.market_state.available_impressions,
                    audience_quality=result.market_state.audience_quality,
                    seasonality=result.market_state.seasonality_factor,
                    winner_count=len(result.winners),
                    loser_count=len(result.losers),
                    winners=result.winners,
                    losers=result.losers,
                )
                thread_db.add(round_record)

                # Update agent records
                for agent in agents:
                    record = thread_db.query(AgentRecord).filter_by(
                        simulation_id=sim_id, name=agent.name
                    ).first()
                    if record:
                        record.remaining_budget = agent.state.remaining_budget
                        record.impressions_won = agent.state.impressions_won
                        record.total_spent = agent.state.total_spent
                        record.total_conversions = agent.state.total_conversions
                        record.final_win_rate = agent.state.win_rate
                        record.final_strategy = agent.state.role

                thread_db.commit()

                # Stop if all budgets depleted
                active = [a for a in agents if a.state.remaining_budget > 0]
                if len(active) < 2:
                    logger.info(f"Simulation ended early at round {round_num + 1}: only {len(active)} agents active")
                    break

            # Finalize simulation immediately — set completed BEFORE slow Nash solver
            sim_record = thread_db.query(Simulation).filter(Simulation.id == sim_id).first()
            if sim_record:
                sim_record.status = "completed"
                sim_record.total_rounds = len(engine.history)
                sim_record.total_revenue = sum(r.total_revenue for r in engine.history)
                sim_record.final_clearing_price = engine.history[-1].clearing_price if engine.history else 0.0
                thread_db.commit()

            llm.shutdown()
            running_simulations[sim_id] = {"status": "completed", "sim_id": sim_id}
            logger.info(f"Simulation {sim_id} completed successfully")

            # Compute Nash equilibrium post-hoc (slow — don't block UI polling)
            try:
                budgets = {a.name: a.state.total_budget for a in agents}
                valuations = {a.name: a.state.target_cpa for a in agents}
                # Per-agent bid levels based on CPA × role range
                # Wider ranges for Nash post-hoc — lets low-CPA agents compete with high-CPA agents
                # (LLM simulation uses tighter ranges for realistic bids)
                nash_role_ranges = {"aggressive": (0.50, 1.20), "balanced": (0.40, 0.90), "conservative": (0.10, 0.60)}
                agent_bid_levels = {}
                for a in agents:
                    lo, hi = nash_role_ranges.get(a.state.role, (0.35, 0.60))
                    cpa = a.state.target_cpa
                    levels = sorted(set([
                        round(cpa * pct, 2)
                        for pct in [lo + (hi - lo) * i / 9 for i in range(10)]
                    ]))
                    agent_bid_levels[a.name] = levels if levels else [max(1.0, cpa * lo)]
                nash_supply = max(1, int(len(agents) * 0.7))  # match simulation: gauss(agents*0.7, 1)
                solver = NashEquilibriumSolver()
                nash = solver.compute_equilibrium(
                    budgets, valuations, impression_supply=nash_supply, agent_bid_levels=agent_bid_levels
                )
                if sim_record:
                    sim_record.nash_equilibrium = nash
                    thread_db.commit()
            except Exception as nash_err:
                logger.warning(f"Nash computation failed for sim {sim_id}: {nash_err}")

        except Exception as e:
            # Mark as failed
            try:
                sim_record = thread_db.query(Simulation).filter(Simulation.id == sim_id).first()
                if sim_record:
                    sim_record.status = "failed"
                    thread_db.commit()
            except:
                pass
            logger.error(f"Simulation {sim_id} failed: {e}")
            running_simulations[sim_id] = {"status": "failed", "error": str(e)}
        finally:
            thread_db.close()

    # Start background task
    running_simulations[sim_id] = {"status": "running"}
    asyncio.get_event_loop().run_in_executor(executor, _run_simulation)

    # Return immediately with running status
    return SimulationSummary(
        id=sim_id,
        name=request.name,
        total_rounds=0,
        total_agents=len(request.agents),
        final_clearing_price=0.0,
        total_revenue=0.0,
        status="running",
        created_at=sim.created_at,
    )


@app.get("/simulation/{sim_id}", response_model=SimulationDetail)
async def get_simulation(sim_id: int, db: Session = Depends(get_db)):
    """Get detailed simulation results."""
    sim = db.query(Simulation).filter(Simulation.id == sim_id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    agents = [
        {
            "name": a.name,
            "role": a.role,
            "total_budget": a.total_budget,
            "remaining_budget": a.remaining_budget,
            "impressions_won": a.impressions_won,
            "total_spent": a.total_spent,
            "total_conversions": a.total_conversions,
            "win_rate": a.final_win_rate,
            "strategy": a.final_strategy,
        }
        for a in sim.agents
    ]

    rounds = [
        {
            "round_number": r.round_number,
            "clearing_price": r.clearing_price,
            "total_revenue": r.total_revenue,
            "available_impressions": r.available_impressions,
            "audience_quality": r.audience_quality,
            "seasonality": r.seasonality,
            "winner_count": r.winner_count,
            "loser_count": r.loser_count,
            "winners": r.winners,
            "losers": r.losers,
        }
        for r in sim.rounds
    ]

    return SimulationDetail(
        id=sim.id,
        name=sim.name,
        total_rounds=sim.total_rounds,
        total_agents=sim.total_agents,
        final_clearing_price=sim.final_clearing_price,
        total_revenue=sim.total_revenue,
        status=sim.status,
        created_at=sim.created_at,
        agents=agents,
        rounds=rounds,
        nash_equilibrium=sim.nash_equilibrium,
    )


@app.get("/simulations", response_model=List[SimulationSummary])
async def list_simulations(db: Session = Depends(get_db)):
    """List all simulations."""
    sims = db.query(Simulation).order_by(Simulation.created_at.desc()).all()
    return [
        SimulationSummary(
            id=s.id,
            name=s.name,
            total_rounds=s.total_rounds,
            total_agents=s.total_agents,
            final_clearing_price=s.final_clearing_price,
            total_revenue=s.total_revenue,
            status=s.status,
            created_at=s.created_at,
        )
        for s in sims
    ]


@app.post("/nash/compute", response_model=NashEquilibriumResponse)
async def compute_nash(request: Dict[str, Any]):
    """Compute Nash equilibrium for given agents."""
    try:
        budgets = request.get("budgets", {})
        valuations = request.get("valuations", {})
        supply = request.get("impression_supply", 100)

        solver = NashEquilibriumSolver()
        result = solver.compute_equilibrium(budgets, valuations, supply)

        return NashEquilibriumResponse(
            strategies=result["strategies"],
            clearing_price=result["clearing_price"],
            convergence=result["convergence"],
            iterations=result["iterations"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))