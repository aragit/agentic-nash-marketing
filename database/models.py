"""SQLAlchemy models for Nash Marketing Agents."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    ForeignKey, JSON, Boolean, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from configs.settings import settings

Base = declarative_base()


class Simulation(Base):
    """A complete auction simulation episode."""
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String, default="unnamed")
    total_rounds = Column(Integer, default=0)
    total_agents = Column(Integer, default=0)
    final_clearing_price = Column(Float, default=0.0)
    total_revenue = Column(Float, default=0.0)
    nash_equilibrium = Column(JSON, default=dict)
    status = Column(String, default="running")  # running, completed, failed

    # Relationships
    rounds = relationship("AuctionRound", back_populates="simulation", cascade="all, delete")
    agents = relationship("AgentRecord", back_populates="simulation", cascade="all, delete")


class AgentRecord(Base):
    """Persistent record of an agent's performance in a simulation."""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    name = Column(String)
    role = Column(String)
    total_budget = Column(Float)
    remaining_budget = Column(Float)
    target_cpa = Column(Float)
    impressions_won = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    total_conversions = Column(Integer, default=0)
    final_win_rate = Column(Float, default=0.0)
    final_strategy = Column(String)

    simulation = relationship("Simulation", back_populates="agents")


class AuctionRound(Base):
    """Individual auction round results."""
    __tablename__ = "auction_rounds"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    round_number = Column(Integer)
    clearing_price = Column(Float)
    total_revenue = Column(Float)
    available_impressions = Column(Integer)
    audience_quality = Column(Float)
    seasonality = Column(Float)
    winner_count = Column(Integer)
    loser_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # JSON arrays of winner/loser details
    winners = Column(JSON, default=list)
    losers = Column(JSON, default=list)

    simulation = relationship("Simulation", back_populates="rounds")


class BidRecord(Base):
    """Individual bid record for audit trail."""
    __tablename__ = "bids"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    round_number = Column(Integer)
    agent_name = Column(String)
    bid_amount = Column(Float)
    paid_amount = Column(Float, default=0.0)
    won = Column(Boolean, default=False)
    conversions = Column(Integer, default=0)
    strategy = Column(String)
    justification = Column(String)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create engine and session
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()