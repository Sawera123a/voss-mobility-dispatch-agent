"""
SQLAlchemy ORM models - these define the tables that get created in
the voss_mobility PostgreSQL database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base


class DispatchRun(Base):
    __tablename__ = "dispatch_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    forecast_summary = Column(Text)
    raw_agent_output = Column(Text)

    moves = relationship("DispatchMove", back_populates="run", cascade="all, delete-orphan")


class DispatchMove(Base):
    __tablename__ = "dispatch_moves"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("dispatch_runs.id"))
    vehicles = Column(Integer)
    from_zone = Column(String)
    to_zone = Column(String)
    reason = Column(Text)
    approved = Column(Boolean)  
    rejection_reason = Column(Text, nullable=True)

    run = relationship("DispatchRun", back_populates="moves")


class DemandForecastLog(Base):
    __tablename__ = "demand_forecast_log"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    zone = Column(String)
    hour = Column(Integer)
    predicted_demand = Column(Float)
