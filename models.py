"""
OpenEnv Pydantic models for Virtual Power Plant environment.
Defines typed Action, Observation, and State classes.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# NESTED TELEMETRY MODELS
# ============================================================================

class BatteryTelemetry(BaseModel):
    """Current state of a battery asset."""
    asset_id: str
    soc: float = Field(..., ge=0.0, le=1.0, description="State of Charge (0.0-1.0)")
    power_kw: float = Field(..., ge=-5.0, le=5.0, description="Current power (negative=charging)")
    degradation_cumulative: float = Field(..., ge=0.0, description="Cumulative degradation")


class SolarTelemetry(BaseModel):
    """Current state of a solar asset."""
    asset_id: str
    generation_kw: float = Field(..., ge=0.0, le=5.0, description="Current generation (kW)")


class EvTelemetry(BaseModel):
    """Current state of an EV asset."""
    asset_id: str
    soc: float = Field(..., ge=0.0, le=1.0, description="Battery state of charge (0.0-1.0)")
    charger_available: bool = Field(True, description="Can accept charge")
    power_demand_kw: float = Field(..., ge=0.0, le=7.0, description="Current demand (kW)")


# ============================================================================
# ACTION
# ============================================================================

class VppAction(Action):
    """Agent control signal sent each timestep.
    
    Defines how the agent wants to manage the portfolio:
    - global_charge_rate: Proportion to charge (positive) or discharge (negative)
    - battery_reserve_pct: Minimum SoC to maintain for grid support
    """
    global_charge_rate: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Portfolio charge rate: -1.0 (sell all) to +1.0 (buy all)"
    )
    battery_reserve_pct: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Safety buffer: minimum SoC not to breach during grid support"
    )


# ============================================================================
# OBSERVATION
# ============================================================================

class VppObservation(Observation):
    """State visible to agent each timestep.
    
    Includes:
    - Asset telemetry (battery, solar, EV status)
    - Grid conditions (frequency, voltage, price)
    - Forecasts (next 24h prices and solar generation)
    """
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    step_id: int = Field(..., ge=0, le=95, description="Step within episode (0-95)")
    
    # Asset telemetry
    battery_telemetry: List[BatteryTelemetry] = Field(..., description="Battery status")
    solar_telemetry: List[SolarTelemetry] = Field(..., description="Solar generation")
    ev_telemetry: List[EvTelemetry] = Field(..., description="EV status")
    
    # Grid conditions
    grid_frequency_hz: float = Field(..., ge=59.0, le=61.0, description="Grid frequency (Hz)")
    grid_voltage_v: float = Field(120.0, ge=100.0, le=140.0, description="Voltage (V)")
    market_price_per_mwh: float = Field(..., ge=0.0, description="Current market price ($/MWh)")
    
    # Forecasts
    forecast_next_24h_price: List[float] = Field(
        default_factory=list,
        description="Forecasted prices for next 24h (96 × 15-min intervals)"
    )
    forecast_next_24h_solar_kw: List[float] = Field(
        default_factory=list,
        description="Forecasted solar generation (96 values)"
    )
    # Reward and episode termination info (populated by the environment)
    reward: float = Field(0.0, description="Immediate reward for last action")
    done: bool = Field(False, description="Episode termination flag")
    cumulative_reward: float = Field(0.0, description="Cumulative reward so far")


# ============================================================================
# STATE (for grading and episode management)
# ============================================================================

class VppState(State):
    """God-view state used for grading and episode tracking.
    
    Hidden from agent; used for task scoring and debugging.
    """
    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(0, ge=0, description="Current step within episode")
    
    # Task configuration
    task_tier: str = Field(..., description="Task difficulty: 'easy', 'medium', or 'hard'")
    
    # Cumulative metrics
    cumulative_revenue_usd: float = Field(0.0, description="Total revenue ($)")
    cumulative_battery_degradation: float = Field(0.0, ge=0.0, description="Degradation amount")
    
    # Gate events
    grid_events_handled: int = Field(0, ge=0, description="Count of frequency emergencies responded to")
    battery_violation_count: int = Field(0, ge=0, description="Over-discharge violations")