 # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Vpp Environment.

The vpp environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# STATIC DATA (The Registry)
# Sent ONCE to the Agent at the start of the simulation.
class BatteryAsset(BaseModel):
    asset_id: str = Field(..., description="Unique ID for the home battery (e.g., 'home-001')")
    capacity_kwh: float = Field(..., description="Max energy storage capacity in kWh (e.g., 13.5)")
    max_power_kw: float = Field(..., description="Max charge/discharge speed in kW (e.g., 5.0)")
    efficiency_rt: float = Field(0.90, description="Round-trip efficiency. 0.90 means 10% energy is lost as heat.")

# DYNAMIC DATA (The Telemetry)
# Lightweight data that changes every single step.
class BatteryTelemetry(BaseModel):
    asset_id: str = Field(..., description="Must match an ID from the BatteryAsset registry")
    soc: float = Field(..., ge=0.0, le=1.0, description="Current State of Charge (0.0 to 1.0)")
    current_house_load_kw: float = Field(..., description="Real-time power drawn by the home's appliances")
    current_solar_gen_kw: float = Field(..., description="Real-time power produced by the home's solar panels")

# ACTION (The Dispatch Command)
# Sent by the Agent back to the Environment.
class VppAction(Action):
    # -1.0 means "Sell max power to grid", +1.0 means "Buy max power from grid"
    global_charge_rate: float = Field(..., ge=-1.0, le=1.0, description="Command sent to all batteries.")
    
    # The "Social Contract" constraint
    min_reserve_pct: float = Field(0.2, ge=0.0, le=1.0, description="Safety buffer. Do not discharge if SoC hits this level.")

# OBSERVATION (The World State)
# Sent to the Agent every 15 minutes (96 times a day).
class VppObservation(Observation):
    timestamp: datetime
    step_id: int = Field(..., description="Current 15-min interval index (0 to 95)")
    
    # The physical state of all 100 homes
    telemetry: List[BatteryTelemetry] 
    
    # The Grid Vitals (Triggers for the "Hard" Tasks)
    grid_frequency_hz: float = Field(50.0, description="Target: 50.0. Drop below 49.8 is an emergency.")
    grid_voltage_v: float = Field(230.0, description="Target: 230.0. Spike above 250V requires charging to absorb power.")
    market_price_per_mwh: float = Field(..., description="Current wholesale energy price in USD.")
    
    # The Forecasts (For Agent Reasoning)
    forecast_24h_price: List[float] = Field(..., description="Predicted prices for the next 96 steps.")
    forecast_24h_solar: List[float] = Field(..., description="Predicted solar intensity for the next 96 steps.")

# STATE (The Ground Truth)
# Hidden from the Agent. Used by the Engine to calculate scores.
class VppState(State):
    """The omniscient ground truth of the VPP Environment."""
    
    # --- 1. Temporal Trackers ---
    current_step: int = Field(..., description="The current 15-minute interval index (e.g., 42).")
    task_tier: str = Field(..., description="The active scenario: 'easy', 'medium', or 'hard'.")
    
    # --- 2. Financial Accumulators (For the Grader) ---
    cumulative_revenue_usd: float = Field(0.0, description="Total money earned from selling to the grid.")
    cumulative_cost_usd: float = Field(0.0, description="Total money spent buying from the grid.")
    cumulative_profit_usd: float = Field(0.0, description="Revenue minus Cost.")
    
    # --- 3. Safety & Performance Trackers (For Penalties) ---
    blackout_events_count: int = Field(0, description="Times the battery hit 0% while the home needed power.")
    safety_violations_count: int = Field(0, description="Times the agent drained the battery below min_reserve_pct.")
    grid_emergencies_ignored: int = Field(0, description="Times the grid frequency dropped but the agent did not discharge.")
    
    # --- 4. Physical Ground Truth (Hidden from Agent) ---
    actual_weather_mode: str = Field(..., description="e.g., 'clear_sky', 'rolling_clouds', 'storm'. Determines actual solar yield.")
    # We use a dictionary here for ultra-fast lookups in the Engine
    battery_true_soc: Dict[str, float] = Field(..., description="Dictionary mapping asset_id to its precise true SoC.")