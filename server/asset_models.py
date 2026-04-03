"""
Physical asset models for VPP environment.
Defines Battery, Solar, and EV assets with realistic parameters.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BatteryAsset:
    """Lithium-ion battery storage system."""
    asset_id: str = "battery_1"
    capacity_kwh: float = 10.0  # 10 kWh capacity
    max_power_kw: float = 5.0  # 5 kW charge/discharge rate
    efficiency_roundtrip: float = 0.92  # 92% round-trip efficiency
    degradation_rate: float = 0.0005  # Per kWh cycled (0.05% per 100 kWh)
    min_soc: float = 0.1  # Cannot discharge below 10%
    max_soc: float = 0.95  # Cannot charge above 95%
    
    def __post_init__(self):
        self.soc: float = 0.5  # Start at 50% charge
        self.power_kw: float = 0.0  # Current power output (negative = charging)


@dataclass
class SolarAsset:
    """Rooftop solar PV system."""
    asset_id: str = "solar_1"
    capacity_kw: float = 5.0  # 5 kW peak capacity
    
    def __post_init__(self):
        self.generation_kw: float = 0.0  # Current generation


@dataclass
class EvAsset:
    """Electric vehicle with home charger."""
    asset_id: str = "ev_1"
    battery_capacity_kwh: float = 60.0  # Tesla Model 3 capacity
    charger_max_power_kw: float = 7.0  # Single-phase home charger (7 kW)
    charging_efficiency: float = 0.98  # Charger efficiency
    
    def __post_init__(self):
        self.soc: float = 0.3  # Start at 30% charge
        self.charger_available: bool = True
        self.power_demand_kw: float = 0.0  # Current demand


@dataclass
class GridState:
    """Current grid conditions."""
    frequency_hz: float = 60.0
    voltage_v: float = 120.0
    market_price_per_mwh: float = 50.0
    household_load_kw: float = 1.0  # Baseline household consumption
