"""
VPP Environment Implementation - FIXED VERSION
Fixes reward calculation errors that caused constant rewards.
"""

import csv
import os
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import List, Tuple, Dict
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import VppAction, VppObservation, VppState
    from ..models import BatteryTelemetry, SolarTelemetry, EvTelemetry
    from .asset_models import BatteryAsset, SolarAsset, EvAsset, GridState
except ImportError:
    try:
        from models import VppAction, VppObservation, VppState
        from models import BatteryTelemetry, SolarTelemetry, EvTelemetry
        from asset_models import BatteryAsset, SolarAsset, EvAsset, GridState
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models import VppAction, VppObservation, VppState
        from models import BatteryTelemetry, SolarTelemetry, EvTelemetry
        from server.asset_models import BatteryAsset, SolarAsset, EvAsset, GridState


class VppEnvironment(Environment):
    """Virtual Power Plant Environment Implementation."""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    # Class variable for grading (updated after each episode)
    _last_cumulative_reward: float = 0.0
    
    def __init__(self):
        """Initialize environment."""
        self._state: VppState = None
        self.battery = BatteryAsset()
        self.solar = SolarAsset()
        self.ev = EvAsset()
        self.grid = GridState()
        
        self.price_series: List[float] = []
        self.grid_frequency_series: List[float] = []
        
        self._step_count = 0
        self.cumulative_reward = 0.0
        self.solar_cloud_cover = 0.0
        
        # Track previous degradation for incremental penalty
        self.previous_degradation = 0.0
        
        # Cache data directory
        self.data_dir = Path(__file__).parent / "data"

        self.done = False
    
    def reset(self, task_id: str = "easy-arbitrage", seed: int = None, **kwargs) -> VppObservation:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self._state = VppState(
            episode_id=str(uuid4()),
            step_count=0,
            task_tier=task_id,
            cumulative_revenue_usd=0.0,
            cumulative_battery_degradation=0.0,
            grid_events_handled=0,
            battery_violation_count=0,
        )
        self._step_count = 0
        self.cumulative_reward = 0.0
        self.previous_degradation = 0.0  # Reset degradation tracking
        
        # Load scenario data
        if task_id == "easy-arbitrage":
            self.price_series = self._load_prices("prices_easy.csv")
            self.grid_frequency_series = self._generate_grid_frequency(0.1)
            self.solar_cloud_cover = 0.0
        elif task_id == "medium-forecast-error":
            self.price_series = self._load_prices("prices_medium.csv")
            self.grid_frequency_series = self._generate_grid_frequency(0.2)
            self.solar_cloud_cover = 0.1
        else:  # hard-frequency-response
            self.price_series = self._load_prices("prices_hard.csv")
            self.grid_frequency_series = self._generate_grid_frequency(0.5)
            self.solar_cloud_cover = 0.4
        
        # Initialize assets
        self.battery.soc = 0.5
        self.ev.soc = 0.3
        
        return self._generate_observation()
    
    def step(self, action: VppAction) -> VppObservation:
        """Execute one control step (15 minutes)."""
        self._step_count += 1
        step_idx = self._step_count % 96
        dt_hours = 0.25  # 15 minutes

        # ===================================================================
        # 1. PHYSICS: Update asset states
        # ===================================================================
        
        price = self.price_series[step_idx]
        solar_gen = self._generate_solar_at_step(step_idx)
        net_load = 1.0  # Household load in kW
        
        # --- Battery Dispatch ---
        # Action: global_charge_rate in [-1, 1]
        # Positive = charge (buy from grid), Negative = discharge (sell to grid)
        battery_power_kw = action.global_charge_rate * self.battery.max_power_kw
        
        # Calculate energy change with proper efficiency
        if battery_power_kw > 0:
            # CHARGING: energy stored = power × time × efficiency
            energy_change_kwh = battery_power_kw * dt_hours * self.battery.efficiency_roundtrip
        else:
            # DISCHARGING: energy delivered = power × time (already at inverter output)
            # Efficiency loss already accounted for in power measurement
            energy_change_kwh = battery_power_kw * dt_hours
        
        # Update SOC
        new_battery_soc = self.battery.soc + (energy_change_kwh / self.battery.capacity_kwh)
        
        # Clamp to physical limits and track violations
        violation_occurred = False
        if new_battery_soc < self.battery.min_soc:
            new_battery_soc = self.battery.min_soc
            self._state.battery_violation_count += 1
            violation_occurred = True
        if new_battery_soc > self.battery.max_soc:
            new_battery_soc = self.battery.max_soc
            violation_occurred = True
        
        self.battery.soc = new_battery_soc
        self.battery.power_kw = battery_power_kw
        
        # --- Battery Degradation ---
        # Degradation based on energy cycled (not power)
        energy_cycled_kwh = abs(energy_change_kwh)
        step_degradation = energy_cycled_kwh * self.battery.degradation_rate
        self._state.cumulative_battery_degradation += step_degradation
        
        # --- EV Charging ---
        ev_demand = self._get_ev_demand_at_step(step_idx)
        self.ev.soc = min(1.0, self.ev.soc + (ev_demand * dt_hours) / self.ev.battery_capacity_kwh)
        self.ev.power_demand_kw = ev_demand

        # ===================================================================
        # 2. REWARD CALCULATION (Multi-objective)
        # ===================================================================
        
        # A. ENERGY ARBITRAGE PROFIT
        # Grid power flow: positive = buying (cost), negative = selling (revenue)
        # grid_dispatch = battery_charge + household_load + ev_demand - solar_generation
        
        if battery_power_kw > 0:
            # Charging: buying from grid
            grid_power_from_battery = battery_power_kw / self.battery.efficiency_roundtrip  # Account for losses
        else:
            # Discharging: selling to grid
            grid_power_from_battery = battery_power_kw  # Already at grid connection
        
        grid_dispatch_kw = grid_power_from_battery + net_load + ev_demand - solar_gen
        
        # Cost/Revenue: buying is negative revenue, selling is positive revenue
        # Price is in $/MWh, convert to $/kWh
        price_per_kwh = price / 1000.0
        energy_traded_kwh = grid_dispatch_kw * dt_hours
        
        if grid_dispatch_kw > 0:
            # Net buying from grid: cost (negative reward)
            revenue = -energy_traded_kwh * price_per_kwh
        else:
            # Net selling to grid: revenue (positive reward)
            revenue = -energy_traded_kwh * price_per_kwh  # Negative dispatch means selling
        
        # B. BATTERY DEGRADATION PENALTY (only for THIS step)
        # FIX: Use incremental degradation, not cumulative!
        degradation_delta = self._state.cumulative_battery_degradation - self.previous_degradation
        battery_penalty = -degradation_delta * 1000.0  # $1000 per kWh of capacity lost
        self.previous_degradation = self._state.cumulative_battery_degradation
        
        # C. GRID STABILITY BONUS
        grid_frequency = self.grid_frequency_series[step_idx]
        frequency_bonus = 0.0
        
        if grid_frequency < 59.8:
            # Grid emergency detected
            actual_reserve = self.battery.soc
            required_reserve = action.battery_reserve_pct
            
            if actual_reserve >= required_reserve:
                # Agent maintained reserve: bonus
                frequency_bonus = 5.0
                self._state.grid_events_handled += 1
            else:
                # Agent failed to maintain reserve: penalty
                frequency_bonus = -10.0
        
        # D. CONSTRAINT VIOLATION PENALTY
        violation_penalty = -20.0 if violation_occurred else 0.0
        
        # E. TOTAL REWARD
        reward = revenue + battery_penalty + frequency_bonus + violation_penalty
        
        # Debug logging
        print(
            f"[STEP {self._step_count:3d}] "
            f"Price=${price:6.2f}/MWh | "
            f"SOC={self.battery.soc:5.1%} | "
            f"Power={battery_power_kw:+6.2f}kW | "
            f"GridFlow={grid_dispatch_kw:+6.2f}kW | "
            f"Revenue={revenue:+8.2f} | "
            f"BattPen={battery_penalty:+8.2f} | "
            f"FreqBonus={frequency_bonus:+6.2f} | "
            f"ViolPen={violation_penalty:+6.2f} | "
            f"REWARD={reward:+8.2f}",
            flush=True
        )
        
        self.cumulative_reward += reward
        self._state.cumulative_revenue_usd += revenue

        # ===================================================================
        # 3. EPISODE TERMINATION
        # ===================================================================
        
        done = (step_idx >= 95) or (self._state.battery_violation_count > 5)

        # Update class variable for grading
        if done:
            VppEnvironment._last_cumulative_reward = self.cumulative_reward
            print(f"\n[EPISODE END] Cumulative Reward: {self.cumulative_reward:.2f}", flush=True)

        self.reward = reward
        self.done = done

        return self._generate_observation()
    
    def _generate_observation(self) -> VppObservation:
        """Create current state observation."""
        step_idx = self._step_count % 96
        solar_gen = self._generate_solar_at_step(step_idx)
        
        return VppObservation(
            timestamp=datetime.now().isoformat(),
            step_id=step_idx,
            battery_telemetry=[BatteryTelemetry(
                asset_id="battery_1",
                soc=self.battery.soc,
                power_kw=self.battery.power_kw,
                degradation_cumulative=self._state.cumulative_battery_degradation,
            )],
            solar_telemetry=[SolarTelemetry(
                asset_id="solar_1",
                generation_kw=solar_gen,
            )],
            ev_telemetry=[EvTelemetry(
                asset_id="ev_1",
                soc=self.ev.soc,
                charger_available=True,
                power_demand_kw=self._get_ev_demand_at_step(step_idx),
            )],
            grid_frequency_hz=self.grid_frequency_series[step_idx],
            grid_voltage_v=120.0,
            market_price_per_mwh=self.price_series[step_idx],
            forecast_next_24h_price=self.price_series[step_idx:] + self.price_series[:step_idx],
            forecast_next_24h_solar_kw=[self._generate_solar_at_step(i) for i in range(96)],
            reward=getattr(self, 'reward', 0.0),
            done=self.done,
            cumulative_reward=self.cumulative_reward,
        )
    
    @property
    def state(self) -> VppState:
        """Return episode metadata."""
        self._state.step_count = self._step_count
        return self._state
    
    @classmethod
    def get_current_task_score(cls) -> float:
        """Grader: Normalize cumulative reward to 0.0-1.0."""
        # Adjust normalization based on task difficulty
        # These are estimated maximum achievable rewards
        max_rewards = {
            "easy-arbitrage": 500.0,
            "medium-forecast-error": 300.0,
            "hard-frequency-response": 200.0
        }
        
        # Default normalization
        max_reward = 500.0
        
        if cls._last_cumulative_reward is None:
            return 0.0
        
        # Normalize: score = reward / max_possible_reward
        score = cls._last_cumulative_reward / max_reward
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, score))
    
    # =======================================================================
    # HELPER METHODS
    # =======================================================================
    
    def _load_prices(self, filename: str) -> List[float]:
        """Load market prices from CSV file."""
        filepath = self.data_dir / filename
        prices = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prices.append(float(row['price_per_mwh']))
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            # Fallback to synthetic data with variation
            prices = [50.0 + 10.0 * np.sin(2 * np.pi * i / 96) for i in range(96)]
        
        if len(prices) != 96:
            print(f"Warning: {filename} has {len(prices)} prices, padding/truncating to 96")
            prices = (prices * 96)[:96]  # Repeat or truncate
        
        return prices
    
    def _generate_solar_at_step(self, step_idx: int) -> float:
        """Generate solar generation using bell curve."""
        hour = step_idx / 4.0  # Convert 15-min steps to hours
        
        # Bell curve: peak at 12h (noon)
        if hour < 6 or hour > 18:
            return 0.0
        
        # Cosine-squared gives smooth bell curve
        base = 5.0 * max(0.0, np.cos(np.pi * (hour - 12.0) / 12.0) ** 2)
        
        # Apply cloud cover
        return max(0.0, base * (1.0 - self.solar_cloud_cover))
    
    def _get_ev_demand_at_step(self, step_idx: int) -> float:
        """EV charging demand pattern."""
        # Normal charging window: 6 PM (step 72) to midnight (step 95)
        if 72 <= step_idx <= 95:
            return 0.2  # 0.2 kW average over window
        
        # Hard task: Emergency charge at step 50
        elif self._state.task_tier == "hard-frequency-response" and 50 <= step_idx <= 60:
            return 0.4
        
        return 0.0
    
    def _generate_grid_frequency(self, volatility: float) -> List[float]:
        """Generate 96-step grid frequency profile."""
        freq = [60.0] * 96
        
        # Add emergency dip at steps 60-70 (only in hard task, but generate for all)
        for i in range(60, 70):
            freq[i] = 59.5 + 0.5 * np.sin(np.pi * (i - 60) / 10.0)
        
        # Add noise
        for i in range(96):
            freq[i] += np.random.normal(0, volatility * 0.1)
        
        return freq