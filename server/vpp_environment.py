# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Vpp Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from datetime import datetime
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import VppAction, VppObservation, VppState, BatteryAsset, BatteryTelemetry
except ImportError:
    from models import VppAction, VppObservation, VppState, BatteryAsset, BatteryTelemetry

"""
NOTE : THIS IS A PLACEHOLDER ENGINE
       TO BE IMPLEMENTED BY MEMBER 2 & 3
"""
class VppEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    NOTE : THIS IS A PLACEHOLDER ENGINE. TO BE IMPLEMENTED BY MEMBER 2 & 3

    Example:
        >>> env = VppEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Vpp environment ready!"
        >>>
        >>> obs = env.step(VppAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """
        Initialize the VppEnvironment.
        """
        self._reset_count = 0
        self._last_cumulative_reward = 0.0

        self.assets = [
            BatteryAsset(asset_id=f"home-{i:03}", capacity_kwh=13.5, max_power_kw=5.0)
            for i in range(10) # Start with 10 for testing, scale to 100 later
        ]
        self._internal_state = None

    def reset(self, task_id: str = "easy-arbitrage") -> VppObservation:
        """
        Reset the environment.

        Args:
            task_id: The ID of the task to reset

        Returns:
            VppObservation: The initial observation after reset.
        """
        self._reset_count += 1

        # Initialize God-Mode State
        self._internal_state = VppState(
            episode_id=str(uuid4()),
            step_count=0,
            current_step=0,
            task_tier=task_id,
            actual_weather_mode="clear_sky",
            battery_true_soc={a.asset_id: 0.5 for a in self.assets} # Start half full
        )
        return self._generate_observation()

    def step(self, action: VppAction) -> tuple[VppObservation, float, bool, dict]:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: VppAction containing the action to take

        Returns:
            tuple[VppObservation, float, bool, dict]:
            The observation after the action is taken,
            the reward, whether the episode is done,
            and a dictionary of extra info 
        """
        self._internal_state.step_count += 1
        self._internal_state.current_step += 1
        
        reward = 0.0
        # 1. Physics Engine: Update Battery SoC
        for asset in self.assets:
            old_soc = self._internal_state.battery_true_soc[asset.asset_id]
            
            # Simple Charge/Discharge logic
            # Change in energy = Power * Time (15 mins = 0.25h) * Efficiency
            power_flow = action.global_charge_rate * asset.max_power_kw
            energy_change = power_flow * 0.25 * asset.efficiency_rt
            
            new_soc = old_soc + (energy_change / asset.capacity_kwh)
            self._internal_state.battery_true_soc[asset.asset_id] = max(0.0, min(1.0, new_soc))

        # 2. Reward Logic: Profit = (Power Sold * Price)
        price = self._get_current_price()
        reward = action.global_charge_rate * price * -1.0 # Negative charge = Selling = Profit
        
        done = self._internal_state.current_step >= 95 # End of 24h cycle

        if done:
            # Store the final theoretical calculation here. 
            # NOTE : For now, we are just storing the last reward step as a placeholder.
            VppEnvironment._last_cumulative_reward += reward
        return self._generate_observation(), reward, done, {}

    def _generate_observation(self) -> VppObservation:
        telemetry = [
            BatteryTelemetry(
                asset_id=a.asset_id, 
                soc=self._internal_state.battery_true_soc[a.asset_id],
                current_house_load_kw=1.2,
                current_solar_gen_kw=3.5
            ) for a in self.assets
        ]
        return VppObservation(
            timestamp=datetime.now(),
            step_id=self._internal_state.current_step,
            telemetry=telemetry,
            market_price_per_mwh=self._get_current_price(),
            forecast_24h_price=[self._get_current_price()] * 96,
            forecast_24h_solar=[5.0] * 96
        )

    def _get_current_price(self):
        return 50.0 + random.uniform(-10, 10) # Mock price logic

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._internal_state

    @classmethod
    def get_current_task_score(cls) -> float:
        """
        Calculates a normalized score (0.0 to 1.0) based on performance.
        """
        # NOTE: This is placeholder logic! 
        # In a real VPP, you would divide the actual profit by the theoretical max profit.
        # For now, we return a mock normalized score based on the reward being positive.
        if cls._last_cumulative_reward <= 0:
            return 0.0
        
        # Arbitrary normalization for testing: cap at 1.0
        normalized = min(1.0, cls._last_cumulative_reward / 1000.0) 
        return normalized
