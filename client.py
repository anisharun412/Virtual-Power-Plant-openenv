# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VPP Environment Client Implementation."""

from typing import Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        VppAction,
        VppObservation,
        VppState,
        BatteryTelemetry,
        SolarTelemetry,
        EvTelemetry,
    )
except ImportError:
    from models import (
        VppAction,
        VppObservation,
        VppState,
        BatteryTelemetry,
        SolarTelemetry,
        EvTelemetry,
    )


class VppEnv(EnvClient[VppAction, VppObservation, VppState]):
    """Client for VPP Environment.
    
    Connects to the VPP server via HTTP/WebSocket and handles
    serialization/deserialization of typed action/observation models.
    
    Example (Sync):
        >>> with VppEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id="easy-arbitrage")
        ...     for step in range(96):
        ...         action = VppAction(global_charge_rate=0.5, battery_reserve_pct=0.2)
        ...         result = env.step(action)
        ...         if result.done:
        ...             break

    Example (Async):
        >>> async with VppEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="easy-arbitrage")
        ...     result = await env.step(action)
    """

    def _step_payload(self, action: VppAction) -> Dict:
        """Convert typed action to JSON payload."""
        return {
            "global_charge_rate": action.global_charge_rate,
            "battery_reserve_pct": action.battery_reserve_pct,
        }

    def _parse_result(self, payload: Dict) -> StepResult[VppObservation]:
        """Parse server JSON response to typed Observation."""
        obs_data = payload.get("observation", {})

        # Reconstruct nested telemetry models
        battery_telem = [
            BatteryTelemetry(**t) for t in obs_data.get("battery_telemetry", [])
        ]
        solar_telem = [
            SolarTelemetry(**t) for t in obs_data.get("solar_telemetry", [])
        ]
        ev_telem = [EvTelemetry(**t) for t in obs_data.get("ev_telemetry", [])]

        observation = VppObservation(
            timestamp=obs_data.get("timestamp", ""),
            step_id=obs_data.get("step_id", 0),
            battery_telemetry=battery_telem,
            solar_telemetry=solar_telem,
            ev_telemetry=ev_telem,
            grid_frequency_hz=obs_data.get("grid_frequency_hz", 60.0),
            grid_voltage_v=obs_data.get("grid_voltage_v", 120.0),
            market_price_per_mwh=obs_data.get("market_price_per_mwh", 0.0),
            forecast_next_24h_price=obs_data.get("forecast_next_24h_price", []),
            forecast_next_24h_solar_kw=obs_data.get("forecast_next_24h_solar_kw", []),
            # Reward and termination info (from step payload or embedded observation)
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
            cumulative_reward=payload.get("info", {}).get("cumulative_reward", obs_data.get("cumulative_reward", 0.0)),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> VppState:
        """Parse server JSON response to typed State."""
        return VppState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_tier=payload.get("task_tier", "easy-arbitrage"),
            cumulative_revenue_usd=payload.get("cumulative_revenue_usd", 0.0),
            cumulative_battery_degradation=payload.get(
                "cumulative_battery_degradation", 0.0
            ),
            grid_events_handled=payload.get("grid_events_handled", 0),
            battery_violation_count=payload.get("battery_violation_count", 0),
        )
