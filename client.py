# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vpp Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import VppAction, VppObservation, BatteryTelemetry


class VppEnv(
    EnvClient[VppAction, VppObservation, State]
):
    """
    Client for the Vpp Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with VppEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(VppAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = VppEnv.from_docker_image("vpp-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(VppAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: VppAction) -> Dict:
        # Send the VPP-specific action fields

        return {
            "global_charge_rate": action.global_charge_rate,
            "min_reserve_pct": action.min_reserve_pct
        }

    def _parse_result(self, payload: Dict) -> StepResult[VppObservation]:
        obs_data = payload.get("observation", {})
        
        # Reconstruct the telemetry list from JSON
        telemetry = [BatteryTelemetry(**t) for t in obs_data.get("telemetry", [])]
        
        observation = VppObservation(
            timestamp=obs_data.get("timestamp"),
            step_id=obs_data.get("step_id", 0),
            telemetry=telemetry,
            grid_frequency_hz=obs_data.get("grid_frequency_hz", 50.0),
            grid_voltage_v=obs_data.get("grid_voltage_v", 230.0),
            market_price_per_mwh=obs_data.get("market_price_per_mwh", 0.0),
            forecast_24h_price=obs_data.get("forecast_24h_price", []),
            forecast_24h_solar=obs_data.get("forecast_24h_solar", [])
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
