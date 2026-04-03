import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client import VppEnv


def main():
    env = VppEnv(base_url="http://localhost:8000")
    payload = {
        "observation": {
            "timestamp": "now",
            "step_id": 0,
            "battery_telemetry": [],
            "solar_telemetry": [],
            "ev_telemetry": [],
            "grid_frequency_hz": 60.0,
            "grid_voltage_v": 120.0,
            "market_price_per_mwh": 50.0,
            "forecast_next_24h_price": [],
            "forecast_next_24h_solar_kw": [],
        },
        "reward": 12.34,
        "done": False,
        "info": {"cumulative_reward": 123.45},
    }
    res = env._parse_result(payload)
    print("parsed reward:", res.reward)
    print("parsed done:", res.done)
    print("obs.reward:", getattr(res.observation, 'reward', None))
    print("obs.cumulative_reward:", getattr(res.observation, 'cumulative_reward', None))


if __name__ == '__main__':
    main()
