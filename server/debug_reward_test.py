import os
import sys

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.vpp_environment import VppEnvironment
from models import VppAction


def main():
    env = VppEnvironment()
    obs = env.reset(task_id="easy-arbitrage", seed=0)
    action = VppAction(global_charge_rate=0.0, battery_reserve_pct=0.2)
    for i in range(3):
        obs, r, d, info = env.step(action)
        print(f"STEP_TEST {i+1}: reward={r:.6f} done={d} info={info}")


if __name__ == "__main__":
    main()
