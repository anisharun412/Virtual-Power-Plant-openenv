# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the VPP Environment.
Exposes VppEnvironment over HTTP and WebSocket endpoints.
"""

from openenv.core.env_server.http_server import create_app

try:
    from ..models import VppAction, VppObservation, VppState
    from .vpp_environment import VppEnvironment
except ImportError:
    from models import VppAction, VppObservation, VppState
    from server.vpp_environment import VppEnvironment


# Create the FastAPI app with OpenEnv HTTP server
app = create_app(
    VppEnvironment,
    VppAction,
    VppObservation,
    env_name="vpp",
    max_concurrent_envs=10,  # Support up to 10 concurrent WebSocket sessions
)


@app.get("/tasks")
async def get_tasks():
    """Return available tasks and action schema for validation."""
    return {
        "tasks": [
            {
                "id": "easy-arbitrage",
                "description": "Stable market prices, predictable solar generation",
                "difficulty": 1,
            },
            {
                "id": "medium-forecast-error",
                "description": "Normal market volatility, cloud cover uncertainty",
                "difficulty": 2,
            },
            {
                "id": "hard-frequency-response",
                "description": "Extreme price spikes, grid emergency frequency response required",
                "difficulty": 3,
            },
        ],
        "action_schema": VppAction.model_json_schema(),
        "observation_schema": VppObservation.model_json_schema(),
    }


@app.get("/grader")
async def get_grader_score():
    """Return normalized score (0.0-1.0) after episode completion."""
    score = VppEnvironment.get_current_task_score()
    return {
        "score": score,
        "info": "Normalized cumulative reward to 0.0-1.0 scale",
    }


# def main():
#     """Entry point for direct execution."""
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# if __name__ == "__main__":
#     main()


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m vpp.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn vpp.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
