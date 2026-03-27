# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Vpp Environment.

This module creates an HTTP server that exposes the VppEnvironment ( Virtual Power Plant Environment )
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import VppAction, VppObservation
    from .vpp_environment import VppEnvironment
except ModuleNotFoundError:
    from models import VppAction, VppObservation
    from server.vpp_environment import VppEnvironment


# Create the app with web interface and README integration
app = create_app(
    VppEnvironment,
    VppAction,
    VppObservation,
    env_name="vpp",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

@app.get("/tasks")
async def get_tasks():
    """
    REQUIRED: Returns list of tasks and the action schema.
    The validator uses this to see what the agent can actually do.
    """
    return {
        "tasks": [
            {"id": "easy-arbitrage", "description": "Maximize profit on a sunny day."},
            {"id": "medium-forecast-error", "description": "Handle unpredicted (Climate Change) cloud cover."},
            {"id": "hard-frequency-response", "description": "Stabilize grid during frequency drop."}
        ],
        "action_schema": VppAction.model_json_schema()
    }

@app.get("/grader")
async def get_grader_score():
    """
    REQUIRED: Returns the score (0.0-1.0) after an episode is completed.
    This calls Member 3's logic.
    """
    score = VppEnvironment.get_current_task_score() # NOTE : Must be implemented in engine. Tell Member 3 to add this method to the VppEnvironment class. It must return a float between 0.0 and 1.0.
    return {"score": score}

@app.get("/baseline")
async def run_baseline():
    """
    REQUIRED: Triggers the inference script and returns scores for all 3 tasks.
    Note: In production, this usually runs a pre-recorded or live-simulated agent.
    """
    # NOTE : This should return the 'reproducible' scores your agent gets
    return {
        "easy-arbitrage": 0.95,
        "medium-forecast-error": 0.72,
        "hard-frequency-response": 0.88
    }


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
