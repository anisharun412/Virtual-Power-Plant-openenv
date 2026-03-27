---
title: Vpp Environment Server
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Vpp Environment

A Virtual Power Plant (VPP) environment for training and evaluating AI agents in energy management.

## Quick Start

The simplest way to use the Vpp environment is through the `VppEnv` class:

```python
from vpp import VppAction, VppEnv

try:
    # Create environment from Docker image
    vppenv = VppEnv.from_docker_image("vpp-env:latest")

    # Reset with a specific task
    result = vppenv.reset(task_id="easy-arbitrage")
    print(f"Episode Started: {result.observation.timestamp}")

    # Run for a few steps
    for i in range(5):
        # Decide action (e.g., charge at 50% rate, 20% min reserve)
        action = VppAction(global_charge_rate=0.5, min_reserve_pct=0.2)
        result = vppenv.step(action)
        
        obs = result.observation
        print(f"Step {obs.step_id} | Price: {obs.market_price_per_mwh:.2f} | Reward: {result.reward:.2f}")

finally:
    # Always clean up
    vppenv.close()
```

That's it! The `VppEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Tasks

The environment provides different tasks with varying difficulty levels based on the `openenv.yaml` spec:

- `easy-arbitrage`: Standard energy trading and arbitrage
- `medium-forecast-error`: Handling weather mismatch and forecast uncertainty
- `hard-frequency-response`: Emergency grid support operations

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t vpp-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**VppAction**: Controls the charging/discharging behavior
- `global_charge_rate` (float) - The rate to charge or discharge (-1.0 to 1.0). Positive values buy energy from the grid (charge), negative values sell energy to the grid (discharge).
- `min_reserve_pct` (float) - Minimum reserve energy percentage (0.0 to 1.0).

### Observation
**VppObservation**: Contains the state of the Virtual Power Plant and market
- `timestamp` (datetime) - Current time in the simulation
- `step_id` (int) - The current step number within the episode
- `telemetry` (List[BatteryTelemetry]) - Telemetry data for each battery asset (e.g., `asset_id`, `soc`, `current_house_load_kw`, `current_solar_gen_kw`)
- `market_price_per_mwh` (float) - Current market price of energy
- `forecast_24h_price` (List[float]) - Forecasted market prices for the next 24 hours
- `forecast_24h_solar` (List[float]) - Forecasted solar generation for the next 24 hours

### Reward
The reward calculates the profit achieved during the step:
- Reward = Power Sold × Price
- Negative `global_charge_rate` implies selling power to the grid, resulting in a positive reward proportional to the current market price.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Vpp environment server running, you can connect directly:

```python
from vpp import VppEnv, VppAction

# Connect to existing server
vppenv = VppEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = vppenv.reset(task_id="easy-arbitrage")
result = vppenv.step(VppAction(global_charge_rate=-0.5, min_reserve_pct=0.2))
```

Note: When connecting to an existing server, `vppenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from vpp import VppAction, VppEnv

# Connect with context manager (auto-connects and closes)
with VppEnv(base_url="http://localhost:7860") as env:
    result = env.reset(task_id="medium-forecast-error")
    print(f"Reset step ID: {result.observation.step_id}")
    
    # Multiple steps with low latency
    for i in range(3):
        result = env.step(VppAction(global_charge_rate=0.8, min_reserve_pct=0.2))
        print(f"Reward: {result.reward:.2f}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    VppEnvironment,  # Pass class, not instance
    VppAction,
    VppObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from vpp import VppAction, VppEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with VppEnv(base_url="http://localhost:7860") as env:
        result = env.reset(task_id="easy-arbitrage")
        for i in range(10):
            result = env.step(VppAction(global_charge_rate=0.5, min_reserve_pct=0.2))
        return client_id, result.reward

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Baseline Inference

A functional LLM trading baseline is provided in `baseline_inference.py`. You can run it locally:

```bash
# Terminal 1 - Start the environment server
uvicorn vpp.server.app:app --reload

# Terminal 2 - Run the baseline and connect to the local server
export OPENAI_API_KEY="your-api-key"
export VPP_SERVER_URL="http://localhost:8000"
python vpp/baseline_inference.py
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/vpp_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```text
vpp/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # VppEnv client
├── models.py              # Action and Observation models
├── baseline_inference.py  # LLM baseline execution
└── server/
    ├── __init__.py        # Server module exports
    ├── vpp_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
