---
title: VPP RL Environment
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - energy
  - rl
---

# Virtual Power Plant (VPP) RL Environment

A high-fidelity energy trading simulator for renewable asset portfolio optimization. This environment implements the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification for reproducible and scalable RL benchmarking.

## Overview

**Task**: Maximize profit while maintaining grid stability by controlling a portfolio of distributed energy resources (batteries, solar, EVs) across a 24-hour period (96 × 15-minute timesteps).

**Assets**:
- **Batteries**: 10 kWh capacity, 5 kW max power, 92% round-trip efficiency
- **Solar**: 5 kW peak, weather-dependent generation
- **EV**: 60 kWh capacity, 7 kW home charger

**Action Space**:
- `global_charge_rate`: [-1.0, 1.0] (sell all → buy all)
- `battery_reserve_pct`: [0.0, 1.0] (grid support buffer)

**Observation Space**:
- Asset telemetry (SoC, power, generation)
- Grid conditions (frequency, voltage, market price)
- 24-hour forecasts (prices, solar generation)

## Quick Start

### Python API

```python
from vpp import VppAction, VppEnv
import asyncio

async def main():
    # Create environment from Docker image
    async with VppEnv.from_docker_image("vpp-env:latest") as env:
        # Reset with a specific task
        result = await env.reset(task_id="easy-arbitrage")
        print(f"Episode Started: {result.observation.step_id}")

        # Run for multiple steps
        for step in range(96):
            # Decide action: charge at 50% rate, maintain 20% min reserve
            action = VppAction(global_charge_rate=0.5, battery_reserve_pct=0.2)
            result = await env.step(action)
            
            if result.done:
                break
                
            obs = result.observation
            price = obs.market_price_per_mwh
            soc = obs.battery_telemetry[0].soc if obs.battery_telemetry else 0
            print(f"Step {obs.step_id} | Price: ${price:.2f}/MWh | SoC: {soc:.1%} | Reward: {result.reward:+.2f}")

asyncio.run(main())
```

## Installation

### From Source

```bash
# Clone repository
cd vpp

# Install with pip
pip install -e .

# Verify installation
python validate.py
```

### Docker

```bash
# Build Docker image
docker build -t vpp-env:latest .

# Run container
docker run -p 8000:7860 vpp-env:latest

# Test in another terminal
curl http://localhost:8000/tasks
```

## Tasks

The environment provides 3 tasks with increasing difficulty:

| Task | Difficulty | Scenario | Market | Solar | Grid |
|------|-----------|----------|--------|-------|------|
| `easy-arbitrage` | ⭐ | Stable trading | ±$2/MWh | Predictable | Normal |
| `medium-forecast-error` | ⭐⭐ | Cloud uncertainty | ±$13/MWh | Variable | Varied freq |
| `hard-frequency-response` | ⭐⭐⭐ | Grid emergency | ±$50/MWh | Intermittent | 59.5 Hz dip |

## API Reference

### Client Usage

```python
from vpp import VppEnv, VppAction

# Async with auto Docker container management
async with VppEnv.from_docker_image("vpp-env:latest") as env:
    await env.reset(task_id="easy-arbitrage")
    action = VppAction(global_charge_rate=-0.5, battery_reserve_pct=0.3)
    result = await env.step(action)

# Sync wrapper
with VppEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(task_id="easy-arbitrage")
    result = env.step(VppAction(global_charge_rate=0.0, battery_reserve_pct=0.2))
```

### HTTP Endpoints

**Available Endpoints**:
- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get episode metadata
- `GET /schema` - Action/observation JSON schemas
- `GET /tasks` - List tasks
- `GET /grader` - Get normalized score (0.0-1.0)
- `WS /ws` - WebSocket for persistent sessions

**Example**:
```bash
# Get tasks
curl http://localhost:8000/tasks

# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy-arbitrage"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"global_charge_rate": 0.5, "battery_reserve_pct": 0.2}'
```

## Baseline Inference

Run the OpenAI-based baseline agent:

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export VPP_SERVER_URL="http://localhost:8000"

python inference.py
```

**Output Format** (JSON logs):
```jsonl
{"type": "START", "task": "easy-arbitrage", "env": "vpp", "model": "gpt-4o-mini"}
{"type": "STEP", "step": 1, "action": {"global_charge_rate": 0.5, ...}, "reward": -5.25, "done": false}
{"type": "END", "success": true, "steps": 96, "score": 0.5234, ...}
```

## Environment Details

### Action Space

**VppAction**:
- `global_charge_rate: float` - Portfolio control [-1.0 (sell max), +1.0 (buy max)]
- `battery_reserve_pct: float` - Grid support buffer [0.0, 1.0]

### Observation Space

**VppObservation**:
- `timestamp: str` - ISO 8601 timestamp
- `step_id: int` - Current step (0-95)
- `battery_telemetry: List[BatteryTelemetry]` - Asset-level metrics (SoC, power, degradation)
- `solar_telemetry: List[SolarTelemetry]` - Generation status
- `ev_telemetry: List[EvTelemetry]` - EV charging status
- `grid_frequency_hz: float` - Grid frequency (59-61 Hz)
- `grid_voltage_v: float` - Voltage (100-140 V)
- `market_price_per_mwh: float` - Current market price ($/MWh)
- `forecast_next_24h_price: List[float]` - 96-value price forecast
- `forecast_next_24h_solar_kw: List[float]` - 96-value generation forecast

### Physics Model

**Battery SoC Update** (per 15-min step):
```
power_flow_kw = global_charge_rate × 5.0 kW
energy_kwh = power_flow_kw × 0.25 hours × 0.92 efficiency  
new_soc = clamp(old_soc + energy_kwh / 10.0 capacity, 0.1, 0.95)
```

**Degradation**:
```
degradation_per_step = |power_flow_kw| × 0.25 hours × 0.0005 rate
cumulative_degradation += degradation_per_step
```

**Solar Generation** (bell curve):
```
base_kw = 5.0 × max(0, cos(π × (hour - 12) / 12)²)
generation = base_kw × (1 - cloud_cover) for hours in [6, 18]
```

### Reward Function

```
profit = -grid_dispatch_power × market_price × dt
battery_penalty = -cumulative_degradation × 100  
grid_bonus = +5 for maintaining reserve during frequency dip
            -10 for failing to maintain reserve

total_reward = profit + battery_penalty + grid_bonus
```

## Reproducibility

All randomness is seed-controlled:

```python
# Deterministic behavior with seed
result = env.reset(task_id="easy-arbitrage", seed=42)
```

Market data is CSV-based (deterministic, 96 15-min intervals):
- `server/data/prices_easy.csv`
- `server/data/prices_medium.csv`  
- `server/data/prices_hard.csv`

## Validation

Run the comprehensive validation suite:

```bash
python validate.py
```

**Checks**:
- ✓ Imports and dependencies
- ✓ Pydantic model validation
- ✓ Environment initialization
- ✓ Task configuration
- ✓ Market data loading
- ✓ Full episode execution
- ✓ Client serialization
- ✓ FastAPI endpoints

## Development

### File Structure

```
vpp/
├── models.py              # Pydantic types
├── client.py              # HTTP/WebSocket client
├── __init__.py            # Package exports
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── inference.py           # Baseline agent
├── validate.py            # Test suite
├── Dockerfile             # Container build
├── README.md              # Documentation
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI server
│   ├── vpp_environment.py # Core simulator
│   ├── asset_models.py    # Physical models
│   └── data/
│       ├── prices_easy.csv
│       ├── prices_medium.csv
│       └── prices_hard.csv
└── requirements.txt
```

### Key Classes

**VppEnvironment** (OpenEnv Environment):
- `reset(task_id, seed)` - Initialize 96-step episode
- `step(action)` - Execute one timestep
- `state` - Episode metadata (hidden)
- `get_current_task_score()` - Normalized reward [0.0, 1.0]

**Asset Models**:
- `BatteryAsset` - 10 kWh, 5 kW, 92% efficiency
- `SolarAsset` - 5 kW peak
- `EvAsset` - 60 kWh, 7 kW charger

## Deployment

### HuggingFace Spaces

```bash
# Login
huggingface-cli login

# Deploy (requires openenv-cli)
openenv push --repo-id username/vpp --private
```

Space will be accessible at: `https://huggingface.co/spaces/username/vpp`

### Azure Container Registry

```bash
docker build -t vpp-env:latest .
docker tag vpp-env:latest myacr.azurecr.io/vpp:latest
docker push myacr.azurecr.io/vpp:latest
```

## Troubleshooting

### Import Errors

```bash
# Verify installation
python -c "from vpp import VppEnv; print('OK')"

# Run validation
python validate.py
```

### Connection Issues

```bash
# Check server status
curl http://localhost:8000/tasks

# Run with verbose
docker logs <container_id>
```

### Reproducibility

Always use seed for deterministic results:

```python
result = env.reset(task_id="easy-arbitrage", seed=123)
```

## Performance

| Metric | Target |
|--------|--------|
| Reset time | <100 ms |
| Step time | <50 ms |
| Memory per session | <100 MB |
| Max concurrent | 10+ |
| Server startup | <5 s |

## Citation

```bibtex
@software{vpp_openenv_2024,
  title={Virtual Power Plant RL Environment},
  url={https://github.com/meta-pytorch/openenv},
  year={2024}
}
```

## License

BSD-3-Clause License - See LICENSE file

---

**Status**: ✓ Production Ready  
**Version**: 0.1.0  
**Last Updated**: 2024

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
