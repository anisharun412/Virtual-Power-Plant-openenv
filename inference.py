#!/usr/bin/env python3
"""
VPP Baseline Inference Script
Uses OpenAI API to run agent against environment.
Outputs structured logs: [START], [STEP], [END]
"""

import asyncio
import os
import json
from typing import List
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = os.getenv("BENCHMARK", "vpp")
ENV_URL = os.getenv("VPP_SERVER_URL", "http://localhost:8000")

MAX_STEPS = 96  # 1 full day
MAX_TOTAL_REWARD = 100.0
SUCCESS_SCORE_THRESHOLD = 0.6


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    """Log step execution."""
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end."""
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}",
        flush=True,
    )

import re

def safe_parse_json(text) -> dict:
    if not text:
        return {}

    # Remove markdown blocks
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Extract JSON using regex
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {}

    json_str = match.group()

    try:
        return json.loads(json_str)
    except:
        return {}

def get_model_action(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    history: List[str],
) -> dict:
    """Query LLM for action."""
    battery_soc = 0.5
    solar_gen = 0.0
    price = 50.0
    freq = 60.0

    if observation.get("battery_telemetry"):
        battery_soc = observation["battery_telemetry"][0].get("soc", 0.5)
    if observation.get("solar_telemetry"):
        solar_gen = observation["solar_telemetry"][0].get("generation_kw", 0.0)
    price = observation.get("market_price_per_mwh", 50.0)
    freq = observation.get("grid_frequency_hz", 60.0)

    prompt = f"""
You are an AI Energy Trader managing a Virtual Power Plant (VPP).
Your goal: Maximize profit while maintaining grid stability and battery health.

CURRENT STATE (Step {step}/96):
- Battery SoC: {battery_soc:.2%}
- Solar Generation: {solar_gen:.2f} kW
- Market Price: ${price:.2f}/MWh
- Grid Frequency: {freq:.1f} Hz
- Last Reward: {last_reward:+.2f}

CONSTRAINTS:
- Charge rate from -1.0 (sell all) to +1.0 (buy all)
- Must maintain battery reserve >= battery_reserve_pct
- If frequency < 59.8 Hz, increase battery_reserve_pct to support grid

DECISION:
Output ONLY a JSON object:
{{
    "global_charge_rate": <float -1.0 to 1.0>,
    "battery_reserve_pct": <float 0.0 to 1.0>
}}
"""

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": prompt}],
            temperature=0.3,
            # max_output_tokens=200,
        )
        action_str = response.output_text.strip()
        # Extract JSON
        action_dict = safe_parse_json(action_str)
        print("MODEL OUTPUT:", action_dict, flush=True)
        return action_dict
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"global_charge_rate": 0.0, "battery_reserve_pct": 0.2}


async def run_task(task_id: str) -> float:
    """Run single task and return normalized score."""
    try:
        from client import VppEnv
        from models import VppAction
    except ImportError:
        print(
            "[DEBUG] Could not import VppEnv. Make sure vpp package is installed.",
            flush=True,
        )
        return 0.0

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = VppEnv(base_url=ENV_URL)

    history: List[str] = []
    rewards: List[float] = []
    success = False
    score = 0.0
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        result = await env.reset(task_id=task_id)
        obs_data = result.observation.model_dump()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action from LLM
            action_dict = get_model_action(client, step, obs_data, last_reward, history)
            action = VppAction(**action_dict)

            # Step
            result = await env.step(action)
            obs_data = result.observation.model_dump()
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            last_reward = reward
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action_dict),
                reward=reward,
                done=done,
                error=None,
            )
            history.append(f"Step {step}: {action_dict} → {reward:+.2f}")

            if done:
                break

        # Compute score
        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(1.0, max(0.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    """Run all 3 tasks."""
    print(f"[DEBUG] Starting VPP baseline inference", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}, Server: {ENV_URL}", flush=True)

    tasks = ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]
    scores = {}

    for task_id in tasks:
        print(f"[DEBUG] Running task: {task_id}", flush=True)
        scores[task_id] = await run_task(task_id)
        print(f"[DEBUG] Task complete: {task_id} = {scores[task_id]:.4f}", flush=True)

    # Final summary
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(
        json.dumps(
            {
                "type": "FINAL",
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "average": round(avg_score, 4),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
