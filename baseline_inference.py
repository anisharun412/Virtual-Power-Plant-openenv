import os
import json
from openai import OpenAI
from vpp.client import VppEnv
from vpp.models import VppAction
import requests

# Initialize the OpenAI client
# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_action(observation_data: dict) -> VppAction:
    """
    Sends the current environment state to the LLM and parses the tactical decision.
    """
    prompt = f"""
    You are an AI Energy Trader managing a Virtual Power Plant (VPP).
    Your goal is to maximize profit while maintaining battery health and grid stability.
    
    CURRENT OBSERVATION:
    {json.dumps(observation_data, indent=2, default=str)}
    
    TASK:
    Decide the 'global_charge_rate' (-1.0 to 1.0) and 'min_reserve_pct' (0.0 to 1.0).
    - Positive charge rate: Buying energy from the grid (cost).
    - Negative charge rate: Selling energy to the grid (profit).
    - Higher prices favor selling; lower prices favor charging.
    
    Return ONLY a JSON object matching this schema:
    {{
        "global_charge_rate": float,
        "min_reserve_pct": float
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a precise industrial controller."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    decision = json.loads(response.choices[0].message.content)
    return VppAction(**decision)

def run_baseline(task_id="easy-arbitrage"):
    print(f"--- Starting LLM Baseline for Task: {task_id} ---")
    
    # Point this to your local server or Hugging Face Space URL
    server_url = os.getenv("VPP_SERVER_URL", "http://localhost:8000")
    
    with VppEnv(base_url=server_url) as env:
        # 1. Reset the environment for the specific task
        result = env.reset(task_id=task_id)
        total_reward = 0.0
        done = False
        
        print(f"Episode Started: {result.observation.timestamp}")

        while not done:
            obs = result.observation
            
            # 2. Convert observation to dict for the LLM
            obs_dict = obs.model_dump()
            
            # 3. Get decision from OpenAI
            try:
                action = get_llm_action(obs_dict)
            except Exception as e:
                print(f"LLM Error: {e}. Falling back to idle.")
                action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
            
            # 4. Execute step
            result = env.step(action)
            total_reward += result.reward
            done = result.done
            
            print(f"Step {obs.step_id} | Price: {obs.market_price_per_mwh:.2f} | "
                  f"Action: {action.global_charge_rate:.2f} | Reward: {result.reward:.2f}")

        # 5. Fetch final score from the grader endpoint
        print(f"--- Episode Complete ---")
        print(f"Cumulative Reward: {total_reward:.2f}")

        try:
            # Ask the server for the normalized score
            grader_response = requests.get(f"{server_url}/grader")
            final_score = grader_response.json().get("score", 0.0)
            print(f"Normalized Grader Score: {final_score:.2f} / 1.0")
            return final_score
        except Exception as e:
            print(f"Error fetching grader score: {e}")
            return 0.0

if __name__ == "__main__":
    # Ensure uvicorn server.app:app is running before execution
    run_baseline(task_id="easy-arbitrage")