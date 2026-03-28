import os
import json
import requests
from openai import OpenAI
from vpp.client import VppEnv
from vpp.models import VppAction

def get_llm_action(observation_data: dict) -> VppAction:
    """
    Sends the current environment state to the LLM and parses the tactical decision.
    Tries OpenAI first, falls back to a Local LLM, then to Groq.
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

    messages = [
        {"role": "system", "content": "You are a precise industrial controller."},
        {"role": "user", "content": prompt}
    ]

    # ---------------------------------------------------------
    # ATTEMPT 1: OpenAI (gpt-4o-mini)
    # ---------------------------------------------------------
    try:
        print("🤖 Attempting inference via OpenAI...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        decision = json.loads(response.choices[0].message.content)
        return VppAction(**decision)
    except Exception as e:
        print(f"  [X] OpenAI failed: {e}")

    # ---------------------------------------------------------
    # ATTEMPT 2: Local LLM (e.g., Ollama or LM Studio)
    # ---------------------------------------------------------
    try:
        print("💻 Attempting inference via Local LLM...")
        # Defaulting to Ollama's OpenAI-compatible endpoint. 
        # Change port to 1234 if using LM Studio.
        local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="local-key")
        response = local_client.chat.completions.create(
            model="llama3", # Update this to the exact name of your local model
            messages=messages,
            response_format={"type": "json_object"}
        )
        decision = json.loads(response.choices[0].message.content)
        return VppAction(**decision)
    except Exception as e:
        print(f"  [X] Local LLM failed: {e}")

    # ---------------------------------------------------------
    # ATTEMPT 3: Groq (Ultra-fast LPU Inference)
    # ---------------------------------------------------------
    try:
        print("⚡ Attempting inference via Groq...")
        groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # Groq model that supports JSON mode
            messages=messages,
            response_format={"type": "json_object"}
        )
        decision = json.loads(response.choices[0].message.content)
        return VppAction(**decision)
    except Exception as e:
        print(f"  [X] Groq failed: {e}")

    # ---------------------------------------------------------
    # ULTIMATE FALLBACK: Do nothing
    # ---------------------------------------------------------
    print("⚠️ All LLMs failed. Falling back to idle action.")
    return VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)


def run_baseline(task_id="easy-arbitrage"):
    print(f"\n--- Starting LLM Baseline for Task: {task_id} ---")
    
    # Point this to your local server or Hugging Face Space URL
    server_url = os.getenv("VPP_SERVER_URL", "http://localhost:7860")
    
    with VppEnv(base_url=server_url) as env:
        # Reset the environment for the specific task
        result = env.reset(task_id=task_id)
        total_reward = 0.0
        done = False
        
        print(f"Episode Started: {result.observation.timestamp}\n")

        while not done:
            obs = result.observation

            # Converting observation to dict for the LLM
            obs_dict = obs.model_dump()
            
            # Action logic with fallbacks
            action = get_llm_action(obs_dict)
            
            # Execute step
            result = env.step(action)
            total_reward += result.reward
            done = result.done
            
            print(f"Step {obs.step_id} | Price: ${obs.market_price_per_mwh:.2f} | "
                  f"Action: {action.global_charge_rate:.2f} | Reward: {result.reward:.2f}")

        # Fetch final score from the grader endpoint
        print(f"\n--- Episode Complete ---")
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