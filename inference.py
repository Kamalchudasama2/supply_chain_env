import asyncio
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_URL = os.getenv("ENV_URL") or "https://kamalchudasama-suplly-chain-env.hf.space"

# OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


# LLM ACTION (JSON PARSING)
def get_action(obs):
    prompt = f"""
You are managing inventory in a supply chain.

Current state:
Inventory: {obs['inventory']}
Last demand: {obs.get('last_demand')}

Decide order_quantity (integer >= 0).

Respond ONLY in JSON:
{{"order_quantity": number}}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a supply chain optimization agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )

        text = (completion.choices[0].message.content or "").strip()

        try:
            action = json.loads(text)
        except:
            action = {"order_quantity": 0}

        return action

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        return {"order_quantity": 0}


# MAIN LOOP
async def run():
    task = "easy"
    MAX_STEPS = 10
    MAX_TOTAL_REWARD = MAX_STEPS * 1.0

    rewards = []
    steps = 0
    success = False
    score = 0.0

    log_start(task, "supply_chain", MODEL_NAME)

    try:
        
        try:
            response = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=10)
            response.raise_for_status()
            r = response.json()
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}", flush=True)
            return

        
        for step in range(1, MAX_STEPS + 1):
            obs = r.get("observation", {})

            action = get_action(obs)

            try:
                response = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
                response.raise_for_status()
                r = response.json()
            except Exception as e:
                print(f"[ERROR] Step failed: {e}", flush=True)
                break

            reward = r.get("reward", 0.0)
            done = r.get("done", True)

            rewards.append(reward)
            steps = step

            log_step(
                step=step,
                action=str(action),
                reward=reward,
                done=done,
                error=None
            )

            if done:
                break

        
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    finally:
        
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(run())
