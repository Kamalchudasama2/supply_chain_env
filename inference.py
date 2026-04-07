import asyncio
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}")


def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}")


def get_action(obs):
    inventory = obs["inventory"]
    return {"order_quantity": max(0, 100 - inventory)}


async def run():
    task = "easy"
    log_start(task, "supply_chain", MODEL_NAME)

    # ✅ RESET (FIXED)
    response = requests.post(f"{ENV_URL}/reset", json={"task": task})
    print("RESET RESPONSE:", response.text)

    r = response.json()

    rewards = []
    steps = 0

    while True:
        obs = r["observation"]

        action = get_action(obs)

        # ✅ STEP (FIXED)
        response = requests.post(f"{ENV_URL}/step", json=action)
        print("RAW RESPONSE:", response.text)

        # Handle non-JSON safely
        try:
            r = response.json()
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            break

        reward = r.get("reward", 0.0)
        done = r.get("done", True)

        rewards.append(reward)
        steps += 1

        log_step(steps, str(action), reward, done, None)

        if done:
            break

    # ✅ SAFE SCORE
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.0, min(score, 1.0))
    success = score > 0

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(run())