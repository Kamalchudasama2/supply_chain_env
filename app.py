from fastapi import FastAPI
from env.env import SupplyChainEnv
from env.tasks import TASKS
from env.models import SupplyChainAction  # ✅ IMPORTANT

app = FastAPI()

env = None


# ✅ Landing Page
@app.get("/")
def home():
    return {
        "message": "Supply Chain OpenEnv is running 🚀",
        "available_endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state"
        }
    }


# ✅ RESET
@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = SupplyChainEnv(TASKS[task])
    return env.reset()


# ✅ STEP (FIXED TYPE)
@app.post("/step")
def step(action: SupplyChainAction):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    try:
        return env.step(action)
    except Exception as e:
        return {"error": str(e)}


# ✅ STATE
@app.get("/state")
def state():
    if env is None:
        return {"error": "Call /reset first"}
    return env.state()