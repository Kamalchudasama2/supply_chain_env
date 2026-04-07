from fastapi import FastAPI
from env.env import SupplyChainEnv
from env.tasks import TASKS
from env.models import SupplyChainAction

app = FastAPI()

env = None

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

@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = SupplyChainEnv(TASKS[task])
    return env.reset()

@app.post("/step")
def step(action: SupplyChainAction):
    global env
    if env is None:
        return {"error": "Call /reset first"}
    return env.step(action)

@app.get("/state")
def state():
    if env is None:
        return {"error": "Call /reset first"}
    return env.state()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
