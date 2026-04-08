from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from .models import Action, Observation

from .gst_recon_env_environment import GSTReconEnv

app = FastAPI(title="GST-Recon-Env Server")

env = None
task_name: Optional[str] = None

@app.get("/")
def health():
    return {"status": "ok"}

class StepRequest(BaseModel):
    action: Action

class ResetRequest(BaseModel):
    task: str = "easy"

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    score: Optional[float] = None
    error: Optional[str] = None
    info: Optional[Dict[str, Any]] = None

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    global env, task_name
    task_name = req.task
    env = GSTReconEnv(task=req.task)
    return env.reset()

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(400, "Call reset first")
    
    obs, reward, done, info = env.step(req.action)
    
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        score=env._calculate_grader_score() if done else None,
        error=env.last_error,
        info=info
    )

@app.get("/state")
def get_state():
    if env is None:
        raise HTTPException(400, "No active episode")
    return env.state()

@app.get("/tasks")
def get_tasks():
    return ["easy", "medium", "hard"]

def main():
    """Entry point for [project.scripts]"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_server():
    main()

if __name__ == "__main__":
    main()
