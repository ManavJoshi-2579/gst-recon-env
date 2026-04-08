from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from .models import Action, Observation

from .gst_recon_env_environment import GSTReconEnv

app = FastAPI(title="GST-Recon-Env Server")

env = None
task_name: Optional[str] = None


def _empty_observation() -> Observation:
    return Observation(
        current_invoice=None,
        available_gstr2b=[],
        matched=[],
        mismatches=[],
        current_itc=0.0,
        total_itc_possible=0.0,
        progress=0.0,
        warnings=[],
        step_count=0,
    )


def _safe_state() -> Dict[str, Any]:
    return {
        "invoices": [],
        "processed": [],
        "risk_score": 0.0,
        "steps": 0,
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    path = request.url.path
    if path == "/step":
        return JSONResponse(
            status_code=200,
            content=StepResponse(
                observation=_empty_observation(),
                reward=0.0,
                done=True,
                score=0.0,
                error="Invalid request payload",
                info={"score": 0.0, "risk": 0.0, "processed": 0},
            ).model_dump(),
        )
    if path == "/reset":
        return JSONResponse(status_code=200, content=_empty_observation().model_dump())
    if path == "/state":
        return JSONResponse(status_code=200, content=_safe_state())
    return JSONResponse(status_code=200, content={"error": "Invalid request payload"})


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    path = request.url.path
    if path == "/step":
        return JSONResponse(
            status_code=200,
            content=StepResponse(
                observation=_empty_observation(),
                reward=0.0,
                done=True,
                score=0.0,
                error=str(exc),
                info={"score": 0.0, "risk": 0.0, "processed": 0},
            ).model_dump(),
        )
    if path == "/reset":
        return JSONResponse(status_code=200, content=_empty_observation().model_dump())
    if path == "/state":
        return JSONResponse(status_code=200, content=_safe_state())
    return JSONResponse(status_code=200, content={"error": str(exc)})

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
    try:
        task_name = req.task
        env = GSTReconEnv(task=req.task)
        return env.reset()
    except Exception:
        env = None
        task_name = req.task
        return _empty_observation()

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global env
    try:
        if env is None:
            env = GSTReconEnv(task=task_name or "easy")

        obs, reward, done, info = env.step(req.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            score=env._calculate_grader_score() if done else None,
            error=env.last_error,
            info=info,
        )
    except Exception as exc:
        return StepResponse(
            observation=_empty_observation(),
            reward=0.0,
            done=True,
            score=0.0,
            error=str(exc),
            info={"score": 0.0, "risk": 0.0, "processed": 0},
        )

@app.get("/state")
def get_state():
    try:
        if env is None:
            return _safe_state()
        return env.state()
    except Exception:
        return _safe_state()

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
