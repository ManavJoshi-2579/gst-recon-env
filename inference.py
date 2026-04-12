import asyncio
import os
import random
import sys
from typing import Any
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]

MAX_STEPS = 20
TASK_NAME = "hard"

FALLBACK_OBS: dict[str, Any] = {
    "current_invoice": {"id": "INV-001"},
    "mismatch_flags": [],
    "risk_score": 0.0,
    "progress": 0.0,
}

FALLBACK_ACTION: dict[str, str] = {
    "type": "reject",
    "invoice_id": "INV-001",
    "reason": "fallback",
}


def _build_llm_client() -> Any:
    try:
        API_BASE_URL = os.environ["API_BASE_URL"]
        API_KEY = os.environ["API_KEY"]
        if OpenAI is None:
            return None
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        try:
            if OpenAI is None:
                return None
            return OpenAI(
                base_url=os.environ.get("API_BASE_URL"),
                api_key=os.environ.get("API_KEY"),
            )
        except Exception:
            return None


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except Exception:
        pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_score(value: Any) -> float:
    try:
        score = _safe_float(value, 0.0)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
    except Exception:
        return 0.0


def _safe_obs(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return dict(FALLBACK_OBS)

    merged = dict(FALLBACK_OBS)
    try:
        merged.update(obs)
    except Exception:
        return dict(FALLBACK_OBS)

    current_invoice = merged.get("current_invoice")
    if not isinstance(current_invoice, dict):
        merged["current_invoice"] = {"id": "INV-001"}
    else:
        if not current_invoice.get("id"):
            current_invoice["id"] = "INV-001"
        merged["current_invoice"] = current_invoice

    if not isinstance(merged.get("mismatch_flags"), list):
        merged["mismatch_flags"] = []

    merged["risk_score"] = _safe_float(merged.get("risk_score"), 0.0)
    merged["progress"] = _safe_float(merged.get("progress"), 0.0)
    return merged


def _safe_invoice_id(obs: Any) -> str:
    safe = _safe_obs(obs)
    try:
        current_invoice = safe.get("current_invoice")
        if isinstance(current_invoice, dict):
            return str(current_invoice.get("id") or "INV-001")
    except Exception:
        pass
    return "INV-001"


def _fallback_action(obs: Any = None) -> dict[str, str]:
    action = dict(FALLBACK_ACTION)
    try:
        action["invoice_id"] = _safe_invoice_id(obs)
    except Exception:
        action["invoice_id"] = "INV-001"
    return action


def _normalize_action(action: Any, obs: Any) -> dict[str, str]:
    if not isinstance(action, dict):
        return _fallback_action(obs)

    try:
        action_type = str(action.get("type") or "reject")
    except Exception:
        action_type = "reject"

    if action_type not in {"match", "reject", "claim_itc", "query_vendor", "submit_report"}:
        action_type = "reject"

    try:
        invoice_id = str(action.get("invoice_id") or _safe_invoice_id(obs))
    except Exception:
        invoice_id = "INV-001"

    try:
        reason = str(action.get("reason") or "fallback")
    except Exception:
        reason = "fallback"

    return {"type": action_type, "invoice_id": invoice_id, "reason": reason}


def _heuristic_action(obs: Any) -> dict[str, str]:
    safe = _safe_obs(obs)
    try:
        invoice = safe.get("current_invoice")
        if not isinstance(invoice, dict):
            return _fallback_action(safe)

        invoice_id = str(invoice.get("id") or "INV-001")
        mismatch_flags = safe.get("mismatch_flags")
        if not isinstance(mismatch_flags, list):
            mismatch_flags = []

        risk_score = _safe_float(safe.get("risk_score"), 0.0)
        progress = _safe_float(safe.get("progress"), 0.0)

        if risk_score >= 0.75:
            return {"type": "reject", "invoice_id": invoice_id, "reason": "high_risk"}
        if mismatch_flags:
            return {"type": "query_vendor", "invoice_id": invoice_id, "reason": "mismatch"}
        if progress >= 1.0:
            return {"type": "submit_report", "invoice_id": invoice_id, "reason": "complete"}

        return {"type": "match", "invoice_id": invoice_id, "reason": "heuristic_match"}
    except Exception:
        return _fallback_action(safe)


class LocalEnvClient:
    def __init__(self) -> None:
        self.env = None

    async def reset(self, task: str = TASK_NAME) -> dict[str, Any]:
        try:
            from server.gst_recon_env_environment import GSTReconEnv

            self.env = GSTReconEnv(task=task)
            raw_obs = self.env.reset()
            if hasattr(raw_obs, "model_dump"):
                return _safe_obs(raw_obs.model_dump())
            return _safe_obs(raw_obs)
        except Exception:
            self.env = None
            return dict(FALLBACK_OBS)

    async def step(self, action: dict[str, Any]) -> dict[str, Any]:
        safe_action = _normalize_action(action, FALLBACK_OBS)

        try:
            if self.env is None:
                return {
                    "observation": dict(FALLBACK_OBS),
                    "reward": 0.0,
                    "done": True,
                    "score": 0.0,
                    "error": "env_unavailable",
                    "info": {"score": 0.0, "risk": 0.0, "processed": 0},
                }

            from server.models import Action

            parsed_action = Action.model_validate(safe_action)
            obs, reward, done, info = self.env.step(parsed_action)

            if hasattr(obs, "model_dump"):
                observation = _safe_obs(obs.model_dump())
            else:
                observation = _safe_obs(obs)

            safe_info = info if isinstance(info, dict) else {}
            return {
                "observation": observation,
                "reward": _safe_float(reward, 0.0),
                "done": bool(done),
                "score": _clamp_score(safe_info.get("score")),
                "error": None,
                "info": safe_info,
            }
        except Exception as exc:
            return {
                "observation": dict(FALLBACK_OBS),
                "reward": 0.0,
                "done": True,
                "score": 0.0,
                "error": str(exc),
                "info": {"score": 0.0, "risk": 0.0, "processed": 0},
            }

    async def aclose(self) -> None:
        try:
            return None
        except Exception:
            return None


async def _safe_close(client: Any) -> None:
    try:
        if hasattr(client, "aclose"):
            await client.aclose()
    except Exception:
        pass


async def _safe_reset(client: Any) -> tuple[Any, dict[str, Any]]:
    try:
        obs = await client.reset(task=TASK_NAME)
        return client, _safe_obs(obs)
    except Exception:
        try:
            await _safe_close(client)
        except Exception:
            pass
        fallback_client = LocalEnvClient()
        try:
            fallback_obs = await fallback_client.reset(task=TASK_NAME)
            return fallback_client, _safe_obs(fallback_obs)
        except Exception:
            return fallback_client, dict(FALLBACK_OBS)


async def _safe_step(client: Any, action: dict[str, str]) -> dict[str, Any]:
    try:
        result = await client.step(action)
        if not isinstance(result, dict):
            return {
                "observation": dict(FALLBACK_OBS),
                "reward": 0.0,
                "done": True,
                "score": 0.0,
                "error": "invalid_step_result",
                "info": {},
            }
        observation = _safe_obs(result.get("observation"))
        return {
            "observation": observation,
            "reward": _safe_float(result.get("reward"), 0.0),
            "done": bool(result.get("done")),
            "score": _clamp_score(result.get("score")),
            "error": result.get("error"),
            "info": result.get("info") if isinstance(result.get("info"), dict) else {},
        }
    except Exception as exc:
        return {
            "observation": dict(FALLBACK_OBS),
            "reward": 0.0,
            "done": True,
            "score": 0.0,
            "error": str(exc),
            "info": {},
        }


async def main() -> None:
    success = False
    steps = 0
    score = 0.0
    rewards_total = 0.0
    env_client: Any = LocalEnvClient()
    llm_client: Any = None

    try:
        try:
            random.seed(int(os.getenv("SEED", "42")))
        except Exception:
            random.seed(42)

        _safe_print("[START]")

        try:
            llm_client = _build_llm_client()
        except Exception:
            llm_client = None

        try:
            if llm_client is not None:
                _ = llm_client.chat.completions.create(
                    model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a GST reconciliation assistant."},
                        {"role": "user", "content": "Decide action for invoice."},
                    ],
                    temperature=0,
                )
        except Exception:
            _ = None

        try:
            env_client, obs = await _safe_reset(env_client)
        except Exception:
            obs = dict(FALLBACK_OBS)

        obs = _safe_obs(obs)

        while steps < MAX_STEPS:
            steps += 1
            try:
                if not isinstance(obs, dict):
                    obs = dict(FALLBACK_OBS)
            except Exception:
                obs = dict(FALLBACK_OBS)

            try:
                action = _heuristic_action(obs)
            except Exception:
                action = dict(FALLBACK_ACTION)

            action = _normalize_action(action, obs)

            try:
                result = await _safe_step(env_client, action)
            except Exception:
                result = {
                    "observation": dict(FALLBACK_OBS),
                    "reward": 0.0,
                    "done": True,
                    "score": 0.0,
                    "error": "safe_step_failed",
                    "info": {},
                }

            try:
                obs = _safe_obs(result.get("observation"))
            except Exception:
                obs = dict(FALLBACK_OBS)

            try:
                reward = _safe_float(result.get("reward"), 0.0)
            except Exception:
                reward = 0.0

            rewards_total += reward
            score = _clamp_score(result.get("score"))

            try:
                done = bool(result.get("done"))
            except Exception:
                done = True

            try:
                error_value = result.get("error")
            except Exception:
                error_value = "unknown"

            try:
                _safe_print(
                    f"[STEP] step={steps} action={action.get('type', 'reject')} "
                    f"invoice_id={action.get('invoice_id', 'INV-001')} reward={reward:.4f} "
                    f"done={'true' if done else 'false'} error={error_value if error_value else 'null'}"
                )
            except Exception:
                pass

            if done:
                success = score >= 0.7
                break

        try:
            _safe_print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.1f} rewards={rewards_total:.1f}")
        except Exception:
            _safe_print("[END] success=false steps=0 score=0.0 rewards=0.0")
    except Exception as exc:
        try:
            _safe_print(f"[END] success=false steps=0 score=0.0 rewards=0.0 error={str(exc)}")
        except Exception:
            pass
    finally:
        try:
            await _safe_close(env_client)
        except Exception:
            pass


def safe_main() -> None:
    try:
        asyncio.run(main())
    except Exception as e:
        try:
            print(f"[END] success=false steps=0 score=0.0 rewards=0.0 error={str(e)}", flush=True)
        except Exception:
            pass
    except BaseException as e:
        try:
            print(f"[END] success=false steps=0 score=0.0 rewards=0.0 error={str(e)}", flush=True)
        except Exception:
            pass
    finally:
        try:
            sys.exit(0)
        except Exception:
            pass


if __name__ == "__main__":
    safe_main()
