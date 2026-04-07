import os
import random
import yaml
import json
import asyncio
import httpx

from client import OpenEnvClient
from server.gst_recon_env_environment import GSTReconEnv
from server.models import Action, ActionType

class LocalEnvClient:
    def __init__(self):
        self.env = None

    async def reset(self, task="easy"):
        self.env = GSTReconEnv(task=task)
        return self.env.reset().model_dump()

    async def step(self, action):
        if self.env is None:
            raise RuntimeError("Call reset first")

        parsed_action = Action.model_validate(action)
        obs, reward, done, info = self.env.step(parsed_action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "score": self.env._calculate_grader_score() if done else None,
            "error": self.env.last_error,
            "info": info,
        }

    async def aclose(self):
        return None

def load_env_config():
    with open("openenv.yaml", "r") as f:
        return yaml.safe_load(f)

def _build_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    import openai

    kwargs = {"api_key": api_key}
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url:
        kwargs["base_url"] = openai_base_url
    return openai.OpenAI(**kwargs)

def _heuristic_action(obs):
    invoice = obs.get("current_invoice")
    if not invoice:
        return {"type": "submit_report"}

    invoice_id = invoice["id"]
    has_match = any(entry.get("invoice_id") == invoice_id for entry in obs.get("available_gstr2b", []))

    if invoice.get("gstin", "").startswith("INVALID") or not invoice.get("is_einvoice", True):
        return {"type": "reject", "invoice_id": invoice_id, "reason": "invalid GSTIN or e-invoice"}
    if has_match:
        return {"type": "match", "invoice_id": invoice_id}
    return {"type": "reject", "invoice_id": invoice_id, "reason": "not found in GSTR-2B"}

async def main():
    api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")
    random.seed(int(os.getenv("SEED", "42")))
    
    print(f"[START] task=hard env=gst_recon_env model={model_name}")
    _ = hf_token
    
    client = OpenEnvClient(base_url=api_base)
    openai_client = _build_openai_client()

    try:
        # Reset environment
        try:
            obs = await client.reset(task="hard")
        except httpx.ConnectError:
            await client._client.aclose()
            print(f"[INFO] server unavailable at {api_base}; using local environment")
            client = LocalEnvClient()
            obs = await client.reset(task="hard")

        step_n = 0
        rewards = []
        success = False
        result = {"score": 0.0}

        while step_n < 50:
            step_n += 1

            # Generate action using LLM
            prompt = f"""
            GST Reconciliation Task. Current state: {json.dumps(obs)}
            Available actions: match/reject/claim_itc/query_vendor/submit_report
            Respond with ONLY valid JSON action: {{"type": "action_name", "invoice_id": "INV-001", "reason": "optional"}}
            """

            try:
                if openai_client is None:
                    raise RuntimeError("OPENAI_API_KEY is not set")

                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                action_json = response.choices[0].message.content.strip()
                action = json.loads(action_json)
            except Exception:
                action = _heuristic_action(obs)

            # Take step
            result = await client.step(action)
            obs = result["observation"]
            reward = result["reward"]
            rewards.append(reward)

            action_str = f"{action.get('type', 'unknown')}(id={action.get('invoice_id', 'N/A')})"
            print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={result['done']} error={result.get('error') or 'null'}")

            if result["done"]:
                success = (result.get("score") or 0.0) >= 0.7
                break

        print(f"[END] success={success} steps={step_n} score={result.get('score') or 0.0} rewards={','.join(f'{r:.2f}' for r in rewards)}")
    finally:
        await client.aclose() if hasattr(client, "aclose") else await client._client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
