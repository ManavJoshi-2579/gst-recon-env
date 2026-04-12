"""OpenEnv client compatibility layer."""
from typing import Dict, Any
import httpx

class OpenEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def reset(self, task: str = "easy") -> Dict[str, Any]:
        resp = await self._client.post("/reset", json={"task": task})
        resp.raise_for_status()
        return resp.json()

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._client.post("/step", json={"action": action})
        resp.raise_for_status()
        return resp.json()