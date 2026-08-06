"""Thin async client for the RunPod Serverless HTTP API.

Docs: https://docs.runpod.io/serverless/endpoints/send-requests
Base URL: https://api.runpod.ai/v2/<endpoint_id>
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from .config import Settings

# Terminal RunPod job states.
DONE = "COMPLETED"
FAILED_STATES = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, settings: Settings):
        if not settings.runpod_configured:
            raise RunPodError("RunPod is not configured (set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)")
        self.settings = settings
        self.base = f"https://api.runpod.ai/v2/{settings.runpod_endpoint_id}"
        self.rest = f"https://rest.runpod.io/v1/endpoints/{settings.runpod_endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {settings.runpod_api_key}",
            "Content-Type": "application/json",
        }

    async def set_max_workers(self, n: int) -> bool:
        """Scale the endpoint's max workers (0 = paused, no cost). Best-effort."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.patch(self.rest, headers=self.headers, json={"workersMax": n})
            return resp.status_code < 400
        except Exception:
            return False

    async def run(self, job_input: Dict[str, Any]) -> str:
        """Submit an async job and return its id. Retries while the endpoint is scaling up
        from paused (409 ENDPOINT_PAUSED)."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = None
            for _ in range(12):
                resp = await client.post(
                    f"{self.base}/run", headers=self.headers, json={"input": job_input}
                )
                if resp.status_code == 409:  # endpoint paused / still scaling up
                    await asyncio.sleep(2)
                    continue
                break
        if resp is None or resp.status_code >= 400:
            raise RunPodError(f"RunPod /run failed: {resp.status_code if resp else '?'} {resp.text if resp else ''}")
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RunPodError(f"RunPod /run returned no job id: {data}")
        return job_id

    async def status(self, job_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.base}/status/{job_id}", headers=self.headers)
        if resp.status_code >= 400:
            raise RunPodError(f"RunPod /status failed: {resp.status_code} {resp.text}")
        return resp.json()

    @staticmethod
    def is_terminal(status: str) -> bool:
        return status == DONE or status in FAILED_STATES


def build_runpod_client(settings: Settings) -> Optional[RunPodClient]:
    if not settings.runpod_configured:
        return None
    return RunPodClient(settings)
