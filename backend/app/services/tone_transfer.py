"""DDSP timbre-transfer orchestration.

Triggers a RunPod `tone_transfer` job that re-voices a stem as a target instrument
(preserving pitch/performance), writes the render to R2, and returns a presigned URL.
Blocking flow (the render takes ~30-120s incl. cold start); reuses the same autoscale +
poll approach as separation.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .. import db
from ..config import Settings
from ..models import Project, ProjectStatus, Stem
from ..runpod_client import DONE, RunPodClient
from ..storage import Storage

# Instruments the worker's DDSP engine can currently render (mono, audio-driven).
DDSP_INSTRUMENTS = ["violin"]


async def _maybe_scale_down(settings: Settings, runpod: RunPodClient, project_id: str) -> None:
    if not settings.runpod_autoscale:
        return
    # don't scale to zero if a separation is still in flight
    others = [p for p in db.list_all() if p.status == ProjectStatus.separating]
    if not others:
        await runpod.set_max_workers(0)


async def run_tone_transfer(
    project: Project,
    stem: Stem,
    instrument: str,
    settings: Settings,
    storage: Storage,
    runpod: RunPodClient,
) -> str:
    """Run the render synchronously; returns a presigned GET URL for the rendered audio."""
    output_key = f"renders/{project.id}/{stem.name}_{instrument}.flac"

    if settings.runpod_autoscale:
        await runpod.set_max_workers(settings.runpod_max_workers)

    job_id = await runpod.run(
        {
            "task": "tone_transfer",
            "audio_key": stem.key,
            "instrument": instrument,
            "mode": "mono",
            "output_key": output_key,
            "bucket": settings.r2_bucket,
        }
    )

    interval = max(1.0, settings.runpod_poll_interval_seconds)
    steps = int(settings.runpod_job_timeout_seconds / interval)
    for _ in range(steps):
        await asyncio.sleep(interval)
        result = await runpod.status(job_id)
        status = result.get("status", "")
        if status == DONE:
            await _maybe_scale_down(settings, runpod, project.id)
            return storage.presign_get(output_key)
        if RunPodClient.is_terminal(status):
            await _maybe_scale_down(settings, runpod, project.id)
            raise RuntimeError(str(result.get("error") or status))

    await _maybe_scale_down(settings, runpod, project.id)
    raise RuntimeError("tone-transfer timed out")
