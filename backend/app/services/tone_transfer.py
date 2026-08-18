"""Timbre-transfer orchestration.

Triggers a RunPod `tone_transfer` job that re-voices a stem, writes the render to R2,
and returns a presigned URL. Blocking flow (the render takes ~30-120s incl. cold start);
reuses the same autoscale + poll approach as separation.

Multiple engines share this orchestrator (dispatched on the payload's `engine` field by
the worker's handler):
  - "ddsp"  — the shipping mono-pitched engine (violin48 checkpoint). Default.
  - "after" — Track A pretrained AFTER checkpoints for polyphonic pitched instruments.
  - "drums" — Track C Inverse Drum Machine for drum-kit swap.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .. import db
from ..config import Settings
from ..models import Project, ProjectStatus, Stem
from ..runpod_client import DONE, RunPodClient
from ..storage import Storage

# --- Per-engine instrument catalogues ---
# The frontend queries these via /meta/{engine}-instruments to populate its dropdown.
# Only violin is currently deployed on the worker; Tracks A/C populate their own lists.
DDSP_INSTRUMENTS = ["violin"]
AFTER_INSTRUMENTS: list[str] = []   # populated once Track A weights land
DRUM_KITS: list[str] = []           # populated once Track C weights land


# In-flight tone-transfer counter (module-scope). Guards against scaling workers to 0
# mid-job when two tone-transfer jobs overlap. Previously _maybe_scale_down only counted
# separations (ProjectStatus.separating), missing concurrent tone-transfer jobs entirely.
_transfer_jobs_in_flight = 0
_transfer_jobs_lock = asyncio.Lock()


async def _inc_in_flight() -> None:
    global _transfer_jobs_in_flight
    async with _transfer_jobs_lock:
        _transfer_jobs_in_flight += 1


async def _dec_in_flight() -> None:
    global _transfer_jobs_in_flight
    async with _transfer_jobs_lock:
        _transfer_jobs_in_flight = max(0, _transfer_jobs_in_flight - 1)


async def _maybe_scale_down(settings: Settings, runpod: RunPodClient, project_id: str) -> None:
    """Scale endpoint workers to 0 when nothing else needs them.

    Checks both in-flight separations AND in-flight tone-transfer jobs — a completed
    tone-transfer must NOT scale down while another tone-transfer or a separation is
    still running. Callers must call this AFTER releasing their own in-flight slot.
    """
    if not settings.runpod_autoscale:
        return
    seps = [p for p in db.list_all() if p.status == ProjectStatus.separating]
    if seps:
        return
    async with _transfer_jobs_lock:
        if _transfer_jobs_in_flight > 0:
            return
    await runpod.set_max_workers(0)


async def run_tone_transfer(
    project: Project,
    stem: Stem,
    instrument: str,
    settings: Settings,
    storage: Storage,
    runpod: RunPodClient,
    engine: str = "ddsp",
) -> str:
    """Run the render synchronously; returns a presigned GET URL for the rendered audio.

    `engine` selects which worker-side engine handles the job. Default "ddsp" preserves
    the existing shipping behaviour; new engines are added by (a) populating the matching
    catalogue above, (b) implementing the render path in a new worker engine class, and
    (c) extending the worker's handler dispatch on the `engine` field.
    """
    # DDSP kept its historical key format; other engines get an engine-prefixed key so
    # renders from different engines targeting the same instrument name don't collide.
    if engine == "ddsp":
        output_key = f"renders/{project.id}/{stem.name}_{instrument}.flac"
    else:
        output_key = f"renders/{project.id}/{stem.name}_{engine}_{instrument}.flac"

    await _inc_in_flight()
    try:
        if settings.runpod_autoscale:
            await runpod.set_max_workers(settings.runpod_max_workers)

        job_id = await runpod.run(
            {
                "task": "tone_transfer",
                "engine": engine,
                "audio_key": stem.key,
                "instrument": instrument,
                "mode": "mono",  # DDSP-specific; other engines ignore it
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
                return storage.presign_get(output_key)
            if RunPodClient.is_terminal(status):
                raise RuntimeError(str(result.get("error") or status))

        raise RuntimeError("tone-transfer timed out")
    finally:
        # Order matters: decrement BEFORE the scale-down check, so this job's own slot
        # is released and the check sees the true in-flight count.
        await _dec_in_flight()
        await _maybe_scale_down(settings, runpod, project.id)
