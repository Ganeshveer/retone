"""Stem-separation orchestration.

Bridges a Project to either the mock path (local, instant) or a real RunPod job.
The RunPod worker (see ../../runpod/handler.py) reads audio from R2, separates it, writes
stems back to R2, and returns a list of ``{name, key}`` entries.
"""
from __future__ import annotations

import os
from typing import List

from .. import db
from ..config import Settings
from ..models import (
    DEFAULT_COLOR,
    STEM_COLORS,
    STEM_TIERS,
    Project,
    ProjectStatus,
    Stem,
)
from ..runpod_client import DONE, RunPodClient
from ..storage import LocalStorage, Storage


def _stem_key(project: Project, stem_name: str, ext: str) -> str:
    return f"stems/{project.id}/{stem_name}{ext}"


def _color(stem_name: str) -> str:
    return STEM_COLORS.get(stem_name, DEFAULT_COLOR)


def populate_urls(project: Project, storage: Storage) -> Project:
    """Attach fresh presigned GET urls to every stem (urls are short-lived)."""
    for stem in project.stems:
        stem.url = storage.presign_get(stem.key)
    return project


async def start_separation(
    project: Project,
    settings: Settings,
    storage: Storage,
    runpod: RunPodClient | None,
) -> Project:
    stem_names = STEM_TIERS[project.tier]

    if settings.mock_mode or runpod is None:
        return _mock_separate(project, stem_names, storage)

    output_prefix = f"stems/{project.id}/"
    if settings.runpod_autoscale:
        await runpod.set_max_workers(settings.runpod_max_workers)  # wake the endpoint
    job_id = await runpod.run(
        {
            "task": "separate",
            "audio_key": project.upload_key,
            "tier": project.tier.value,
            "output_prefix": output_prefix,
            "output_format": "flac",
            "bucket": settings.r2_bucket,
        }
    )
    project.job_id = job_id
    project.status = ProjectStatus.separating
    project.error = None
    return project


def _mock_separate(project: Project, stem_names: List[str], storage: Storage) -> Project:
    """Fake separation: copy the uploaded file into a key per stem so the multitrack
    player has real, playable audio in every lane. Waveforms will look identical — the
    point is to exercise the full upload -> separate -> playback flow with no GPU."""
    _, ext = os.path.splitext(project.upload_key)
    ext = ext or ".mp3"
    stems: List[Stem] = []
    for name in stem_names:
        key = _stem_key(project, name, ext)
        try:
            if isinstance(storage, LocalStorage):
                storage.copy(project.upload_key, key)
            else:
                storage.copy(project.upload_key, key)
        except Exception:
            # If the copy fails (e.g. upload not present yet), fall back to the original.
            key = project.upload_key
        stems.append(Stem(name=name, key=key, color=_color(name)))
    project.stems = stems
    project.status = ProjectStatus.ready
    project.error = None
    return project


async def refresh_status(
    project: Project,
    settings: Settings,
    storage: Storage,
    runpod: RunPodClient | None,
) -> Project:
    """If a RunPod job is in flight, poll it once and fold the result into the project."""
    if project.status != ProjectStatus.separating or not project.job_id or runpod is None:
        return project

    result = await runpod.status(project.job_id)
    status = result.get("status", "")

    if status == DONE:
        output = result.get("output") or {}
        stem_entries = output.get("stems") or []
        stems: List[Stem] = []
        for entry in stem_entries:
            name = entry.get("name")
            key = entry.get("key")
            if name and key:
                stems.append(Stem(name=name, key=key, color=_color(name)))
        project.stems = stems
        project.status = ProjectStatus.ready if stems else ProjectStatus.failed
        if not stems:
            project.error = "worker returned no stems"
        project.bpm = output.get("bpm")
        project.musical_key = output.get("key")
        project.duration_seconds = output.get("duration_seconds")
    elif RunPodClient.is_terminal(status):
        project.status = ProjectStatus.failed
        project.error = str(result.get("error") or status)

    # Scale the endpoint back to 0 once no separations remain in flight (zero idle cost).
    if settings.runpod_autoscale and project.status in (ProjectStatus.ready, ProjectStatus.failed):
        others = [
            p for p in db.list_all()
            if p.status == ProjectStatus.separating and p.id != project.id
        ]
        if not others:
            await runpod.set_max_workers(0)

    return project
