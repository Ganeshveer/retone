"""Project lifecycle: create (+presigned upload) -> separate -> poll -> stems."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request

from .. import db
from ..models import (
    CreateProjectRequest,
    CreateProjectResponse,
    Note,
    Project,
    ProjectResponse,
    ProjectStatus,
)
from ..services import analysis, separation
from ..storage import LocalStorage

router = APIRouter(prefix="/api/projects", tags=["projects"])

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(filename: str) -> str:
    base = os.path.basename(filename).strip() or "audio"
    return _SAFE.sub("_", base)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("", response_model=CreateProjectResponse)
def create_project(req: CreateProjectRequest, request: Request) -> CreateProjectResponse:
    storage = request.app.state.storage
    project_id = uuid.uuid4().hex[:12]
    filename = _safe_filename(req.filename)
    upload_key = f"uploads/{project_id}/{filename}"

    project = Project(
        id=project_id,
        name=req.name,
        original_filename=filename,
        tier=req.tier,
        status=ProjectStatus.created,
        upload_key=upload_key,
        created_at=_now(),
    )
    db.save(project)

    upload_url = storage.presign_put(upload_key, content_type=req.content_type)
    return CreateProjectResponse(project=project, upload_url=upload_url)


@router.get("", response_model=List[Project])
def list_projects() -> List[Project]:
    return db.list_all()


@router.post("/{project_id}/separate", response_model=ProjectResponse)
async def separate(project_id: str, request: Request) -> ProjectResponse:
    settings = request.app.state.settings
    storage = request.app.state.storage
    runpod = request.app.state.runpod

    project = db.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")

    if not storage.exists(project.upload_key):
        raise HTTPException(status_code=409, detail="upload not found; PUT the file first")

    if project.status == ProjectStatus.created:
        project.status = ProjectStatus.uploaded

    project = await separation.start_separation(project, settings, storage, runpod)
    db.save(project)

    if project.status == ProjectStatus.ready:
        separation.populate_urls(project, storage)
    return ProjectResponse(project=project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, request: Request) -> ProjectResponse:
    settings = request.app.state.settings
    storage = request.app.state.storage
    runpod = request.app.state.runpod

    project = db.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")

    if project.status == ProjectStatus.separating:
        project = await separation.refresh_status(project, settings, storage, runpod)
        db.save(project)

    if project.status == ProjectStatus.ready:
        separation.populate_urls(project, storage)
    return ProjectResponse(project=project)


def _find_stem(project: Project, stem_name: str):
    stem = next((s for s in project.stems if s.name == stem_name), None)
    if stem is None:
        raise HTTPException(status_code=404, detail="stem not found")
    return stem


# NOTE: sync def -> FastAPI runs analysis in a threadpool so the librosa work
# (CPU-heavy pyin) doesn't block the event loop.
@router.post("/{project_id}/stems/{stem_name}/analyze", response_model=ProjectResponse)
def analyze_stem(project_id: str, stem_name: str, request: Request) -> ProjectResponse:
    storage = request.app.state.storage
    project = db.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    stem = _find_stem(project, stem_name)

    if not isinstance(storage, LocalStorage):
        raise HTTPException(
            status_code=501,
            detail="note analysis runs in local/mock mode only; RunPod transcription not yet wired",
        )
    path = storage.read_path(stem.key)
    if path is None:
        raise HTTPException(status_code=409, detail="stem file not found")

    notes_raw, bpm, key, dur = analysis.analyze_stem(str(path))
    stem.notes = [
        Note(
            id=f"{stem_name}-{i}",
            start=n["start"],
            dur=n["dur"],
            midi=n["midi"],
            original_midi=n["midi"],
            confidence=n["confidence"],
        )
        for i, n in enumerate(notes_raw)
    ]
    stem.analyzed = True
    if project.bpm is None:
        project.bpm = bpm
    if project.musical_key is None:
        project.musical_key = key
    if project.duration_seconds is None:
        project.duration_seconds = dur

    db.save(project)
    separation.populate_urls(project, storage)
    return ProjectResponse(project=project)


@router.put("/{project_id}/stems/{stem_name}/notes", response_model=ProjectResponse)
def save_notes(
    project_id: str, stem_name: str, notes: List[Note], request: Request
) -> ProjectResponse:
    storage = request.app.state.storage
    project = db.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    stem = _find_stem(project, stem_name)
    stem.notes = notes
    stem.analyzed = True
    db.save(project)
    separation.populate_urls(project, storage)
    return ProjectResponse(project=project)
