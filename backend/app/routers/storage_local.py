"""Local-storage endpoints used only in mock mode.

Presigned URLs from ``LocalStorage`` point here, so the browser's PUT (upload) and GET
(download) go straight through the backend to ``scratch/storage`` on disk — mimicking R2.
"""
from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from ..storage import LocalStorage

router = APIRouter(prefix="/api/storage", tags=["storage"])


def _require_local(request: Request) -> LocalStorage:
    storage = request.app.state.storage
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status_code=404, detail="local storage disabled (using R2)")
    return storage


@router.put("/{key:path}")
async def put_object(key: str, request: Request):
    storage = _require_local(request)
    body = await request.body()
    try:
        storage.write_bytes(key, body)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid key")
    return Response(status_code=200)


@router.get("/{key:path}")
async def get_object(key: str, request: Request):
    storage = _require_local(request)
    try:
        path = storage.read_path(key)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid key")
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    # Accept-Ranges so the browser can seek within audio.
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
