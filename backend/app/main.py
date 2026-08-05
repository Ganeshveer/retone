"""ReTone FastAPI application entrypoint.

Run locally:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import projects, storage_local
from .runpod_client import build_runpod_client
from .storage import build_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.storage = build_storage(settings)
    app.state.runpod = build_runpod_client(settings)
    yield


app = FastAPI(title="ReTone API", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(projects.router)
app.include_router(storage_local.router)


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "env": s.retone_env,
        "mock_mode": s.mock_mode,
        "r2_configured": s.r2_configured,
        "runpod_configured": s.runpod_configured,
    }


@app.get("/")
def root():
    return {"name": "ReTone API", "docs": "/docs", "health": "/health"}
