"""Tiny SQLite-backed project store.

Projects are stored as JSON blobs keyed by id. This is deliberately minimal for the
MVP; swap for Postgres/SQLModel when the note model (M2) lands.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from .models import Project

_DB_PATH = Path("scratch/retone.sqlite3")
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def save(project: Project) -> Project:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO projects (id, created_at, data) VALUES (?, ?, ?)",
            (project.id, project.created_at, project.model_dump_json()),
        )
        conn.commit()
    return project


def get(project_id: str) -> Optional[Project]:
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT data FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    if row is None:
        return None
    return Project.model_validate(json.loads(row[0]))


def list_all(limit: int = 100) -> List[Project]:
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT data FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [Project.model_validate(json.loads(r[0])) for r in rows]
