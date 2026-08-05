"""Domain models and shared constants (stem tiers, colors, request/response schemas)."""
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Tier(str, Enum):
    two = "2stem"
    four = "4stem"
    six = "6stem"


class ProjectStatus(str, Enum):
    created = "created"        # project row exists, awaiting upload
    uploaded = "uploaded"      # audio is in storage
    separating = "separating"  # RunPod job in flight
    ready = "ready"            # stems available
    failed = "failed"


# Which stems each tier produces, in display order.
STEM_TIERS: Dict[Tier, List[str]] = {
    Tier.two: ["vocals", "instrumental"],
    Tier.four: ["vocals", "drums", "bass", "other"],
    Tier.six: ["vocals", "drums", "bass", "guitar", "piano", "other"],
}

# Per-stem track colors (flat, retro-2010s palette). Consistent across waveform + meters.
STEM_COLORS: Dict[str, str] = {
    "vocals": "#E0564B",       # warm red
    "instrumental": "#4A8FC0", # blue
    "drums": "#E8A13A",        # amber
    "bass": "#7C5CD0",         # violet
    "guitar": "#3FA66A",       # green
    "piano": "#2E9FB0",        # teal
    "other": "#8A93A6",        # slate
}

DEFAULT_COLOR = "#8A93A6"


class Note(BaseModel):
    id: str
    start: float          # seconds
    dur: float            # seconds
    midi: float           # current (possibly edited) pitch; fractional allowed (cents)
    original_midi: int    # detected pitch (for reset / cents deviation)
    confidence: float = 1.0


class Stem(BaseModel):
    name: str
    key: str                       # storage object key
    color: str
    url: Optional[str] = None      # presigned GET url (populated on read)
    analyzed: bool = False
    notes: List[Note] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    name: str
    original_filename: str
    tier: Tier
    status: ProjectStatus
    upload_key: str
    created_at: str
    job_id: Optional[str] = None
    error: Optional[str] = None
    stems: List[Stem] = Field(default_factory=list)
    # Musical metadata (populated by later milestones; kept optional now).
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    duration_seconds: Optional[float] = None


# ---- Request / response schemas ----

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    filename: str = Field(..., min_length=1, max_length=300)
    tier: Tier = Tier.four
    content_type: str = "audio/mpeg"


class CreateProjectResponse(BaseModel):
    project: Project
    upload_url: str
    upload_method: str = "PUT"


class ProjectResponse(BaseModel):
    project: Project
