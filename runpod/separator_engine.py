"""Stem-separation engine wrapper around `audio-separator` (MIT).

`audio-separator` auto-downloads model weights and supports RoFormer (2-stem
vocals/instrumental) and Demucs (4/6-stem). We keep one Separator instance and swap the
loaded model per tier, caching so repeated jobs of the same tier don't reload.

To upgrade a tier to the exact chosen research models (SCNet-XL 4-stem, BS-ROFO-SW
6-stem), override MODEL_4STEM / MODEL_6STEM env vars with an MSST checkpoint, or replace
this engine with a thin MSST `inference.py` call — the handler contract (tier in, list of
{name, path} out) stays the same.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).parent
MODEL_DIR = os.environ.get("MODEL_DIR", str(HERE / "models"))
OUTPUT_DIR = os.environ.get("SEP_OUTPUT_DIR", "/tmp/retone_out")

# Stem name -> the token audio-separator puts in output filenames, e.g. "..._(Vocals)_...".
_STEM_TOKENS = {
    "vocals": "vocals",
    "instrumental": "instrumental",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "other": "other",
}


def load_model_config() -> dict:
    cfg = json.loads((HERE / "models.json").read_text())
    # Env overrides (change model without rebuilding the image).
    for tier, env in (("2stem", "MODEL_2STEM"), ("4stem", "MODEL_4STEM"), ("6stem", "MODEL_6STEM")):
        override = os.environ.get(env)
        if override and tier in cfg:
            cfg[tier]["model"] = override
    return cfg


class SeparatorEngine:
    def __init__(self) -> None:
        from audio_separator.separator import Separator  # imported lazily (heavy)

        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self._separator = Separator(
            model_file_dir=MODEL_DIR,
            output_dir=OUTPUT_DIR,
            output_format="FLAC",
        )
        self._loaded_model: str | None = None
        self.config = load_model_config()

    def _ensure_model(self, model_filename: str) -> None:
        if self._loaded_model != model_filename:
            self._separator.load_model(model_filename=model_filename)
            self._loaded_model = model_filename

    def warm(self, tier: str) -> None:
        """Preload a tier's model (called at boot so FlashBoot snapshots it)."""
        self._ensure_model(self.config[tier]["model"])

    def separate(self, tier: str, input_path: str) -> List[Tuple[str, str]]:
        """Run separation; return [(stem_name, output_file_path), ...] in tier order."""
        if tier not in self.config:
            raise ValueError(f"unknown tier: {tier}")
        model = self.config[tier]["model"]
        wanted = self.config[tier]["stems"]
        self._ensure_model(model)

        produced = self._separator.separate(input_path)  # list of output filenames
        produced_paths = [str(Path(OUTPUT_DIR) / p) for p in produced]

        matched: Dict[str, str] = {}
        for stem in wanted:
            token = _STEM_TOKENS.get(stem, stem)
            for path in produced_paths:
                if f"({token})".lower() in Path(path).name.lower():
                    matched[stem] = path
                    break
        # Fallback: some 2-stem models emit "(No Vocals)" for the instrumental.
        if "instrumental" in wanted and "instrumental" not in matched:
            for path in produced_paths:
                if "no vocals" in Path(path).name.lower() or "no_vocals" in Path(path).name.lower():
                    matched["instrumental"] = path
                    break

        ordered = [(s, matched[s]) for s in wanted if s in matched]
        if not ordered:
            raise RuntimeError(
                f"separation produced no recognizable stems for tier {tier}; "
                f"got files: {[Path(p).name for p in produced_paths]}"
            )
        return ordered
