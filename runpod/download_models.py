"""Build-time model prefetch.

Run during `docker build` so model weights are baked into the image — then a cold start is
just weights -> VRAM (no network download at runtime), which FlashBoot can snapshot.

Best-effort: if a model download fails at build (e.g. registry hiccup), the build still
succeeds and the handler downloads the model lazily on first use.

Prefetches two families:
  1. Separator models (audio-separator via HF): the existing behavior.
  2. AFTER (ACIDS-IRCAM) polyphonic timbre-transfer checkpoints from IRCAM's Nextcloud
     share — the .ts files are 200-400 MB each, past GitHub's 100 MB single-file cap,
     so they cannot be committed to the repo directly.
"""
from __future__ import annotations

import base64
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from separator_engine import load_model_config  # noqa: E402


# --- AFTER TorchScript checkpoints ------------------------------------------------
# Fetched from https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P/ via the Nextcloud
# WebDAV public-share endpoint. The share token is not a secret — it's the public URL
# IRCAM ships in the README of acids-ircam/AFTER for anyone to download.
_AFTER_SHARE_TOKEN = "8NFD5gWwbkT4G5P"
_AFTER_BASE_URL = "https://nubo.ircam.fr/public.php/webdav/After%20models"

# Every entry MUST match a MODELS[<name>] entry in runpod/after_engine.py. Only fetch
# the checkpoints the engine actually offers as an instrument.
AFTER_CHECKPOINTS = [
    "afterv2.audio.instr.ts",       # 210 MB — general instruments
    "afterv2.audio.orchestral.ts",  # 234 MB — piano → strings + brass demo target
    "afterv1.audio.drums.ts",       # 411 MB — drums (fallback while Track C IDM lands)
    "afterv2.audio.speech.ts",      # 234 MB — voice-to-voice curiosity
]


def _fetch_after_checkpoint(name: str, dest_dir: Path) -> None:
    """Download a single AFTER .ts checkpoint via Nextcloud WebDAV basic-auth."""
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 100_000_000:  # sanity: >100 MB = probably a real .ts
        print(f"[after] {name} already present ({dest.stat().st_size/1e6:.0f} MB) — skipping", flush=True)
        return
    url = f"{_AFTER_BASE_URL}/{name.replace(' ', '%20')}"
    auth = base64.b64encode(f"{_AFTER_SHARE_TOKEN}:".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    print(f"[after] downloading {name} …", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"[after] wrote {dest} ({dest.stat().st_size/1e6:.0f} MB)", flush=True)
    except Exception as exc:
        # Best-effort — leave a partial file removed and continue; the engine's lazy
        # loader will raise a clean error at runtime if the checkpoint really is missing.
        print(f"[after] WARN {name}: {exc}", flush=True)
        if dest.exists():
            dest.unlink()


def prefetch_after() -> None:
    after_dir = Path(__file__).parent / "after_models"
    after_dir.mkdir(parents=True, exist_ok=True)
    for name in AFTER_CHECKPOINTS:
        _fetch_after_checkpoint(name, after_dir)


def prefetch_separator() -> None:
    cfg = load_model_config()
    tiers = [t for t in ("2stem", "4stem", "6stem") if t in cfg]
    try:
        from audio_separator.separator import Separator

        model_dir = str(Path(__file__).parent / "models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        sep = Separator(model_file_dir=model_dir, output_dir="/tmp/dl")
        for tier in tiers:
            model = cfg[tier]["model"]
            try:
                print(f"[prefetch] {tier} -> {model}", flush=True)
                sep.load_model(model_filename=model)
            except Exception as exc:
                print(f"[prefetch] WARN {tier} {model}: {exc}", flush=True)
    except Exception as exc:
        print(f"[prefetch] skipped ({exc}); models will download at runtime", flush=True)


def main() -> None:
    prefetch_separator()
    prefetch_after()


if __name__ == "__main__":
    main()
