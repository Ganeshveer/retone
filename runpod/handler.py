"""RunPod Serverless handler for ReTone.

Tasks (job["input"]["task"]):
  - "ping"     -> health check, confirms the model warmed at boot.
  - "separate" -> {audio_key, tier, output_prefix, output_format, bucket}
                  pulls audio from R2, separates stems, writes them back to R2,
                  returns {"stems": [{"name","key"}, ...]}.

R2 credentials come from the worker's environment (set on the RunPod endpoint), not from
the job payload. Audio never travels through the job payload (RunPod's ~10MB cap) — only
object keys do.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import runpod

from separator_engine import SeparatorEngine

# --- R2 / S3 config from environment ---
R2_BUCKET_DEFAULT = os.environ.get("R2_BUCKET", "retone")


def _r2_endpoint() -> str:
    ep = os.environ.get("R2_ENDPOINT_URL")
    if ep:
        return ep
    acct = os.environ.get("R2_ACCOUNT_ID")
    if acct:
        return f"https://{acct}.r2.cloudflarestorage.com"
    raise RuntimeError("R2 endpoint not configured (set R2_ENDPOINT_URL or R2_ACCOUNT_ID)")


def _r2_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=_r2_endpoint(),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )


# IMPORTANT: do NOT build the engine or download models at import time. That would block
# the worker from reaching runpod.serverless.start() below, so it would never pull jobs
# (symptom: workers show "ready" but jobs sit IN_QUEUE forever). Load lazily on first job.
ENGINE: SeparatorEngine | None = None


def _get_engine() -> SeparatorEngine:
    global ENGINE
    if ENGINE is None:
        print("[worker] building separator engine (first job)…", flush=True)
        ENGINE = SeparatorEngine()
    return ENGINE


def _do_separate(inp: Dict[str, Any]) -> Dict[str, Any]:
    audio_key = inp["audio_key"]
    tier = inp.get("tier", "4stem")
    output_prefix = inp.get("output_prefix", f"stems/{Path(audio_key).stem}/")
    output_format = (inp.get("output_format") or "flac").lower()
    bucket = inp.get("bucket") or R2_BUCKET_DEFAULT

    s3 = _r2_client()
    engine = _get_engine()

    with tempfile.TemporaryDirectory() as tmp:
        local_in = str(Path(tmp) / Path(audio_key).name)
        s3.download_file(bucket, audio_key, local_in)

        stems = engine.separate(tier, local_in)  # [(name, path), ...]

        results = []
        for name, path in stems:
            key = f"{output_prefix}{name}.{output_format}"
            s3.upload_file(
                path, bucket, key,
                ExtraArgs={"ContentType": f"audio/{output_format}"},
            )
            results.append({"name": name, "key": key})

    return {"stems": results, "tier": tier}


# --- Timbre-transfer engines (each lazy-loaded on first job — see the L52-54 caveat) ---
_DDSP = None
_AFTER = None
_DRUMS = None


def _get_ddsp():
    global _DDSP
    if _DDSP is None:
        print("[worker] loading DDSP engine…", flush=True)
        from ddsp_engine import DDSPEngine

        _DDSP = DDSPEngine()
    return _DDSP


def _get_after():
    """AFTER (ACIDS-IRCAM latent diffusion) — Track A. Not deployed yet."""
    global _AFTER
    if _AFTER is None:
        print("[worker] loading AFTER engine…", flush=True)
        from after_engine import AFTEREngine  # noqa: F401  (module lands with Track A)

        _AFTER = AFTEREngine()
    return _AFTER


def _get_drums():
    """Inverse Drum Machine — Track C. Not deployed yet."""
    global _DRUMS
    if _DRUMS is None:
        print("[worker] loading Drums engine…", flush=True)
        from drum_engine import DrumEngine  # noqa: F401  (module lands with Track C)

        _DRUMS = DrumEngine()
    return _DRUMS


def _do_tone_transfer(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Engine-agnostic tone-transfer wrapper.

    Every engine implements the same contract: `render(audio: np.ndarray, sr: int, inp: dict)
    -> (out: np.ndarray, out_sr: int, extra_meta: dict)`. The wrapper handles R2 in/out;
    the engine class handles the actual re-voicing.
    """
    import soundfile as sf

    engine = inp.get("engine", "ddsp")
    audio_key = inp["audio_key"]
    output_key = inp["output_key"]
    bucket = inp.get("bucket") or R2_BUCKET_DEFAULT

    s3 = _r2_client()
    with tempfile.TemporaryDirectory() as tmp:
        local_in = str(Path(tmp) / Path(audio_key).name)
        s3.download_file(bucket, audio_key, local_in)
        audio, sr = sf.read(local_in, dtype="float32")

        if engine == "ddsp":
            instrument = inp.get("instrument", "violin")
            mode = inp.get("mode", "mono")
            # Stage 2 (poly multi-voice) not yet implemented — fall back to mono for now.
            out, osr = _get_ddsp().render_mono(audio, sr, instrument)
            extra = {"instrument": instrument, "mode": mode}
        elif engine == "after":
            # Track A — placeholder until after_engine.py + weights land.
            raise NotImplementedError(
                "AFTER engine not deployed yet (Track A). Payload was accepted for future compat."
            )
        elif engine == "drums":
            # Track C — placeholder until drum_engine.py + weights land.
            raise NotImplementedError(
                "Drums engine not deployed yet (Track C). Payload was accepted for future compat."
            )
        else:
            raise ValueError(f"unknown engine: {engine!r} (expected one of: ddsp, after, drums)")

        local_out = str(Path(tmp) / "out.flac")
        sf.write(local_out, out, osr)
        s3.upload_file(
            local_out, bucket, output_key, ExtraArgs={"ContentType": "audio/flac"}
        )

    return {"output_key": output_key, "engine": engine, "sr": osr, **extra}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    task = inp.get("task", "separate")
    try:
        if task == "ping":
            return {"pong": True, "engine_ready": ENGINE is not None}
        if task == "separate":
            return _do_separate(inp)
        if task == "tone_transfer":
            return _do_tone_transfer(inp)
        return {"error": f"unknown task: {task}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
