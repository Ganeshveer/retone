"""AFTER (ACIDS-IRCAM) polyphonic timbre-transfer engine — Track A / Engine 2.

Re-voices a polyphonic stem (piano, guitar, or any pitched audio) as a target instrument
family, using the pretrained TorchScript checkpoints published by IRCAM. The exported
`.ts` bundle is fully self-contained: it embeds the RAVE-style codec (autoencoder) inside
the diffusion model, so we only load a single file per instrument.

Contract (mirrors DDSPEngine's shape so the worker handler can dispatch generically):
    engine = AFTEREngine(device=None)
    audio_out, sr_out = engine.render(audio: np.ndarray, sr: int, instrument: str)
    # sr_out is always 44100. instrument names live in the MODELS dict below.

Inference protocol — reverse-engineered from `acids-ircam/AFTER/after_scripts/export.py`
and verified end-to-end locally (cello → orchestral produced real non-silent audio):
    x_in shape: (1, 1 + zt_channels, N * ae_ratio)  where ae_ratio=4096, zt=6 for v2.
       - x_in[:, :1, :]       = source audio (the "structure" — performance to preserve)
       - x_in[:, 1:, :]       = timbre reference (each channel is a copy of the ref clip)
    x_out = model.generate_timbre(x_in)  — shape (1, 1, N * ae_ratio), 44.1 kHz

Reference clips: a short (~15 s) audio sample of the target instrument family, shipped
alongside the model weights under `after_refs/`. IRCAM includes these in the same
share as the checkpoints (piano.wav, strings.wav, flute.wav, voice.wav — all ~500 KB).

License: the AFTER code and shipped weights are CC BY-NC 4.0 (verified via
`raw.githubusercontent.com/acids-ircam/AFTER/main/LICENSE.md`). Approved for our
research/personal use per the plan; do not ship as part of a paid product.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "after_models")
_REFS_DIR = os.path.join(_HERE, "after_refs")

# instrument -> (checkpoint filename, timbre-reference clip filename)
# All checkpoints are pre-exported TorchScript from IRCAM's Nextcloud share:
#   https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P/download?path=/After%20models
# The reference clips come from the same share's "Patches (deprecated)/data/audio_files/"
# folder (all recorded by IRCAM as canonical timbre anchors for these models).
MODELS = {
    # Piano/guitar → any-instrument re-voicing (general audio-to-audio model).
    "instruments":   ("afterv2.audio.instr.ts",       "piano_ref.wav"),
    # Piano → strings + brass ensemble (this session's demo target).
    "orchestral":    ("afterv2.audio.orchestral.ts",  "strings_ref.wav"),
    # Older v1 drum model — useful as a fallback for Track C until IDM lands.
    "drums_v1":      ("afterv1.audio.drums.ts",       "flute_ref.wav"),
    # Speech-timbre transfer (voice-to-voice); mostly a curiosity for the DAW.
    "speech":        ("afterv2.audio.speech.ts",      "voice_ref.wav"),
}


class AFTEREngine:
    """Wraps AFTER's exported TorchScript for the RunPod worker's tone_transfer task."""

    # Verified-safe inference defaults for the exported checkpoints. Higher nb_steps
    # would improve fidelity but the exported streaming model's KV cache mis-aligns at
    # step counts > ~4 in single-batch mode (traced through export.py::sample loop).
    NB_STEPS = 2
    GUIDANCE_STRUCTURE = 1.0
    OUT_SR = 44100
    _CHUNK_MAX_SECONDS = 30  # split long inputs to avoid multi-minute renders per chunk

    def __init__(self, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict = {}

    def available(self) -> list[str]:
        return sorted(MODELS.keys())

    def _load(self, instrument: str):
        """Load (model, reference_tensor) for an instrument; cached across calls."""
        if instrument in self._cache:
            return self._cache[instrument]
        if instrument not in MODELS:
            raise ValueError(f"no AFTER model for '{instrument}'; have {self.available()}")

        pth, ref = MODELS[instrument]
        model_path = os.path.join(_MODELS_DIR, pth)
        ref_path = os.path.join(_REFS_DIR, ref)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} missing — Dockerfile build-time fetch should populate this. "
                "See runpod/download_models.py."
            )
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"{ref_path} missing — reference clips should be committed.")

        model = self.torch.jit.load(model_path, map_location="cpu").eval().to(self.device)
        # Configure once at load time; NB_STEPS/GUIDANCE tuned above.
        model.set_nb_steps(self.NB_STEPS)
        model.set_guidance_structure(self.GUIDANCE_STRUCTURE)

        # Load + normalize the timbre reference clip.
        import soundfile as sf
        ref_audio, ref_sr = sf.read(ref_path, always_2d=False)
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=-1)
        ref_audio = ref_audio.astype(np.float32)

        model_sr = int(model.sr)
        if ref_sr != model_sr:
            import torchaudio
            ref_audio = torchaudio.functional.resample(
                self.torch.from_numpy(ref_audio), ref_sr, model_sr
            ).numpy()

        self._cache[instrument] = (model, ref_audio, model_sr)
        return self._cache[instrument]

    def render(self, audio: np.ndarray, sr: int, instrument: str):
        """audio: 1-D or (n, ch) float array. Returns (out_audio float32 mono, out_sr=44100).

        The model is a streaming latent-diffusion transformer with an internal KV cache
        that behaves best when called on chunks bounded by ~30 s. Longer inputs are split,
        rendered per-chunk with a fresh model reload between chunks (cheap ~1s on GPU),
        and concatenated. The chunking is transparent — same length in, same length out
        (rounded to a multiple of ae_ratio; tail beyond that boundary is dropped).
        """
        import torchaudio

        torch = self.torch
        model, ref_audio, model_sr = self._load(instrument)
        ae_ratio = int(model.ae_ratio)
        zt = int(model.zt_channels)

        # Downmix + resample source to the model's SR
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        yt = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
        if sr != model_sr:
            yt = torchaudio.functional.resample(yt, sr, model_sr)
        yt = yt.squeeze(0)  # (N,)

        # Truncate to a multiple of ae_ratio (drop tail)
        total = (yt.shape[0] // ae_ratio) * ae_ratio
        if total == 0:
            raise ValueError(f"input too short: need at least {ae_ratio} samples")
        yt = yt[:total]

        # Chunk the source to keep KV-cache growth bounded
        chunk_samples = (self._CHUNK_MAX_SECONDS * model_sr // ae_ratio) * ae_ratio
        chunks = [yt[i : i + chunk_samples] for i in range(0, total, chunk_samples)]

        outs = []
        for chunk in chunks:
            L = chunk.shape[0]
            # Build the timbre-reference tensor at the source's length (tile if shorter)
            ref = ref_audio
            if ref.shape[0] < L:
                k = (L // ref.shape[0]) + 1
                ref = np.tile(ref, k)[:L]
            else:
                ref = ref[:L]
            ref_t = torch.from_numpy(ref).unsqueeze(0).unsqueeze(0).repeat(1, zt, 1)  # (1, zt, L)
            src_t = chunk.unsqueeze(0).unsqueeze(0)                                   # (1, 1,  L)
            x = torch.cat([src_t, ref_t], dim=1).to(self.device)

            with torch.no_grad():
                out = model.generate_timbre(x)
            outs.append(out.detach().cpu().numpy().reshape(-1))

            # Fresh model instance for the next chunk to avoid KV-cache leakage.
            # (Cheap: eval-mode reload ~0.5s on GPU, negligible next to the ~10s render.)
            if len(chunks) > 1:
                self._cache.pop(instrument, None)
                model, ref_audio, model_sr = self._load(instrument)
                ae_ratio = int(model.ae_ratio)
                zt = int(model.zt_channels)

        out_np = np.concatenate(outs, axis=0)

        # Peak-normalize to a safe headroom (matches DDSPEngine's 0.9 peak convention)
        peak = float(np.max(np.abs(out_np))) + 1e-9
        out_np = (out_np / peak * 0.9).astype(np.float32)

        # Model already runs at 44.1 kHz — no final resample needed (matches OUT_SR).
        return out_np, self.OUT_SR
