"""DDSP timbre-transfer engine for the RunPod worker (Stage 1: mono, audio-driven).

Re-voices a monophonic stem (vocals / bass / lead) as a target instrument, preserving the
original pitch curve + dynamics, via a DDSP autoencoder (torch, GPU). Ported from the
sweetcocoa/ddsp-pytorch model (Apache-2.0) — vendored under vendor/ddsp_pytorch with the
pre-1.8 torch FFT calls updated to torch.fft. 16 kHz model; output upsampled to 44.1 kHz.

v2 ships the pretrained **violin**. Higher-fidelity 48 kHz models (violin/sax/flute/trumpet)
come from a one-time training pass (v3) and drop into MODELS below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor", "ddsp_pytorch")
_MODELS_DIR = os.path.join(_HERE, "ddsp_models")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

# instrument -> (weights file, config yaml). See ddsp_models/README.md for provenance
# of each checkpoint, training details, and licensing.
MODELS = {
    "violin": ("violin48.pth", "violin48.yaml"),  # v3: 48kHz, own-trained (URMP), best val@step 34k
    "violin_16k_v2": ("violin.pth", "violin.yaml"),  # v2: 16kHz pretrained (sweetcocoa) — kept for A/B
}


class DDSPEngine:
    def __init__(self, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict = {}

    def available(self):
        return sorted(MODELS.keys())

    def _load(self, instrument: str):
        if instrument in self._cache:
            return self._cache[instrument]
        if instrument not in MODELS:
            raise ValueError(f"no DDSP model for '{instrument}'; have {self.available()}")
        from omegaconf import OmegaConf

        from network.autoencoder.autoencoder import AutoEncoder

        pth, yml = MODELS[instrument]
        cfg = OmegaConf.load(os.path.join(_MODELS_DIR, yml))
        net = AutoEncoder(cfg)
        state = self.torch.load(os.path.join(_MODELS_DIR, pth), map_location="cpu")
        net.load_state_dict(state, strict=False)  # smoothing_window buffer is recomputed
        net = net.to(self.device)
        # sub-modules bake a hardcoded self.device — align them to the real device
        for m in net.modules():
            if hasattr(m, "device"):
                m.device = self.device
        net.eval()
        self._cache[instrument] = (net, cfg)
        return net, cfg

    def render_mono(self, audio: np.ndarray, sr: int, instrument: str = "violin"):
        """audio: 1-D or (n,ch) float array. Returns (out_audio float32 mono, out_sr=44100)."""
        import torchaudio
        from scipy.signal import medfilt

        torch = self.torch
        net, cfg = self._load(instrument)
        msr = int(cfg.sample_rate)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        yt = torch.from_numpy(audio).unsqueeze(0)
        if sr != msr:
            yt = torchaudio.functional.resample(yt, sr, msr)
        yt = yt.to(self.device)

        with torch.no_grad():
            f0 = net.get_f0(yt, sample_rate=msr, f0_threshold=0.5)
            f0n = f0.detach().cpu().numpy()
            voiced = f0n > 0
            if voiced.sum() > 5:  # median-smooth voiced f0 to reduce warble
                f0n = np.where(voiced, medfilt(f0n, 5), 0.0)
            f0s = torch.from_numpy(f0n.astype(np.float32)).to(self.device)
            recon = net.reconstruction(yt, sample_rate=msr, add_reverb=True, f0=f0s)

        out = recon.get("audio_reverb", recon["audio_synth"]).detach().cpu().numpy().reshape(-1)
        peak = float(np.max(np.abs(out))) + 1e-9
        out = (out / peak * 0.9).astype(np.float32)

        ot = torchaudio.functional.resample(
            torch.from_numpy(out).unsqueeze(0), msr, 44100
        ).numpy().reshape(-1)
        return ot.astype(np.float32), 44100
