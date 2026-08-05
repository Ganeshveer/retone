"""Per-stem note detection + musical analysis.

Mock/local path: runs librosa.pyin on the stored stem file (CPU, no GPU) and segments the
frame-level f0 into discrete note events. Validated end-to-end on the synthesized demo
melody. Real/RunPod path (later) will use Basic Pitch (polyphonic) / CREPE on the worker.

librosa is imported lazily so it doesn't slow backend startup.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# Krumhansl-Kessler key profiles (tonic at index 0).
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_SR = 22050
_HOP = 256
_FRAME = 2048


def _segment_notes(f0, voiced_flag, voiced_probs, times, midi,
                   pitch_tol: float = 0.6, min_dur: float = 0.05,
                   max_gap: float = 0.06, conf_thresh: float = 0.5) -> List[dict]:
    """Group consecutive voiced frames of similar pitch into note events, with hysteresis
    (running-median reference so vibrato doesn't over-split), gap bridging, and a
    min-duration drop. Returns [{start, dur, midi, confidence}, ...]."""
    import numpy as np

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    active = (
        np.asarray(voiced_flag)
        & (np.asarray(voiced_probs) >= conf_thresh)
        & np.isfinite(midi)
    )

    notes: List[dict] = []
    n = len(midi)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start_i = i
        last_active = i
        ref = midi[i]
        pitches = [midi[i]]
        confs = [voiced_probs[i]]
        gap = 0
        j = i + 1
        while j < n:
            if active[j] and abs(midi[j] - ref) <= pitch_tol:
                pitches.append(midi[j])
                confs.append(voiced_probs[j])
                ref = float(np.median(pitches[-9:]))
                last_active = j
                gap = 0
                j += 1
            else:
                gap += 1
                if gap * dt > max_gap:
                    break
                j += 1
        dur = float(times[last_active] - times[start_i] + dt)
        if dur >= min_dur:
            notes.append({
                "start": round(float(times[start_i]), 4),
                "dur": round(dur, 4),
                "midi": int(round(float(np.median(pitches)))),
                "confidence": round(float(np.clip(np.mean(confs), 0.0, 1.0)), 3),
            })
        i = last_active + 1
    return notes


def _estimate_key(y, sr) -> Optional[str]:
    import numpy as np
    import librosa

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP)
        profile = chroma.mean(axis=1)
        if not np.any(profile):
            return None
        best_score = -np.inf
        best_key = None
        maj = np.asarray(_KS_MAJOR)
        minr = np.asarray(_KS_MINOR)
        for tonic in range(12):
            r_maj = np.corrcoef(profile, np.roll(maj, tonic))[0, 1]
            r_min = np.corrcoef(profile, np.roll(minr, tonic))[0, 1]
            if r_maj > best_score:
                best_score, best_key = r_maj, f"{_PITCH_CLASSES[tonic]} major"
            if r_min > best_score:
                best_score, best_key = r_min, f"{_PITCH_CLASSES[tonic]} minor"
        return best_key
    except Exception:
        return None


def _estimate_bpm(y, sr) -> Optional[float]:
    import numpy as np
    import librosa

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=_HOP)
        bpm = float(np.atleast_1d(tempo)[0])  # scalar in 0.10.x, ndarray in 0.11
        return round(bpm, 1) if bpm and bpm > 1 else None
    except Exception:
        return None


def analyze_stem(path: str) -> Tuple[List[dict], Optional[float], Optional[str], float]:
    """Analyze one stem file. Returns (notes, bpm, key, duration_seconds)."""
    import numpy as np
    import librosa

    y, sr = librosa.load(path, sr=_SR, mono=True)
    duration = float(len(y) / sr)

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=_FRAME, hop_length=_HOP
    )
    times = librosa.times_like(f0, sr=sr, hop_length=_HOP)
    midi = librosa.hz_to_midi(f0)

    notes = _segment_notes(f0, voiced_flag, voiced_probs, times, midi)
    bpm = _estimate_bpm(y, sr)
    key = _estimate_key(y, sr)
    return notes, bpm, key, round(duration, 3)
