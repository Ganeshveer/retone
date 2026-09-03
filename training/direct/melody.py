"""Audio-signal-based melody extraction for the direct pipeline.

The MIDI-only weighted-skyline algorithm in `arrange.split_lead_accompaniment`
picks the melody note per onset cluster from Basic Pitch's transcription
using heuristics (pitch z-score, duration, velocity, register lock). It has
no idea what the actual audio recording sounded like. On real music that
sounds sparse or wrong when the melody sits below the top voice, when the
top voice is a grace note, or when arpeggios spread across octaves.

This module adds a complementary signal: run pYIN — a robust monophonic F0
tracker built into `librosa` — on the source audio. pYIN gives us "at time
t, the dominant pitched voice is around MIDI pitch p" for the whole
recording. Even on polyphonic input pYIN locks onto whichever voice is
loudest / most salient, which for pop, folk, and film-scored music is
almost always the melody.

We then combine that audio-derived melody track with the MIDI note list:
per onset cluster, pick the Basic Pitch note nearest to pYIN's pitch at
that time (with octave tolerance). Fall back to skyline when pYIN loses
lock — silence, unpitched percussion, or extremely dense passages where
no dominant voice emerges.

pYIN is already a librosa dependency — no extra install needed. On a
30-40 s WAV it runs in a few seconds on CPU. If you want higher accuracy
later, swap `extract_melody_pyin` for `extract_melody_crepe` (torchcrepe
or Google's CREPE) — the returned tuple shape is identical.
"""
from __future__ import annotations
import copy
import numpy as np
import pretty_midi

# Reuse the onset clusterer from arrange.py — keep tolerance consistent
from arrange import _cluster_onsets, CLUSTER_TOL


# ─────────────────────── audio → per-frame melody F0 ───────────────────────

def extract_melody_pyin(audio_path: str,
                        sr: int = 22050,
                        hop_length: int = 512,
                        fmin_note: str = "C2",
                        fmax_note: str = "C7",
                        hpss: bool = True):
    """Run pYIN on the audio and return a per-frame monophonic F0 track.

    Parameters
    ----------
    audio_path : path to a WAV, FLAC, or other librosa-readable file
    sr         : resample rate. 22050 is librosa's default and enough for
                 melody-register fundamentals up to ~2 kHz.
    hop_length : frames advance by this many samples. 512 @ 22050 Hz ≈
                 23 ms per frame, ~43 frames per second.
    fmin_note  : lower bound of the pYIN search. C2 (~65 Hz) skips deep
                 bass notes so pYIN latches onto the melody register.
    fmax_note  : upper bound. C7 (~2093 Hz) covers piccolo & flute range.
    hpss       : if True, run harmonic-percussive separation first so
                 drums / attacks don't distract pYIN. Cheap and worth it.

    Returns
    -------
    times      : (T,) array — frame times in seconds
    midi       : (T,) array — MIDI pitch of the tracked voice (float,
                 NaN when unvoiced)
    voiced_p   : (T,) array — voicing probability in [0, 1]. Use as a
                 confidence gate.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    if hpss:
        y, _ = librosa.effects.hpss(y)
    fmin = librosa.note_to_hz(fmin_note)
    fmax = librosa.note_to_hz(fmax_note)
    # librosa.pyin returns (f0, voiced_flag, voiced_probs); f0 is NaN when
    # the frame is unvoiced.
    f0, _voiced_flag, voiced_probs = librosa.pyin(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=2048,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
    return times, midi, voiced_probs


def _pitch_class_distance(a: float, b: float) -> float:
    """Smallest distance in semitones between two pitches, ignoring octave."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


# ─────────────────────── audio-guided lead / accomp split ──────────────────

def split_by_melody(pm,
                    audio_path: str,
                    confidence_threshold: float = 0.4,
                    tolerance_semitones: float = 2.0,
                    tol_cluster: float = CLUSTER_TOL,
                    octave_forgiveness: bool = True,
                    fallback: str = "skyline",
                    verbose: bool = False):
    """Split polyphonic MIDI into (lead, accompaniment) using pYIN as guide.

    For each onset cluster:
      1. Read pYIN's tracked pitch at the cluster's start time.
      2. If pYIN was voiced above `confidence_threshold`, pick the note in
         the cluster nearest to pYIN's pitch. Octave forgiveness ON means
         we compare pitch classes only (a real melody note that Basic
         Pitch put an octave off from pYIN's guess still lands on lead).
      3. If pYIN was unvoiced or no note in the cluster is within
         `tolerance_semitones` of the guide, fall back to `fallback`
         ("skyline" = top pitch, or "midi" for weighted-skyline).

    Parameters
    ----------
    pm                  : pretty_midi.PrettyMIDI to split.
    audio_path          : the source WAV/FLAC the MIDI was transcribed from.
    confidence_threshold: pYIN voicing probability below which we treat the
                          frame as unpitched (0.4 is a moderate gate).
    tolerance_semitones : the note picked as lead must be within this many
                          semitones of pYIN's pitch (or octave-equivalent).
    tol_cluster         : onset-cluster tolerance (defaults to arrange.py's
                          35 ms).
    octave_forgiveness  : compare pitch class (mod 12) so Basic Pitch's
                          octave errors don't disqualify melody notes.
    fallback            : "skyline" or "midi" (weighted-skyline). "skyline"
                          is the more robust fallback in practice; "midi"
                          adds hysteresis but can lock onto wrong voice.

    Returns
    -------
    lead_pm     : PrettyMIDI with only melody notes.
    accomp_pm   : PrettyMIDI with everything else.
    stats       : dict with per-run diagnostics
                  (frames_voiced, clusters_seen, clusters_pyin_matched,
                   clusters_fallback).
    """
    times, midi, voiced_p = extract_melody_pyin(audio_path)
    if verbose:
        vf = float((voiced_p >= confidence_threshold).mean())
        print(f"  pYIN: {len(times)} frames, "
              f"{vf*100:.1f}% voiced above threshold")

    lead_pm   = copy.deepcopy(pm)
    accomp_pm = copy.deepcopy(pm)

    stats = dict(frames_voiced=int((voiced_p >= confidence_threshold).sum()),
                 total_frames=int(len(times)),
                 clusters=0, matched=0, fallback=0)

    for lead_inst, accomp_inst in zip(lead_pm.instruments, accomp_pm.instruments):
        lead_kept, accomp_kept = [], []
        for g in _cluster_onsets(lead_inst.notes, tol=tol_cluster):
            stats["clusters"] += 1
            t = g[0].start

            # Look up pYIN at cluster start
            idx = int(np.searchsorted(times, t))
            idx = max(0, min(len(times) - 1, idx))
            pyin_pitch = float(midi[idx]) if idx < len(midi) else float("nan")
            pyin_conf  = float(voiced_p[idx]) if idx < len(voiced_p) else 0.0

            best_note = None
            if not np.isnan(pyin_pitch) and pyin_conf >= confidence_threshold:
                # Rank candidates by distance to pYIN pitch (with octave
                # forgiveness).
                def dist(n):
                    if octave_forgiveness:
                        return _pitch_class_distance(n.pitch, pyin_pitch)
                    return abs(n.pitch - pyin_pitch)

                closest = min(g, key=dist)
                if dist(closest) <= tolerance_semitones:
                    best_note = closest
                    stats["matched"] += 1

            if best_note is None:
                # Fallback
                stats["fallback"] += 1
                if fallback == "skyline":
                    best_note = max(g, key=lambda n: n.pitch)
                else:
                    # Simple weighted-skyline: prefer top pitch, break ties
                    # with velocity + duration.
                    best_note = max(
                        g,
                        key=lambda n: (n.pitch,
                                       n.velocity,
                                       n.end - n.start),
                    )

            lead_kept.append(best_note)
            for n in g:
                if n is not best_note:
                    accomp_kept.append(n)

        lead_inst.notes   = sorted(lead_kept,   key=lambda n: n.start)
        accomp_inst.notes = sorted(accomp_kept, key=lambda n: n.start)

    if verbose:
        c = stats["clusters"]
        m = stats["matched"]
        f = stats["fallback"]
        print(f"  split: {c} clusters, pYIN matched {m} ({100*m/max(c,1):.0f}%), "
              f"fallback {f} ({100*f/max(c,1):.0f}%)")
    return lead_pm, accomp_pm, stats
