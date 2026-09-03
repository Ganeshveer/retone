"""Direct instrument-conversion pipeline: audio → MIDI → arrange → SF2 → WAV.

    audio → transcribe(model) → PrettyMIDI → arrange(target) → FluidSynth(SF2) → WAV [+ light reverb]

No ML model at render time; only Basic Pitch (or ByteDance for piano, or
Transkun) at the transcription stage. Timbre fidelity is limited only by the
chosen SF2.

Aug 2026 — polyphony awareness. When the target is monophonic (violin_solo,
trumpet_solo, flute_solo, sax, …) and the source is polyphonic, the pipeline
splits into a lead (top-pitch skyline) rendered on the target and an
accompaniment (everything below) rendered on a complementary poly instrument.
The accompaniment is either user-supplied (`--accompaniment ...`), catalog-
provided (`Instrument.default_accompaniment`), or auto-picked with a musical
"color-wheel" scorer. When the source is already monophonic (a vocal, solo
flute, solo violin) the split is skipped and the accompaniment is ignored —
there is no polyphonic content to route.

Plucked / short-decay targets (harp, guitar, harpsichord, marimba, pizzicato)
carry a per-instrument `min_ioi_s` in the catalog; the renderer thins their
retriggered notes via `arrange.enforce_min_ioi` before rendering. See
`arrange.py` for the primitives and the plan file for the numeric defaults.

Usage:
    python render_direct.py --input song.wav --instrument violin_solo \\
        --out out.wav --transcriber basic_pitch

    # Force a specific accompaniment instead of the auto-pick:
    python render_direct.py --input song.wav --instrument violin_solo \\
        --accompaniment guitar_nylon --out out.wav

    # Disable accompaniment entirely (mono target, no split):
    python render_direct.py --input song.wav --instrument violin_solo \\
        --accompaniment none --out out.wav

See instruments.py for the full catalog (~70 instruments across 10 categories).
"""
from __future__ import annotations
import argparse, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, librosa, soundfile as sf, pretty_midi
from scipy.signal import fftconvolve

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "poly"))                     # dataprep.py lives here
sys.path.insert(0, "/workspace/retone_poly/lib")                   # pod layout fallback

# dataprep pulls in meldataset + torch — imported lazily so the module can be
# introspected in environments without the training deps installed (tests).
_dp = None
def _get_dp():
    global _dp
    if _dp is None:
        import dataprep
        _dp = dataprep
    return _dp

from instruments import INSTRUMENTS, Instrument, by_category, list_categories
from arrange import (ARRANGERS, clamp_to_range,
                     enforce_min_ioi, split_lead_accompaniment,
                     apply_mono_legato)
try:
    from melody import split_by_melody as _split_by_melody
    _HAVE_PYIN_SPLIT = True
except Exception:
    _HAVE_PYIN_SPLIT = False


# ────────────────────────── transcribers ───────────────────────────────────

def transcribe_basic_pitch(audio_path, onset=0.3, frame=0.2, min_len=40):
    """Spotify's Basic Pitch. Fast, small (~20 MB), Apache-2.0. Slightly loose
    thresholds by default (recall > precision) — the arrangement stage cleans
    up dense output."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(audio_path, ICASSP_2022_MODEL_PATH,
                         onset_threshold=onset, frame_threshold=frame,
                         minimum_note_length=min_len)
    return midi


def transcribe_bytedance_piano(audio_path):
    """ByteDance PianoTranscription — piano-only, real velocity + pedal, 96.72
    F1 on MAPS. Requires `pip install piano_transcription_inference` (~200 MB
    weights auto-downloaded on first call). Runs on GPU if available.

    Reference: `arXiv:2010.01815`.
    """
    from piano_transcription_inference import PianoTranscription, sample_rate
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device="cuda" if _has_cuda() else "cpu",
                                       checkpoint_path=None)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        out_mid = f.name
    transcriptor.transcribe(y, out_mid)
    pm = pretty_midi.PrettyMIDI(out_mid)
    os.unlink(out_mid)
    return pm


def transcribe_transkun(audio_path):
    """Transkun (Yan, ISMIR 2024, MIT) — piano-only, ~97.5 F1 on MAESTRO,
    best subjective quality of the pip-installable piano transcribers as of
    late 2025. Requires `pip install transkun` (weights auto-downloaded).
    Shells out to the `transkun` CLI. Accepts WAV only — convert OGG/FLAC/MP3
    to WAV first with librosa.

    Reference: [Transkun GitHub](https://github.com/Yujia-Yan/Transkun).
    """
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        out_mid = f.name
    subprocess.run(["transkun", audio_path, out_mid], check=True,
                   capture_output=True)
    pm = pretty_midi.PrettyMIDI(out_mid)
    os.unlink(out_mid)
    return pm


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


TRANSCRIBERS = {
    "basic_pitch":       transcribe_basic_pitch,        # any polyphonic, ~85 F1, always available
    "bytedance_piano":   transcribe_bytedance_piano,    # piano only, 96.72 F1, faster on GPU
    "transkun":          transcribe_transkun,           # piano only, ~97.5 F1 (best on clean), slower
}


def choose_transcriber(input_audio: str, source_kind: str | None = None) -> str:
    """Pick the best transcriber for the source. Piano sources get ByteDance
    (F1 96.72 vs Basic Pitch ~85, with real velocity + pedal); anything else
    gets Basic Pitch.

    `source_kind` (optional) short-circuits the heuristic:
        "piano"   -> bytedance_piano (or transkun if you know it's clean)
        "poly"    -> basic_pitch (any polyphonic non-piano)
        "mono"    -> basic_pitch (works fine for solo instruments too)

    When `source_kind` is None, we fall back to a filename hint: names
    containing 'piano', 'pianoman', 'elise', 'canon' bias to the piano
    branch. This is intentionally cheap — the audio-classifier idea would
    add a big dependency to save one call.
    """
    if source_kind == "piano":
        return "bytedance_piano"
    if source_kind in ("poly", "mono"):
        return "basic_pitch"
    name = os.path.basename(input_audio).lower()
    piano_hints = ("piano", "pianoman", "elise", "canon", "chopin",
                   "beethoven", "mozart_piano", "rachmaninoff")
    for h in piano_hints:
        if h in name:
            return "bytedance_piano"
    return "basic_pitch"


# ────────────────────────── render + polish ────────────────────────────────

def render_sf2(pm, sf2_path, program, seconds, out_wav):
    """Feed the (arranged) PrettyMIDI through FluidSynth via dataprep.render."""
    dp = _get_dp()
    orig_sf, orig_prog = dp.SF2, dp.PROGRAMS.get("piano", 0)
    try:
        dp.SF2 = sf2_path
        dp.PROGRAMS["piano"] = program
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp_mid = f.name
        pm.write(tmp_mid)
        dp.render(tmp_mid, "piano", out_wav, max_seconds=seconds)
        os.unlink(tmp_mid)
    finally:
        dp.SF2, dp.PROGRAMS["piano"] = orig_sf, orig_prog


def synth_hall_ir(sr=44100, seconds=1.0, decay=3.0, seed=42):
    """Cheap exp-decaying dense-noise IR — small hall / plate approximation.

    Seeded by default (`seed=42`) so every render of the same source produces
    an identical reverb tail. Two runs of the same source now byte-diff only
    if the render pipeline itself changed. Pass `seed=None` for the old
    behavior (random every call).
    """
    n = int(sr * seconds)
    rng = np.random.default_rng(seed) if seed is not None else np.random
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-decay * np.arange(n) / n)
    return ir / (np.abs(ir).max() or 1)


def apply_light_reverb(wav_path, wet=0.15, ir=None):
    """Mix a small amount of hall reverb into a dry fluidsynth render."""
    y, sr = sf.read(wav_path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if ir is None:
        ir = synth_hall_ir(sr=sr)
    wet_signal = fftconvolve(y, ir, mode="full")[: len(y)]
    wet_signal = wet_signal / (np.abs(wet_signal).max() or 1) * (np.abs(y).max() or 1)
    mixed = (1 - wet) * y + wet * wet_signal
    peak = float(np.abs(mixed).max() or 1)
    sf.write(wav_path, (mixed / peak * 0.9).astype(np.float32), sr)


def _mix_and_write(a_wav, b_wav, out_wav, a_gain=1.0, b_gain=0.35):
    """Weighted sum of two mono renders → normalized WAV.

    Default `b_gain=0.35` (~-9 dB below the lead) matches standard
    orchestration practice — soloist over ensemble sits ~8-12 dB above
    the accompaniment bed. The previous default (-5 dB) put the accomp
    close enough to the lead that harp/rhodes/guitar muddied every mix.
    """
    ya, sr_a = sf.read(a_wav)
    yb, sr_b = sf.read(b_wav)
    if ya.ndim == 2: ya = ya.mean(axis=1)
    if yb.ndim == 2: yb = yb.mean(axis=1)
    assert sr_a == sr_b, f"sample-rate mismatch: {sr_a} vs {sr_b}"
    n = max(len(ya), len(yb))
    if len(ya) < n: ya = np.pad(ya, (0, n - len(ya)))
    if len(yb) < n: yb = np.pad(yb, (0, n - len(yb)))
    mixed = a_gain * ya + b_gain * yb
    peak = float(np.abs(mixed).max() or 1)
    sf.write(out_wav, (mixed / peak * 0.9).astype(np.float32), sr_a)


# ────────────────────────── polyphony helpers ──────────────────────────────

def peak_simultaneous_voices(pm) -> int:
    """Max number of overlapping notes anywhere in `pm`. 1 = monophonic."""
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append((n.start, +1))
            events.append((n.end,   -1))
    if not events:
        return 0
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def polyphony_profile(pm, dt: float = 0.010) -> dict:
    """Time-weighted polyphony stats. More robust than peak alone.

    Returns dict with:
      peak         : max concurrent notes anywhere
      p90, p50     : 90th / 50th percentile of concurrent notes during ACTIVE
                     windows (t where at least one note is sounding)
      mean_active  : mean concurrent notes during active windows
      active_s     : total seconds with at least one note sounding

    Rationale: transcribers (esp. Basic Pitch on monophonic sources) produce
    brief spurious overlaps that inflate `peak` — a solo violin can transcribe
    with peak=4 despite being ~99% single-voice. Percentile stats are
    resistant to that noise.
    """
    import numpy as np
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append((n.start, +1))
            events.append((n.end,   -1))
    if not events:
        return dict(peak=0, p90=0, p50=0, mean_active=0.0, active_s=0.0)
    events.sort()
    t_min = events[0][0]
    t_max = events[-1][0]
    if t_max - t_min < dt:
        return dict(peak=0, p90=0, p50=0, mean_active=0.0, active_s=0.0)

    grid = np.arange(t_min, t_max, dt)
    poly = np.zeros(len(grid), dtype=np.int32)
    cur, idx = 0, 0
    for i, t in enumerate(grid):
        while idx < len(events) and events[idx][0] <= t:
            cur += events[idx][1]
            idx += 1
        poly[i] = cur
    active = poly[poly > 0]
    if active.size == 0:
        return dict(peak=int(poly.max()), p90=0, p50=0, mean_active=0.0, active_s=0.0)
    return dict(
        peak        = int(poly.max()),
        p90         = int(np.percentile(active, 90)),
        p50         = int(np.percentile(active, 50)),
        mean_active = float(active.mean()),
        active_s    = float(active.size * dt),
    )


def source_is_polyphonic(pm, p90_threshold: int = 3, p50_threshold: int = 2) -> bool:
    """True if the source has real polyphonic content (not transcription noise).

    Uses `polyphony_profile` and requires BOTH:
      - 90th-percentile active polyphony ≥ p90_threshold (default 3)
      - 50th-percentile active polyphony ≥ p50_threshold (default 2)

    Percentiles beat peak because transcribers produce brief spurious
    overlaps on monophonic sources. The two-gate design catches vocal /
    solo-wind stems whose transcriptions have a high p90 (from breathy
    artifacts) but a low p50 (mostly single-voice).
    """
    prof = polyphony_profile(pm)
    return prof["p90"] >= p90_threshold and prof["p50"] >= p50_threshold


# Attack / family tags for the auto-accompaniment picker. Category can't
# express both attack character and instrument family, so we tag names/cats
# explicitly here. Kept short — hand-editable.
_ATTACK_TAG: dict[str, str] = {
    # percussive/plucked: struck once, decays
    **{n: "percussive" for n in [
        "piano", "keys", "guitar", "harp"
    ]},                                                # by category
    # sustained: bowed, blown, or drawn indefinitely
    **{n: "sustained" for n in [
        "organ", "strings", "brass", "wind", "voice", "synth"
    ]},
}
# Overrides that break category defaults
_ATTACK_OVERRIDE: dict[str, str] = {
    "strings_pizzicato":   "percussive",
    "violin1_pizzicato":   "percussive",
    "violin1_staccato":    "percussive",
    "viola_pizzicato":     "percussive",
    "cello_pizzicato":     "percussive",
    "bass_pizzicato":      "percussive",
    "bass_acoustic":       "percussive",
    "bass_electric_fingered": "percussive",
    "bass_electric_picked": "percussive",
}

_FAMILY_TAG: dict[str, str] = {
    "piano":  "keys",   "keys":  "keys",
    "organ":  "organ",
    "guitar": "plucked","harp":  "plucked", "bass": "plucked",
    "strings": "bowed",
    "brass": "brass",
    "wind":  "blown",
    "voice": "voice",
    "synth": "synth",
}


def _attack(inst: Instrument) -> str:
    return _ATTACK_OVERRIDE.get(inst.name, _ATTACK_TAG.get(inst.category, "sustained"))


def _family(inst: Instrument) -> str:
    return _FAMILY_TAG.get(inst.category, inst.category)


# Deterministic tiebreak — hand-picked safe defaults, most preferred first.
_ACCOMPANIMENT_PREF = [
    "harp_sonatina", "harp_fluidr3",
    "piano_grand", "piano_bright", "piano_ep1_rhodes",
    "guitar_nylon", "guitar_steel",
    "organ_church", "pad_warm",
]


def pick_accompaniment(lead: Instrument, require_sf2: bool = True) -> str:
    """Musical color-wheel scorer. Picks a complementary poly instrument.

    Score components (additive):
      +3  opposite attack character (percussive vs sustained) — the biggest
          musical contrast; strings under harp, wind under piano.
      +2  different family (keys / plucked / organ / bowed / brass / blown / voice / synth)
      +1  wide register (range_hi - range_lo >= 60) — piano/harp/organ can
          voice bass and mid-chords in the same instrument.
      -1  category ∈ {synth, voice} — those make weak comping beds.
      -3  same category as lead — never pair violin_solo with strings_ensemble_1.

    Returns the slug (a key of INSTRUMENTS). Falls back to piano_grand if no
    candidate has an existing SF2.
    """
    lead_attack = _attack(lead)
    lead_family = _family(lead)

    def score(i: Instrument) -> tuple[int, int]:
        s = 0
        if _attack(i) != lead_attack:      s += 3
        if _family(i) != lead_family:      s += 2
        if (i.range_hi - i.range_lo) >= 60: s += 1
        if i.category in {"synth", "voice"}: s -= 1
        if i.category == lead.category:    s -= 3
        # deterministic secondary rank: earlier in the preferred list wins
        pref = _ACCOMPANIMENT_PREF.index(i.name) if i.name in _ACCOMPANIMENT_PREF else 99
        return (s, -pref)  # higher score first, then earlier-preferred

    candidates = [i for i in INSTRUMENTS.values()
                  if i.polyphony == "poly" and i.name != lead.name]
    if require_sf2:
        candidates = [i for i in candidates if os.path.exists(i.sf2)]
    if not candidates:
        return "piano_grand"                # FluidR3 is always shipped
    return max(candidates, key=score).name


# ────────────────────────── entrypoint ─────────────────────────────────────

def render_one(input_audio, instrument_name, out_wav, transcriber="basic_pitch",
               seconds=None, reverb_wet=0.15,
               accompaniment_name: str | None = None,
               source_poly_p90: int = 3,
               verbose=True):
    """Render one input through one target instrument.

    Arguments beyond the original signature:
      accompaniment_name : instrument slug, "none", or None (auto). Only used
                           when the target is monophonic AND the source is
                           polyphonic; otherwise silently ignored.
      source_poly_p90    : 90th-percentile active polyphony ≥ this = source
                           is poly. Default 3. Percentile beats peak because
                           transcribers produce brief spurious overlaps on
                           monophonic sources.
    """
    if instrument_name not in INSTRUMENTS:
        raise KeyError(f"unknown instrument {instrument_name!r}; "
                       f"see instruments.py or --list")
    lead = INSTRUMENTS[instrument_name]
    if not os.path.exists(lead.sf2):
        raise FileNotFoundError(f"SF2 missing for {instrument_name}: {lead.sf2}")

    tr = TRANSCRIBERS[transcriber]
    if verbose:
        print(f"  transcribe ({transcriber}) …")
    t0 = time.time()
    pm = tr(input_audio)
    _get_dp().apply_sustain(pm)
    # Strip CC 64 (sustain pedal) events: apply_sustain has already extended
    # note ends to pedal-release. Leaving the CCs makes FluidSynth honor the
    # pedal a second time, giving ~2x too long ringing tails.
    for inst in pm.instruments:
        inst.control_changes = [c for c in inst.control_changes if c.number != 64]
    n_notes = sum(len(i.notes) for i in pm.instruments)
    prof = polyphony_profile(pm)
    if verbose:
        print(f"    {n_notes} notes, peak={prof['peak']}, "
              f"p90={prof['p90']}, mean_active={prof['mean_active']:.1f} "
              f"in {time.time()-t0:.0f}s")

    if seconds:
        for inst_pm in pm.instruments:
            inst_pm.notes = [n for n in inst_pm.notes if n.start < seconds]
            for n in inst_pm.notes:
                n.end = min(n.end, seconds)

    # ── decide split ────────────────────────────────────────────────────
    src_is_poly  = prof["p90"] >= source_poly_p90 and prof["p50"] >= 2
    want_accomp  = (accompaniment_name != "none")
    do_split     = (lead.polyphony == "mono" and src_is_poly and want_accomp)

    accomp: Instrument | None = None
    if do_split:
        if accompaniment_name and accompaniment_name not in ("auto", "none"):
            if accompaniment_name not in INSTRUMENTS:
                raise KeyError(f"unknown accompaniment {accompaniment_name!r}")
            accomp = INSTRUMENTS[accompaniment_name]
        elif lead.default_accompaniment:
            accomp = INSTRUMENTS[lead.default_accompaniment]
            if not os.path.exists(accomp.sf2):     # fall back if not on disk
                accomp = INSTRUMENTS[pick_accompaniment(lead)]
        else:
            accomp = INSTRUMENTS[pick_accompaniment(lead)]

        # Prefer audio-guided melody extraction (pYIN) when we have the
        # source audio and the librosa/melody module is available. Fall
        # back to weighted-skyline if pYIN raises or gives no coverage.
        used_pyin = False
        if _HAVE_PYIN_SPLIT and os.path.exists(input_audio):
            try:
                lead_pm, accomp_pm, stats = _split_by_melody(
                    pm, input_audio, verbose=verbose,
                )
                # If pYIN was voiced on almost none of the audio, its
                # per-cluster picks are basically random — use skyline
                # instead. 10 % voiced coverage is the guard.
                voiced_frac = (stats["frames_voiced"]
                               / max(stats["total_frames"], 1))
                if voiced_frac >= 0.10:
                    used_pyin = True
                else:
                    if verbose:
                        print(f"  pYIN voiced coverage {voiced_frac:.0%} — "
                              f"falling back to weighted-skyline")
            except Exception as e:
                if verbose:
                    print(f"  pYIN split failed ({e.__class__.__name__}: {e}) "
                          f"— falling back to weighted-skyline")

        if not used_pyin:
            lead_pm, accomp_pm = split_lead_accompaniment(pm)

        if verbose:
            n_lead   = sum(len(i.notes) for i in lead_pm.instruments)
            n_accomp = sum(len(i.notes) for i in accomp_pm.instruments)
            algo     = "pYIN" if used_pyin else "weighted-skyline"
            print(f"  split ({algo}) lead / accomp: {n_lead} / {n_accomp} "
                  f"notes — accomp = {accomp.name}")
    else:
        lead_pm, accomp_pm = pm, None
        if lead.polyphony == "mono" and not src_is_poly and verbose:
            print(f"  target is mono but source is monophonic "
                  f"(p90={prof['p90']}) — no accompaniment")
        elif lead.polyphony == "mono" and not want_accomp and verbose:
            print(f"  accompaniment disabled by --accompaniment none")

    # ── per-track passes ────────────────────────────────────────────────
    # For narrow-range MONO targets on a poly source we prefer to drop
    # out-of-range notes rather than fold them by octaves — the
    # accompaniment carries the bass; folding creates audible octave
    # jumps in the melody. For a wide-range poly target playing solo the
    # octave fold is preferable (no accomp to catch the note).
    lead_clamp_mode = "drop" if accomp is not None else "octave"
    lead_pm = clamp_to_range(lead_pm, lead.range_lo, lead.range_hi,
                             mode=lead_clamp_mode)
    if lead.min_ioi_s > 0:
        n_before = sum(len(i.notes) for i in lead_pm.instruments)
        # Lead: only per-pitch thinning (drop same-pitch re-strikes). Skip
        # the global 14-onsets/s cap so a soft skyline melody note is
        # never dropped in favor of louder inner voices of the accomp.
        lead_pm = enforce_min_ioi(lead_pm, lead.min_ioi_s,
                                  global_max_onsets=None)
        n_after  = sum(len(i.notes) for i in lead_pm.instruments)
        if verbose and n_after < n_before:
            print(f"  density limit (lead, min_ioi={lead.min_ioi_s:.2f}s): "
                  f"{n_before} -> {n_after} notes")
    lead_pm = ARRANGERS[lead.arranger](lead_pm)
    # Mono-mode CC only fires when we actually split — a bare mono target on
    # a mono source is played by the SF2 as-is; asserting mono CC on raw
    # Basic Pitch output causes transcription overlaps to silence real notes.
    if lead.polyphony == "mono" and accomp is not None:
        lead_pm = apply_mono_legato(lead_pm)

    if accomp is not None:
        accomp_pm = clamp_to_range(accomp_pm, accomp.range_lo, accomp.range_hi)
        if accomp.min_ioi_s > 0:
            n_before = sum(len(i.notes) for i in accomp_pm.instruments)
            accomp_pm = enforce_min_ioi(accomp_pm, accomp.min_ioi_s)
            n_after  = sum(len(i.notes) for i in accomp_pm.instruments)
            if verbose and n_after < n_before:
                print(f"  density limit (accomp, min_ioi={accomp.min_ioi_s:.2f}s): "
                      f"{n_before} -> {n_after} notes")
        accomp_pm = ARRANGERS[accomp.arranger](accomp_pm)

    # ── render ─────────────────────────────────────────────────────────
    if verbose:
        print(f"  render {lead.display} via {os.path.basename(lead.sf2)} …")

    if accomp is None:
        render_sf2(lead_pm, lead.sf2, lead.program, seconds or 300, out_wav)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fa:
            tmp_lead = fa.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fb:
            tmp_accomp = fb.name
        try:
            render_sf2(lead_pm, lead.sf2, lead.program, seconds or 300, tmp_lead)
            if verbose:
                print(f"  render {accomp.display} via {os.path.basename(accomp.sf2)} …")
            render_sf2(accomp_pm, accomp.sf2, accomp.program, seconds or 300, tmp_accomp)
            _mix_and_write(tmp_lead, tmp_accomp, out_wav,
                           a_gain=1.0, b_gain=0.35)
        finally:
            for f in (tmp_lead, tmp_accomp):
                if os.path.exists(f):
                    os.unlink(f)

    if reverb_wet > 0:
        apply_light_reverb(out_wav, wet=reverb_wet)
    if verbose:
        print(f"  -> {out_wav}  ({os.path.getsize(out_wav)/1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Direct instrument-conversion pipeline")
    ap.add_argument("--input", help="input audio file")
    ap.add_argument("--instrument", help="target instrument (see --list)")
    ap.add_argument("--category", help="render into every instrument of this category (batch)")
    ap.add_argument("--out",     help="output WAV path (single-instrument mode)")
    ap.add_argument("--outdir",  help="output directory (category mode)")
    ap.add_argument("--transcriber", default="basic_pitch",
                    choices=list(TRANSCRIBERS.keys()),
                    help="basic_pitch (any polyphonic) | bytedance_piano | transkun (both piano-only, higher F1)")
    ap.add_argument("--seconds", type=int, default=None,
                    help="clip render to first N seconds of the input")
    ap.add_argument("--reverb-wet", type=float, default=0.15,
                    help="0.0 = dry, 0.3 = wet. Default 0.15.")
    ap.add_argument("--accompaniment", default="auto",
                    help="accompaniment instrument slug, or 'auto' (default) "
                         "or 'none' to disable. Only applies to mono targets "
                         "on poly sources.")
    ap.add_argument("--list", action="store_true",
                    help="print the instrument catalog and exit")
    args = ap.parse_args()

    if args.list:
        for cat in list_categories():
            entries = by_category(cat)
            print(f"\n═══ {cat.upper()} ({len(entries)}) ═══")
            for i in entries:
                tag = f"[{i.polyphony}]" if i.polyphony == "mono" else "     "
                ioi = f"  ioi={i.min_ioi_s:.2f}" if i.min_ioi_s > 0 else ""
                print(f"  {tag} {i.name:32s} {i.display}{ioi}")
        print(f"\nTOTAL: {len(INSTRUMENTS)} instruments")
        return

    if not args.input:
        ap.error("--input is required (except with --list)")

    if args.category:
        outdir = args.outdir or "renders"
        os.makedirs(outdir, exist_ok=True)
        entries = by_category(args.category)
        if not entries:
            ap.error(f"no instruments in category {args.category!r}; "
                     f"choose from: {list_categories()}")
        print(f"batch: {len(entries)} instruments in category {args.category!r}")
        for inst in entries:
            out = os.path.join(outdir, f"{Path(args.input).stem}__{inst.name}.wav")
            print(f"\n[{inst.name}]")
            try:
                render_one(args.input, inst.name, out,
                           transcriber=args.transcriber,
                           seconds=args.seconds, reverb_wet=args.reverb_wet,
                           accompaniment_name=args.accompaniment)
            except Exception as e:
                print(f"  FAIL: {e}")
    else:
        if not args.instrument or not args.out:
            ap.error("--instrument and --out required (or --category and --outdir)")
        render_one(args.input, args.instrument, args.out,
                   transcriber=args.transcriber,
                   seconds=args.seconds, reverb_wet=args.reverb_wet,
                   accompaniment_name=args.accompaniment)


if __name__ == "__main__":
    main()
