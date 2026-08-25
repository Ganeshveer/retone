"""Direct instrument-conversion pipeline: audio → MIDI → arrange → SF2 → WAV.

    audio → transcribe(model) → PrettyMIDI → arrange(target) → FluidSynth(SF2) → WAV [+ light reverb]

No ML model at render time; only Basic Pitch (or ByteDance for piano) at the
transcription stage. Timbre fidelity is limited only by the chosen SF2.

Usage:
    python render_direct.py --input song.wav --instrument violin1_sustain \\
        --out out.wav --transcriber basic_pitch

    # Best piano quality: use the ByteDance piano transcriber
    pip install piano_transcription_inference
    python render_direct.py --input piano_song.wav --instrument piano_grand \\
        --out out.wav --transcriber bytedance_piano

    # Batch: render one input through every instrument in a category
    python render_direct.py --input song.wav --category strings --outdir renders/

See instruments.py for the full catalog (~65 instruments across 10 categories).
"""
from __future__ import annotations
import argparse, os, sys, tempfile, time
from pathlib import Path
import numpy as np, librosa, soundfile as sf, pretty_midi
from scipy.signal import fftconvolve

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "poly"))                     # dataprep.py lives here
sys.path.insert(0, "/workspace/retone_poly/lib")                   # pod layout fallback

import dataprep as dp
from instruments import INSTRUMENTS, by_category, list_categories
from arrange import ARRANGERS


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


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


TRANSCRIBERS = {
    "basic_pitch":       transcribe_basic_pitch,
    "bytedance_piano":   transcribe_bytedance_piano,
}


# ────────────────────────── render + polish ────────────────────────────────

def render_sf2(pm, sf2_path, program, seconds, out_wav):
    """Feed the (arranged) PrettyMIDI through FluidSynth via dataprep.render."""
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


def synth_hall_ir(sr=44100, seconds=1.0, decay=3.0):
    """Cheap exp-decaying dense-noise IR — small hall / plate approximation."""
    n = int(sr * seconds)
    ir = np.random.randn(n).astype(np.float32) * np.exp(-decay * np.arange(n) / n)
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


# ────────────────────────── entrypoint ─────────────────────────────────────

def render_one(input_audio, instrument_name, out_wav, transcriber="basic_pitch",
               seconds=None, reverb_wet=0.15, verbose=True):
    """Render one input through one target instrument. Reusable from other scripts."""
    if instrument_name not in INSTRUMENTS:
        raise KeyError(f"unknown instrument {instrument_name!r}; "
                       f"see instruments.py or --list")
    inst = INSTRUMENTS[instrument_name]
    if not os.path.exists(inst.sf2):
        raise FileNotFoundError(f"SF2 missing for {instrument_name}: {inst.sf2}")

    tr = TRANSCRIBERS[transcriber]
    if verbose:
        print(f"  transcribe ({transcriber}) …")
    t0 = time.time()
    pm = tr(input_audio)
    dp.apply_sustain(pm)
    n_notes = sum(len(i.notes) for i in pm.instruments)
    if verbose:
        print(f"    {n_notes} notes in {time.time()-t0:.0f}s")

    if seconds:
        for inst_pm in pm.instruments:
            inst_pm.notes = [n for n in inst_pm.notes if n.start < seconds]
            for n in inst_pm.notes:
                n.end = min(n.end, seconds)

    if verbose:
        print(f"  arrange ({inst.arranger}) …")
    pm = ARRANGERS[inst.arranger](pm)

    if verbose:
        print(f"  render {inst.display} via {os.path.basename(inst.sf2)} …")
    render_sf2(pm, inst.sf2, inst.program, seconds or 300, out_wav)
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
                    help="basic_pitch (default, any polyphonic) OR bytedance_piano (best for piano input)")
    ap.add_argument("--seconds", type=int, default=None,
                    help="clip render to first N seconds of the input")
    ap.add_argument("--reverb-wet", type=float, default=0.15,
                    help="0.0 = dry, 0.3 = wet. Default 0.15.")
    ap.add_argument("--list", action="store_true",
                    help="print the instrument catalog and exit")
    args = ap.parse_args()

    if args.list:
        for cat in list_categories():
            entries = by_category(cat)
            print(f"\n═══ {cat.upper()} ({len(entries)}) ═══")
            for i in entries:
                print(f"  {i.name:32s} {i.display}")
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
                           seconds=args.seconds, reverb_wet=args.reverb_wet)
            except Exception as e:
                print(f"  FAIL: {e}")
    else:
        if not args.instrument or not args.out:
            ap.error("--instrument and --out required (or --category and --outdir)")
        render_one(args.input, args.instrument, args.out,
                   transcriber=args.transcriber,
                   seconds=args.seconds, reverb_wet=args.reverb_wet)


if __name__ == "__main__":
    main()
