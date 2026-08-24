"""Fetch real MAESTRO Disklavier audio pairs via HTTP range requests
(no 108 GB download). Also builds the training cache in real-audio mode.

Design:
- pick performances between 60s and 400s
- shuffle deterministically for coverage across years/composers
- extract only what fits in --budget-gb (default 8 GB) of RAW WAV
- each file yields multiple training crops (crops-per-file matches synth path)
- writes to cache_piano_real/ next to the synthetic cache_piano/
- training scans BOTH dirs, so no code change needed on the training side
"""
import os, sys, csv, time, random, hashlib, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from remotezip import RemoteZip

sys.path.insert(0, "/workspace/retone_poly/lib")
from dataprep import audio_to_mel, midi_to_roll, apply_sustain, MEL, FRAME_RATE
import pretty_midi

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
DL_DIR = Path("data/maestro_real")
CACHE_DIR = Path("cache/piano_new")                                # staging dir — swapped into cache/piano/ after fetch, no duplicates
SR = MEL["sampling_rate"]                                          # 44100
HOP = MEL["hop_size"]                                              # 512


def pick_files(rows, budget_bytes, min_dur=60, max_dur=400, seed=42):
    """Pick a spread of performances until we hit budget.
    File-size heuristic: 44100 * 2ch * 2B * dur = 176_400 * dur bytes."""
    rng = random.Random(seed)
    ok = [r for r in rows if min_dur <= float(r["duration"]) <= max_dur]
    rng.shuffle(ok)
    picked, total = [], 0
    for r in ok:
        est = int(176400 * float(r["duration"]))
        if total + est > budget_bytes:
            continue
        picked.append(r); total += est
        if total > budget_bytes * 0.95:
            break
    return picked, total


def process_pair(wav_path, midi_path, out_dir, seconds, hop_seconds, tag):
    """Turn one (wav, midi) pair into multiple (mel, roll) crops.

    Real audio → same mel path as synthetic (peak-normalize, log-mel).
    MIDI → same piano-roll path as synthetic (with sustain applied).
    Alignment: MAESTRO claims <3ms MIDI/audio sync — well below one 512-sample frame."""
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)                                         # to mono
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for inst in pm.instruments:
        inst.program = 0                                           # canonicalize to Acoustic Grand
    apply_sustain(pm)                                              # bake CC64 into note offsets

    n_samples = len(y)
    n_frames = n_samples // HOP
    seq_samples = seconds * SR
    hop_samples = hop_seconds * SR

    written = 0
    for start in range(0, n_samples - seq_samples + 1, hop_samples):
        end = start + seq_samples
        y_seg = y[start:end]
        # skip near-silent segments
        if float(np.max(np.abs(y_seg))) < 1e-3:
            continue
        m = audio_to_mel(y_seg)                                    # (128, T)
        T = m.shape[1]
        t_start = start / SR
        t_end = end / SR
        # slice the MIDI to the segment's time range and shift to t=0
        pm_seg = pretty_midi.PrettyMIDI()
        inst_seg = pretty_midi.Instrument(program=0)
        for inst in pm.instruments:
            for note in inst.notes:
                if note.end <= t_start or note.start >= t_end:
                    continue
                inst_seg.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0.0, note.start - t_start),
                    end=min(t_end - t_start, note.end - t_start),
                ))
        pm_seg.instruments.append(inst_seg)
        r = midi_to_roll(pm_seg, n_frames=T)                       # (3, 128, T)
        out = out_dir / f"pair_real_{tag}_{start//SR:04d}s.npz"    # `pair_*.npz` is the glob PairDataset expects
        np.savez_compressed(out, mel=m.astype(np.float32), roll=r.astype(np.float32))
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=8.0,
                    help="Max GB of raw WAV to download (fits in pod disk).")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--hop-seconds", type=int, default=30,
                    help="Crop hop within a performance (30s = 2x overlap vs 60s window).")
    ap.add_argument("--min-dur", type=int, default=60)
    ap.add_argument("--max-dur", type=int, default=400,
                    help="Cap per-performance duration (bigger = fewer files at same budget).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap number of files (debug).")
    ap.add_argument("--wipe", action="store_true",
                    help="Delete existing cache/piano/*.npz before writing new (no duplicates).")
    args = ap.parse_args()

    DL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.wipe:
        removed = 0
        for p in CACHE_DIR.glob("*.npz"):
            p.unlink(); removed += 1
        print(f"wiped {removed} existing crops from {CACHE_DIR}")

    print(f"opening remote zip …")
    t0 = time.time()
    rz = RemoteZip(MAESTRO_URL)
    print(f"  ready in {time.time()-t0:.1f}s ({len(rz.namelist())} entries)")

    # get the manifest from inside the zip
    csv_name = [n for n in rz.namelist() if n.endswith("maestro-v3.0.0.csv")][0]
    rz.extract(csv_name, "/tmp/_maestro_probe/")
    with open(f"/tmp/_maestro_probe/{csv_name}") as f:
        rows = list(csv.DictReader(f))

    budget = int(args.budget_gb * 1e9)
    picked, est_total = pick_files(rows, budget, args.min_dur, args.max_dur)
    if args.limit:
        picked = picked[:args.limit]
        est_total = sum(int(176400 * float(r["duration"])) for r in picked)
    print(f"selected {len(picked)} performances, est {est_total/1e9:.1f} GB raw WAV")

    audio_names_in_zip = set(n for n in rz.namelist() if n.endswith(".wav"))
    midi_names_in_zip = set(n for n in rz.namelist() if n.endswith(".midi"))

    total_crops = 0
    total_bytes = 0
    t_start = time.time()
    for i, r in enumerate(picked):
        wav_target = r["audio_filename"]
        midi_target = r["midi_filename"]
        wav_in_zip = next((n for n in audio_names_in_zip if n.endswith(wav_target)), None)
        midi_in_zip = next((n for n in midi_names_in_zip if n.endswith(midi_target)), None)
        if not (wav_in_zip and midi_in_zip):
            print(f"  [{i}] SKIP missing: {wav_target}")
            continue

        # extract to DL_DIR
        rz.extract(wav_in_zip, str(DL_DIR))
        rz.extract(midi_in_zip, str(DL_DIR))
        wav_path = DL_DIR / wav_in_zip
        midi_path = DL_DIR / midi_in_zip

        tag = hashlib.md5(wav_target.encode()).hexdigest()[:10]
        try:
            n = process_pair(wav_path, midi_path, CACHE_DIR,
                             args.seconds, args.hop_seconds, tag)
        except Exception as e:
            print(f"  [{i}] ERR building crops from {wav_target}: {e}")
            n = 0

        # delete raw WAV/MIDI after processing to keep disk lean
        sz = wav_path.stat().st_size
        total_bytes += sz
        wav_path.unlink()
        midi_path.unlink()

        total_crops += n
        elapsed = time.time() - t_start
        rate = total_bytes / max(elapsed, 1) / 1e6
        print(f"  [{i+1}/{len(picked)}] {wav_target[-60:]}  "
              f"{sz/1e6:5.1f} MB → {n} crops   "
              f"cum {total_bytes/1e9:.2f} GB @ {rate:.1f} MB/s")

    print(f"\nDONE: {total_crops} crops written to {CACHE_DIR}, "
          f"{total_bytes/1e9:.2f} GB total WAV processed, "
          f"{(time.time()-t_start)/60:.1f} min elapsed")


if __name__ == "__main__":
    main()
