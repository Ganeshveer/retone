"""Batch: render multiple source songs through the same curated instrument set.

Auto-picks a transcriber per source:
  - piano solo audio (declared in SOURCES)  → bytedance_piano  (F1 96.72 + real velocity/pedal)
  - other polyphonic sources                → basic_pitch      (universal)

Files land as <source>__INPUT.wav (reference) plus <source>__<instrument>.wav
so the naming makes A/B against the input obvious.

Run:
    python training/direct/render_multi_song.py
Edit SOURCES / TARGETS at the top of the file for your own set.
"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(__file__))

from render_direct import render_one
import librosa, soundfile as sf, numpy as np

OUTDIR = "/workspace/retone_poly/multi_song_samples"

# (source_name, input_path, transcriber, clip_seconds)
SOURCES = [
    ("furelise",        "/workspace/retone_poly/test/seg/furelise_0-12.wav",                    "bytedance_piano", 12),
    ("furelise_real",   "/workspace/retone_poly/test/real_furelise_piano.wav",                  "bytedance_piano", 15),
    ("bohemian",        "/workspace/retone_poly/test/seg/bohemian_piano_20-32.wav",             "bytedance_piano", 12),
    ("georgia_strings", "/workspace/retone_poly/test/seg/georgia_strings_0-10.wav",             "basic_pitch",     10),
    ("georgia_full",    "/workspace/retone_poly/test/RAY_CHARLES_-_Georgia_On_My_Mind_Instrumental.mp3", "basic_pitch", 20),
    ("dontstop",        "/workspace/retone_poly/test/Knightsbridge_-_Don_t_Stop_Me_Now_Instrumenta.mp3", "basic_pitch", 15),
]

# Curated diverse target set — one per family.
TARGETS = [
    "piano_sonatina_grand",    # best piano (real sample)
    "cello_sustain",           # rich bowed ensemble strings
    "violin_solo",             # single-voice strings
    "trumpet_solo",            # brass
    "flute_solo",              # woodwind
    "harp_sonatina",           # plucked
]


def copy_input(src, dst, seconds):
    """Save the exact window fed to the pipeline, mono 44.1 kHz, peak-normalized."""
    y, _ = librosa.load(src, sr=44100, mono=True, duration=seconds)
    peak = float(np.max(np.abs(y)) or 1)
    sf.write(dst, (y / peak * 0.9).astype(np.float32), 44100)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    for src_name, src_path, transcriber, seconds in SOURCES:
        if not os.path.exists(src_path):
            print(f"\n[{src_name}] SKIP — missing {src_path}"); continue
        print(f"\n═══ {src_name} ═══ transcriber={transcriber}, first {seconds}s")
        copy_input(src_path, f"{OUTDIR}/{src_name}__INPUT.wav", seconds)
        for inst in TARGETS:
            out = f"{OUTDIR}/{src_name}__{inst}.wav"
            t1 = time.time()
            try:
                render_one(src_path, inst, out, transcriber=transcriber,
                           seconds=seconds, reverb_wet=0.15, verbose=False)
                print(f"  {inst:26s} -> {os.path.getsize(out)/1e6:.2f} MB in {time.time()-t1:.0f}s")
            except Exception as e:
                print(f"  {inst:26s} FAIL: {e}")

    zip_path = shutil.make_archive("/workspace/multi_song_samples", "zip", OUTDIR)
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min")
    print(f"Files: {OUTDIR}")
    print(f"Download-ready zip: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
