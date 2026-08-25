"""A/B: same source, same target — vary only the transcriber.

Renders 3 piano songs × 3 transcribers × 6 targets = 54 files. Naming
`<song>__<transcriber>__<instrument>.wav` makes the ONLY variable the
transcriber, so a listener can pick a winner cleanly.

Requires (installed lazily by the individual transcribe_* functions the
first time each is used):
    pip install basic-pitch piano_transcription_inference transkun

Transkun only accepts WAV — the OGG source is transparently converted to
a sibling .wav at first use.

Run:
    python training/direct/render_transcriber_ab.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from render_direct import render_one
import librosa, soundfile as sf


OUTDIR = "/workspace/retone_poly/transcriber_ab"

SONGS = [
    ("furelise",       "/workspace/retone_poly/test/seg/furelise_0-12.wav",         12),
    ("furelise_real",  "/workspace/retone_poly/test/real_furelise_piano.ogg",       15),
    ("bohemian",       "/workspace/retone_poly/test/seg/bohemian_piano_20-32.wav",  12),
]

TARGETS = [
    "piano_sonatina_grand", "cello_sustain", "violin_solo",
    "trumpet_solo",         "flute_solo",    "harp_sonatina",
]

TRANSCRIBERS_TO_TRY = ["basic_pitch", "bytedance_piano", "transkun"]


def ensure_wav(path):
    """Transkun rejects non-WAV inputs. If we hit an .ogg/.mp3 etc, transparently
    write a sibling .wav (mono 44.1 kHz) once and return that."""
    if path.lower().endswith(".wav"):
        return path
    wav_path = os.path.splitext(path)[0] + ".wav"
    if not os.path.exists(wav_path):
        y, _ = librosa.load(path, sr=44100, mono=True)
        sf.write(wav_path, y, 44100)
    return wav_path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    for song, path, sec in SONGS:
        if not os.path.exists(path):
            print(f"[{song}] SKIP missing {path}"); continue
        print(f"\n═══ {song} ═══")
        for tr in TRANSCRIBERS_TO_TRY:
            # Transkun rejects non-WAV; auto-convert once.
            src = ensure_wav(path) if tr == "transkun" else path
            for target in TARGETS:
                out = f"{OUTDIR}/{song}__{tr}__{target}.wav"
                t1 = time.time()
                try:
                    render_one(src, target, out, transcriber=tr,
                               seconds=sec, reverb_wet=0.15, verbose=False)
                    print(f"  {tr:18s} {target:26s} {os.path.getsize(out)/1e6:.2f} MB "
                          f"in {time.time()-t1:.0f}s")
                except Exception as e:
                    print(f"  {tr:18s} {target:26s} FAIL: {e}")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min → {OUTDIR}")


if __name__ == "__main__":
    main()
