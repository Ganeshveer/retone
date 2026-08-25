"""Variety pack: 8 popular instrumentals × 12 diverse targets + INPUT reference.

Sources deliberately span moods and instrumentation so we A/B how the
pipeline handles piano-heavy vs orchestral vs rock-band material.

Transcription is CACHED per source — one transcribe call per song, reused
across all 12 target renders. 12× fewer transcriber invocations.
"""
import os, sys, time, copy, shutil
sys.path.insert(0, "/workspace/retone_poly/direct")
import librosa, soundfile as sf, numpy as np

from render_direct import render_sf2, apply_light_reverb, TRANSCRIBERS
from instruments import INSTRUMENTS
from arrange import ARRANGERS
import dataprep as dp
import pretty_midi

OUTDIR = "/workspace/retone_poly/variety_samples"

# (source_name, path, transcriber, clip_seconds)
SOURCES = [
    ("dontstop_queen",       "/workspace/retone_poly/test/Knightsbridge_-_Don_t_Stop_Me_Now_Instrumenta.mp3", "basic_pitch",     15),
    ("dontstop_12s_seg",     "/workspace/retone_poly/test/seg/dontstop_0-12.wav",                            "basic_pitch",     12),
    ("georgia_raycharles",   "/workspace/retone_poly/test/RAY_CHARLES_-_Georgia_On_My_Mind_Instrumental.mp3", "basic_pitch",     15),
    ("georgia_strings_intro","/workspace/retone_poly/test/seg/georgia_strings_0-10.wav",                     "basic_pitch",     10),
    ("early_heldout_piano",  "/workspace/retone_poly/test/EARLY_heldout_piano.wav",                          "bytedance_piano", 15),
    ("africa_toto",          "/workspace/retone_poly/test/download/africa_toto.mp3",                         "basic_pitch",     15),
    ("pianoman_billyjoel",   "/workspace/retone_poly/test/download/pianoman.mp3",                            "bytedance_piano", 15),
    ("canon_in_d_pachelbel", "/workspace/retone_poly/test/download/canon.mp3",                               "basic_pitch",     15),
]

# 12 targets spanning families
TARGETS = [
    "piano_sonatina_grand",   # piano — real sample
    "piano_ep1_rhodes",        # piano — Rhodes
    "piano_bright",            # piano — bright acoustic
    "cello_sustain",           # strings — bowed low ensemble
    "violin_solo",             # strings — single voice
    "viola_sustain",           # strings — mid ensemble
    "trumpet_solo",            # brass
    "flute_solo",              # wind — bright
    "clarinet_solo",           # wind — dark
    "guitar_nylon",            # plucked warm
    "harp_sonatina",           # plucked — real sample
    "choir_aahs",              # voice
]


def copy_input(src, dst, seconds):
    y, _ = librosa.load(src, sr=44100, mono=True, duration=seconds)
    peak = float(np.max(np.abs(y)) or 1)
    sf.write(dst, (y / peak * 0.9).astype(np.float32), 44100)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    for src_name, src_path, transcriber, seconds in SOURCES:
        if not os.path.exists(src_path):
            print(f"\n[{src_name}] SKIP — missing {src_path}"); continue

        print(f"\n═══ {src_name} ═══ transcriber={transcriber}, {seconds}s")
        copy_input(src_path, f"{OUTDIR}/{src_name}__INPUT.wav", seconds)

        tr = TRANSCRIBERS[transcriber]
        t_tr = time.time()
        pm_base = tr(src_path)
        dp.apply_sustain(pm_base)
        n_notes = sum(len(i.notes) for i in pm_base.instruments)
        print(f"  transcribe: {n_notes} notes in {time.time()-t_tr:.0f}s")

        # trim to the target window once
        for inst_pm in pm_base.instruments:
            inst_pm.notes = [n for n in inst_pm.notes if n.start < seconds]
            for n in inst_pm.notes:
                n.end = min(n.end, seconds)

        for inst_name in TARGETS:
            if inst_name not in INSTRUMENTS:
                print(f"  {inst_name:26s} UNKNOWN"); continue
            inst = INSTRUMENTS[inst_name]
            if not os.path.exists(inst.sf2):
                print(f"  {inst_name:26s} SF2 missing: {inst.sf2}"); continue

            pm = copy.deepcopy(pm_base)
            pm = ARRANGERS[inst.arranger](pm)
            out = f"{OUTDIR}/{src_name}__{inst_name}.wav"
            t_r = time.time()
            try:
                render_sf2(pm, inst.sf2, inst.program, seconds, out)
                apply_light_reverb(out, wet=0.15)
                print(f"  {inst_name:26s} -> {os.path.getsize(out)/1e6:.2f} MB in "
                      f"{time.time()-t_r:.0f}s")
            except Exception as e:
                print(f"  {inst_name:26s} FAIL: {e}")

    zip_path = shutil.make_archive("/workspace/variety_samples", "zip", OUTDIR)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"Files: {OUTDIR}")
    print(f"Zip:   {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
