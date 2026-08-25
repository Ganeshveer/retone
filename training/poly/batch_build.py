"""Batch-build 8 per-instrument caches. 3-way parallel — fluidsynth is
CPU-bound, so 3 concurrent renders keep 3 cores busy without fighting the
GPU training that's still running.
"""
import os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SF2_DIR = "/workspace/sf2"
FLUIDR3 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SEED = "/workspace/retone_poly/seeds/strings_ensemble_best_val0.116.pt"
MIDI_LIMIT = 200
CONCURRENCY = 3                                     # fluidsynth × 3, leave cores for training

INSTRUMENTS = [
    ("violin_sonat",    f"{SF2_DIR}/Strings - 1st Violins Sustain.sf2", 0),
    ("viola_sonat",     f"{SF2_DIR}/Strings - Violas Sustain.sf2",      0),
    ("cello_sonat",     f"{SF2_DIR}/Strings - Celli Sustain.sf2",       0),
    ("bass_sonat",      f"{SF2_DIR}/Strings - Basses Sustain.sf2",      0),
    ("violin_solo",     f"{SF2_DIR}/Strings - Violin Solo.sf2",         0),
    ("cello_solo",      f"{SF2_DIR}/Strings - Cello Solo.sf2",          0),
    ("trumpet_solo",    f"{SF2_DIR}/Brass - Trumpet Solo.sf2",          0),
    ("synth_strings_1", FLUIDR3,                                       50),   # FluidR3 patch 50
]


def build_one(item):
    inst, sf, prog = item
    if not os.path.exists(sf):
        return (inst, "MISSING_SF2", 0)
    t0 = time.time()
    r = subprocess.run(
        ["python3", "/workspace/retone_poly/build_instrument_cache.py",
         "--instrument", inst, "--sf2", sf, "--program", str(prog),
         "--midi-limit", str(MIDI_LIMIT), "--seed-from", SEED],
        capture_output=True, text=True)
    return (inst, f"exit={r.returncode}", time.time() - t0)


if __name__ == "__main__":
    t0 = time.time()
    print(f"batch_build: {len(INSTRUMENTS)} instruments × {MIDI_LIMIT} MIDIs, "
          f"{CONCURRENCY}-way parallel", flush=True)
    with ProcessPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(build_one, item): item[0] for item in INSTRUMENTS}
        for fut in as_completed(futs):
            inst, status, dt = fut.result()
            print(f"  [{inst}] {status} in {dt/60:.1f} min "
                  f"(cumulative {(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
