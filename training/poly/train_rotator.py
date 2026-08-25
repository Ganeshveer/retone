"""Round-robin trainer: cycles through instruments, each gets STEPS_PER_TURN
additional training steps per turn, then rotates. Each instrument's state is
preserved in ckpt/<instrument>/latest.pt so the next turn resumes cleanly.

train_poly.py exits after RETONE_MAX_STEPS_THIS_RUN additional steps (env
var), so this rotator just launches it per instrument in a subprocess and
waits for the process to exit. No signal-handling ugliness.

Runs forever until max total budget hit or SIGINT.
"""
import os, subprocess, sys, time

INSTRUMENTS = [
    "violin_sonat", "viola_sonat", "cello_sonat", "bass_sonat",
    "violin_solo",  "cello_solo",  "trumpet_solo", "synth_strings_1",
]

STEPS_PER_TURN = 2500                                # per user's spec
MAX_TOTAL_HOURS = 48                                 # backstop

CACHE_ROOT = "/workspace/retone_poly/cache"
POLY_DIR   = "/workspace/retone_poly"


def has_cache(inst):
    d = os.path.join(CACHE_ROOT, inst)
    if not os.path.isdir(d):
        return False
    import glob
    return len(glob.glob(os.path.join(d, "pair_*.npz"))) > 0


def train_one_turn(inst):
    env = os.environ.copy()
    env["RETONE_INSTRUMENT"]           = inst
    env["RETONE_MAX_STEPS_THIS_RUN"]   = str(STEPS_PER_TURN)
    print(f"\n--- rotator: training {inst} for +{STEPS_PER_TURN} steps ---", flush=True)
    t0 = time.time()
    r = subprocess.run(
        ["python3", os.path.join(POLY_DIR, "train_poly.py")],
        env=env,
        stdout=open(os.path.join(POLY_DIR, f"train_{inst}.log"), "a"),
        stderr=subprocess.STDOUT,
    )
    dt = time.time() - t0
    print(f"    {inst} exit={r.returncode} in {dt/60:.1f} min", flush=True)
    return r.returncode


def main():
    t0 = time.time()
    turn = 0
    while (time.time() - t0) / 3600 < MAX_TOTAL_HOURS:
        turn += 1
        print(f"\n════════ rotator TURN {turn} ═════════════", flush=True)
        touched = 0
        for inst in INSTRUMENTS:
            if not has_cache(inst):
                print(f"  [skip] {inst} — no cache yet", flush=True)
                continue
            train_one_turn(inst)
            touched += 1
        if touched == 0:
            print("  no cached instruments yet — sleeping 60 s", flush=True)
            time.sleep(60)
    print("rotator: reached MAX_TOTAL_HOURS budget", flush=True)


if __name__ == "__main__":
    main()
