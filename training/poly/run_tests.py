"""Render every test segment through both trained models. Writes matched
INPUT/OUTPUT pairs so each conversion can be A/B'd directly."""
import os, sys, glob, subprocess, shutil

TEST = "/workspace/retone_poly/test"
OUT  = "/workspace/retone_poly/results"
os.makedirs(OUT, exist_ok=True)

# (segment, [target instruments])  — every segment through every model
JOBS = [
    ("seg/furelise_0-12.wav",        ["strings", "piano"]),
    ("seg/bohemian_piano_20-32.wav", ["strings", "piano"]),
    ("seg/georgia_strings_0-10.wav", ["piano", "strings"]),
    ("seg/dontstop_0-12.wav",        ["strings", "piano"]),
    ("heldout/held_0.midi",          ["piano", "strings"]),
]

def ckpt_for(inst):
    """Prefer best.pt; fall back to the newest step snapshot."""
    b = "/workspace/retone_poly/ckpt/%s/best.pt" % inst
    if os.path.exists(b):
        return b
    steps = sorted(glob.glob("/workspace/retone_poly/ckpt/%s/step_*.pt" % inst),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    return steps[-1] if steps else None

for seg, targets in JOBS:
    src = os.path.join(TEST, seg)
    if not os.path.exists(src):
        print("MISSING %s" % src); continue
    name = os.path.basename(seg).rsplit(".", 1)[0]
    is_midi = seg.endswith(".midi")

    # copy the input alongside the outputs so listening is A/B, not hunting
    if not is_midi:
        shutil.copy(src, os.path.join(OUT, "%s__INPUT.wav" % name))

    for inst in targets:
        ck = ckpt_for(inst)
        if not ck:
            print("no checkpoint for %s" % inst); continue
        step = "best" if ck.endswith("best.pt") else ck.split("_")[-1].split(".")[0]
        out = os.path.join(OUT, "%s__AS_%s.wav" % (name, inst))
        cmd = [sys.executable, "/workspace/retone_poly/infer.py",
               "--ckpt", ck, "--input", src, "--out", out, "--seconds", "12"]
        if is_midi:
            cmd.append("--midi")
        print("\n=== %s -> %s (ckpt %s) ===" % (name, inst, step), flush=True)
        env = dict(os.environ, RETONE_INSTRUMENT=inst)
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        for line in (r.stdout or "").splitlines():
            if any(k in line for k in ("notes:", "rendered", "polyphony", "->", "transcribed", "model @")):
                print("   ", line.strip())
        if r.returncode != 0:
            print("    FAILED:", (r.stderr or "").strip().splitlines()[-1:])

print("\n=== results ===")
for f in sorted(os.listdir(OUT)):
    print("   ", f)
