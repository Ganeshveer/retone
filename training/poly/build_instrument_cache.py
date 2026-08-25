"""Build a per-instrument training cache — one instrument, one soundfont, one model.

Replaces the mixed-strings pipeline. Each invocation renders MAESTRO MIDIs
through ONE soundfont into cache/<instrument>/ and seeds ckpt/<instrument>/best.pt
from a parent checkpoint so training warm-starts instead of learning from scratch.

Usage:
    python build_instrument_cache.py \
        --instrument violin_sonat \
        --sf2 "/workspace/sf2/Strings - 1st Violins Sustain.sf2" \
        --midi-limit 300 \
        --seed-from seeds/strings_ensemble_best_val0.116.pt

    # Then to train that instrument:
    RETONE_INSTRUMENT=violin_sonat python train_poly.py

Rationale for warm-start: the parent checkpoint (val 0.116 on FluidR3 GM
strings) has learned generalizable structure — chord shape → mel timbre,
sustain envelopes, polyphonic combinations. Fine-tuning per-instrument
converges vastly faster than training from random init.
"""
import argparse, glob, os, random, shutil, sys
from pathlib import Path

# dataprep lives in this dir OR under lib/ depending on repo layout;
# try both so the same script works locally and on the pod.
_here = Path(__file__).parent
for _cand in (_here, _here / "lib", _here.parent / "lib"):
    if (_cand / "dataprep.py").exists():
        sys.path.insert(0, str(_cand)); break
import dataprep as dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", required=True,
                    help="short name — becomes cache/<instrument>/ and ckpt/<instrument>/")
    ap.add_argument("--sf2", required=True,
                    help="single soundfont path — this instrument's ONE timbre")
    ap.add_argument("--midi-dir",
                    default="/workspace/retone_poly/data/maestro/maestro-v3.0.0",
                    help="glob **/*.midi from here")
    ap.add_argument("--midi-limit", type=int, default=300,
                    help="cap on MIDIs to render. 300 × 60 s = ~5 h of data, plenty.")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--program", type=int, default=0,
                    help="MIDI program to force. Sonatina single-instrument SF2s ignore this and play their sole preset either way.")
    ap.add_argument("--seed-from", default=None,
                    help="optional checkpoint path to copy to ckpt/<instrument>/best.pt as warm-start")
    ap.add_argument("--cache-root", default="/workspace/retone_poly/cache")
    ap.add_argument("--ckpt-root", default="/workspace/retone_poly/ckpt")
    args = ap.parse_args()

    if not os.path.exists(args.sf2):
        print(f"MISSING SF2: {args.sf2}"); return

    cache_dir = Path(args.cache_root) / args.instrument
    ckpt_dir  = Path(args.ckpt_root)  / args.instrument
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(glob.glob(f"{args.midi_dir}/**/*.midi", recursive=True))
    if not midi_files:
        print(f"NO MIDIs under {args.midi_dir}"); return
    rng = random.Random(940513)
    rng.shuffle(midi_files)
    midi_files = midi_files[:args.midi_limit]
    print(f"[{args.instrument}] rendering {len(midi_files)} MIDIs through {Path(args.sf2).name}")

    # Point dataprep at the requested SF2 + program for the duration of this render.
    # dp.PROGRAMS is a dict keyed by "instrument" — force whichever key we're using.
    orig_sf, orig_progs = dp.SF2, dict(dp.PROGRAMS)
    try:
        dp.SF2 = args.sf2
        dp.PROGRAMS[args.instrument] = args.program
        # dataprep.build renders + caches (roll, mel) pairs. Uses "pair_" prefix
        # by default, matches PairDataset's glob.
        dp.build(midi_files, args.instrument, str(cache_dir), seconds=args.seconds)
    finally:
        dp.SF2, dp.PROGRAMS = orig_sf, orig_progs

    n = len(list(cache_dir.glob("pair_*.npz")))
    print(f"[{args.instrument}] cache/{args.instrument}/ now has {n} pair_*.npz files")

    # Warm-start: copy parent checkpoint as this instrument's best.pt.
    # train_poly.py's else-branch loads best.pt when no latest.pt exists — model
    # weights only, opt/sched fresh, step reset to 0.
    if args.seed_from:
        src = Path(args.seed_from)
        if not src.exists():
            print(f"WARN: --seed-from {src} missing, skipping warm-start")
        else:
            dst = ckpt_dir / "best.pt"
            shutil.copy(src, dst)
            print(f"[{args.instrument}] seeded ckpt/{args.instrument}/best.pt from {src.name}")

    print(f"\nnext: RETONE_INSTRUMENT={args.instrument} python train_poly.py")


if __name__ == "__main__":
    main()
