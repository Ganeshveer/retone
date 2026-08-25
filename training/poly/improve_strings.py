"""Strings quality improvement — cache rebuild + optional real-audio mix.

Diagnosis (val 0.116 but sounds thin/harsh): trained on MAESTRO MIDI through
ONE soundfont (FluidR3_GM patch 48). Model learned exactly that timbre.
Prescription (per research):

  1. Multi-soundfont diversity  — same MIDI, several SF2s (VSCO 2 CE, GeneralUser
     GS, Sonatina). Cheap, direct hit on "one-timbre overfit".
  2. Real strings audio        — URMP / MusicNet strings-only / Slakh strings
     stems. Small but real, closes the sim-to-real gap.
  3. Reduce FluidR3 weight     — subsample the existing FluidR3 crops so the
     new (diverse) crops dominate the mixture. Otherwise 260 old crops still
     dominate a fresh 260 new-crop set.

This script does 1 and 3. Step 2 lands in `fetch_urmp_strings.py` (drop-in
sibling once we settle on which datasets to pull).

Usage:
    python improve_strings.py \
        --soundfonts /usr/share/sounds/sf2/FluidR3_GM.sf2 \
                     /workspace/sf2/VSCO2CE_Strings.sf2 \
                     /workspace/sf2/GeneralUser_GS.sf2 \
        --keep-fluidr3-frac 0.4 \
        --midi-limit 400

Uses the existing dataprep.build() to render each soundfont's variant into
cache/strings/. Filenames are prefixed with the soundfont's short name so the
mixture is auditable.
"""
import argparse, glob, hashlib, os, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dataprep as dp


def sf_tag(path):
    """3-6 char label for the soundfont — used in output filenames."""
    stem = Path(path).stem.lower()
    for kw in ("fluidr3", "vsco", "generaluser", "sonatina", "salamander"):
        if kw in stem:
            return kw[:6]
    return hashlib.md5(path.encode()).hexdigest()[:6]


def prune_fluidr3(cache_dir, keep_frac):
    """Subsample existing FluidR3-only crops so the new diverse renders can
    dominate the training mixture. Assumes existing files were untagged
    (from before this script), so we treat *all* pair_*.npz (that aren't
    pair_real_* or pair_<sftag>_*) as legacy FluidR3."""
    legacy = []
    tagged_prefixes = {"pair_real_"} | {f"pair_{sf_tag(sf)}_" for sf in ()}   # empty at this point
    for f in sorted(glob.glob(str(cache_dir / "pair_*.npz"))):
        name = Path(f).name
        if any(name.startswith(p) for p in tagged_prefixes):
            continue
        legacy.append(f)
    if not legacy:
        print("  no legacy FluidR3-only crops to prune")
        return
    keep_n = int(len(legacy) * keep_frac)
    rng = random.Random(940513)
    rng.shuffle(legacy)
    to_delete = legacy[keep_n:]
    for f in to_delete:
        os.remove(f)
    print(f"  pruned FluidR3-only crops: {len(legacy)} → kept {keep_n} "
          f"(deleted {len(to_delete)})")


def render_soundfont_variant(sf2_path, midi_files, cache_dir, seconds, tag):
    """Render each MIDI through this soundfont, appending crops to cache_dir
    with a filename prefix that identifies the soundfont."""
    print(f"  rendering through {Path(sf2_path).name} (tag={tag}) — "
          f"{len(midi_files)} MIDIs …")
    # dataprep.build writes filenames like pair_<hash>_<time>s.npz — we want
    # pair_<tag>_<hash>_<time>s.npz so we can audit the mixture. Monkey-patch:
    orig_sf2 = dp.SF2
    dp.SF2 = sf2_path
    try:
        # dataprep.build's signature: (midi_files, instrument, out_dir, seconds=60, ...)
        dp.build(midi_files, "strings", str(cache_dir), seconds=seconds,
                 filename_prefix=f"pair_{tag}_")
    finally:
        dp.SF2 = orig_sf2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundfonts", nargs="+", required=True,
                    help="Absolute paths to SF2 files. FluidR3 first is fine.")
    ap.add_argument("--midi-dir",
                    default="/workspace/retone_poly/data/maestro/maestro-v3.0.0",
                    help="Directory to glob **/*.midi from.")
    ap.add_argument("--midi-limit", type=int, default=400,
                    help="Cap on MIDIs per soundfont — with 3 SFs this is 3× multiplier.")
    ap.add_argument("--cache-dir",
                    default="/workspace/retone_poly/cache/strings")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--keep-fluidr3-frac", type=float, default=0.4,
                    help="Fraction of existing (untagged) FluidR3-only crops to keep.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for sf in args.soundfonts:
        if not os.path.exists(sf):
            print(f"MISSING SF2: {sf}"); return

    midi_files = sorted(glob.glob(f"{args.midi_dir}/**/*.midi", recursive=True))
    if not midi_files:
        print(f"NO MIDIs under {args.midi_dir}"); return
    rng = random.Random(940513)
    rng.shuffle(midi_files)
    midi_files = midi_files[:args.midi_limit]
    print(f"selected {len(midi_files)} MIDIs from {args.midi_dir}")

    print("\n=== prune legacy FluidR3-only crops ===")
    prune_fluidr3(cache_dir, args.keep_fluidr3_frac)

    print(f"\n=== render {len(args.soundfonts)} soundfont variants ===")
    for sf in args.soundfonts:
        render_soundfont_variant(sf, midi_files, cache_dir, args.seconds, sf_tag(sf))

    n_final = len(glob.glob(str(cache_dir / "pair_*.npz")))
    print(f"\nDONE — cache/strings/ now has {n_final} pair_*.npz files.")
    print("Next: restart strings training. The patched resume in train_poly.py will")
    print("      seed `best` from best.pt (val 0.116) so it isn't clobbered.")


if __name__ == "__main__":
    main()
