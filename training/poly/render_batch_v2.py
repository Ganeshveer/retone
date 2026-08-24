"""render_batch v2 — cleaner Stage-1 + polyphony cap + best.pt.

Fixes for the "too many keys pressed" and "shouting strings" complaints:
- raise Basic Pitch onset/frame thresholds and min-note-length (fewer spurious notes)
- cap polyphony per-frame (real pianists play 4-8 notes, not 15+; strings ensembles voice ~4)
- use best.pt (better val) instead of latest.pt
- soften the strings condition (velocity scale) to give the model dynamic headroom
"""
import os, sys, time, copy, argparse
import numpy as np, torch, librosa, soundfile as sf, pretty_midi

sys.path.insert(0, "/workspace/retone_poly")
sys.path.insert(0, "/workspace/retone_poly/lib")
sys.path.insert(0, "/workspace/retone_poly/BigVGAN")

import train_lib as T
from train_lib import PianoRollToMel, MEL_MEAN, MEL_STD
import dataprep as dp
from vocoder import load_bigvgan

OUTDIR = "/workspace/retone_poly/samples_v2"
CKPT_ROOT = "/workspace/retone_poly/ckpt"
DEVICE = "cuda"

SONGS = [
    ("furelise_synth",   "/workspace/retone_poly/test/seg/furelise_0-12.wav",           12),
    ("furelise_real",    "/workspace/retone_poly/test/real_furelise_piano.ogg",         15),
    ("bohemian_piano",   "/workspace/retone_poly/test/seg/bohemian_piano_20-32.wav",    12),
    ("georgia_strings",  "/workspace/retone_poly/test/seg/georgia_strings_0-10.wav",    10),
    ("dontstop",         "/workspace/retone_poly/test/Knightsbridge_-_Don_t_Stop_Me_Now_Instrumenta.mp3", 15),
]

# per-model rendering knobs
POLYPHONY_CAP = {"piano": 6, "strings": 4}
VELOCITY_SCALE = {"piano": 1.0, "strings": 0.7}   # attenuate to give strings headroom vs. shouting
BP = dict(onset_threshold=0.65, frame_threshold=0.5, minimum_note_length=100)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ck.get("cfg", T.CFG)
    m = PianoRollToMel(cfg).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("step", 0), ck.get("val", None)


def transcribe(audio_path, **kwargs):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(audio_path, ICASSP_2022_MODEL_PATH, **kwargs)
    return midi


def cap_polyphony(pm, k):
    """Keep at most k concurrently-sounding notes per frame, ranked by velocity.

    The natural thinning would be to prune weakest ONSETS, but two notes struck
    on beat 1 and beat 2 both "sound" on beat 2 if the beat-1 note is longer.
    Iterating frame-by-frame and dropping the lowest-velocity active notes is the
    right resolution — beat 1's tail dies before beat 2's onset if k is exceeded.
    """
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend((n, inst) for n in inst.notes)
    if not all_notes:
        return pm

    dt = 1.0 / dp.FRAME_RATE
    end = max(n.end for n, _ in all_notes)
    n_frames = int(end / dt) + 1
    # sort notes so we can query "active at time t" cheaply
    by_start = sorted(all_notes, key=lambda x: x[0].start)

    keep = set()
    drop = set()
    i = 0
    active = []  # (note, inst) currently sounding
    for f in range(n_frames):
        t = f * dt
        while i < len(by_start) and by_start[i][0].start <= t:
            active.append(by_start[i]); i += 1
        active = [(n, inst) for (n, inst) in active if n.end > t]
        if len(active) <= k:
            for n, _ in active:
                keep.add(id(n))
            continue
        # keep the k loudest at this frame
        ranked = sorted(active, key=lambda x: -x[0].velocity)
        for n, _ in ranked[:k]:
            keep.add(id(n))
        for n, _ in ranked[k:]:
            drop.add(id(n))

    # a note that's ever kept anywhere gets kept overall (don't chop a note into pieces)
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if id(n) in keep or id(n) not in drop]
    return pm


def scale_velocity(pm, factor):
    for inst in pm.instruments:
        for n in inst.notes:
            n.velocity = max(1, min(127, int(n.velocity * factor)))
    return pm


@torch.no_grad()
def render(pm, model, vocoder, seconds, chunk=1024, overlap=64):
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < seconds]
        for n in inst.notes:
            n.end = min(n.end, seconds)
        inst.control_changes = [c for c in inst.control_changes if c.time < seconds]
    end = max((n.end for i in pm.instruments for n in i.notes), default=0.0)
    if end <= 0:
        return None
    roll = dp.midi_to_roll(pm, int(end * dp.FRAME_RATE) + 1)
    x = torch.from_numpy(roll).unsqueeze(0).to(DEVICE)
    mels = []
    for s in range(0, x.shape[-1], chunk - overlap):
        seg = x[..., s:s + chunk]
        if seg.shape[-1] < 16:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = model(seg).float()
        mels.append(m[..., :-overlap] if s + chunk < x.shape[-1] else m)
    mel = torch.cat(mels, dim=-1) * MEL_STD + MEL_MEAN
    audio = vocoder(mel).squeeze().cpu().numpy()
    peak = np.abs(audio).max()
    return (audio / peak * 0.9).astype(np.float32) if peak > 0 else audio, roll


def copy_input(src, dst, seconds):
    y, sr = librosa.load(src, sr=44100, mono=True, duration=seconds)
    peak = float(np.max(np.abs(y)) or 1)
    y = (y / peak * 0.9).astype(np.float32)
    sf.write(dst, y, 44100)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("loading BigVGAN vocoder ...")
    voc = load_bigvgan()

    print("loading BEST checkpoints ...")
    models = {}
    for inst in ("piano", "strings"):
        p = f"{CKPT_ROOT}/{inst}/best.pt"
        if not os.path.exists(p):
            print(f"  {inst}: MISSING {p}"); continue
        m, step, val = load_model(p)
        val_str = f"val{val:.4f}" if isinstance(val, float) else "valNA"
        models[inst] = (m, step, val_str)
        print(f"  {inst}/best.pt: step {step}, {val_str}, K_poly={POLYPHONY_CAP[inst]}, vel_scale={VELOCITY_SCALE[inst]}")

    for song, path, seconds in SONGS:
        if not os.path.exists(path):
            print(f"\n[{song}] MISSING: {path}"); continue

        print(f"\n═══ [{song}] ({seconds}s) ═══")
        in_out = os.path.join(OUTDIR, f"{song}__INPUT.wav")
        copy_input(path, in_out, seconds)

        t0 = time.time()
        pm = transcribe(path, **BP)
        dp.apply_sustain(pm)
        n_before = sum(len(i.notes) for i in pm.instruments)
        print(f"  transcribed: {n_before} notes (stricter thresholds) in {time.time()-t0:.0f}s")
        if n_before == 0:
            print("  NO NOTES"); continue

        for inst, (model, step, val_str) in models.items():
            pm_copy = copy.deepcopy(pm)
            pm_copy = cap_polyphony(pm_copy, POLYPHONY_CAP[inst])
            pm_copy = scale_velocity(pm_copy, VELOCITY_SCALE[inst])
            n_after = sum(len(i.notes) for i in pm_copy.instruments)
            t0 = time.time()
            audio, roll = render(pm_copy, model, voc, seconds)
            if audio is None:
                print(f"  {inst}: no notes in window"); continue
            poly = roll[1].sum(axis=0); poly = poly[poly > 0]
            out = os.path.join(OUTDIR, f"{song}__AS_{inst}_step{step}_{val_str}_v2.wav")
            sf.write(out, audio, 44100)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.abs(audio).max())
            print(f"  {inst}: {n_before}→{n_after} notes after poly-cap  "
                  f"| polyphony mean {poly.mean():.1f} max {int(poly.max()) if len(poly) else 0}  "
                  f"| render {time.time()-t0:.1f}s  peak {peak:.3f} rms {rms:.4f}")
            print(f"    -> {out}")
            del audio, roll

    print("\nDONE.")


if __name__ == "__main__":
    main()
