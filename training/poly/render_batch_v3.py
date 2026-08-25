"""render_batch v3 — A/B arrangement vs plain (no cap, no post-transcribe editing).

For each song, four renders:
  <song>__AS_piano_ARRANGED.wav    — transcribe → arrange_for_piano → model
  <song>__AS_piano_PLAIN.wav       — transcribe → model  (baseline)
  <song>__AS_strings_ARRANGED.wav  — transcribe → arrange_for_strings → model
  <song>__AS_strings_PLAIN.wav     — transcribe → model  (baseline)

Piano uses best_pre_realdata.pt (step 14000, val 0.119) — the true best.
Strings uses best.pt (step 12000, val 0.159).
NO polyphony cap. NO velocity scale. Isolating arrange.py's contribution.
"""
import os, sys, time, copy
import numpy as np, torch, librosa, soundfile as sf, pretty_midi

sys.path.insert(0, "/workspace/retone_poly")
sys.path.insert(0, "/workspace/retone_poly/lib")
sys.path.insert(0, "/workspace/retone_poly/BigVGAN")

import train_lib as T
from train_lib import PianoRollToMel, MEL_MEAN, MEL_STD
import dataprep as dp
from vocoder import load_bigvgan
from arrange import arrange_for_piano, arrange_for_strings

OUTDIR = "/workspace/retone_poly/samples_v3"
DEVICE = "cuda"

MODELS = {
    # After 8h more training on real+synth, piano LATEST is the newer model, at
    # step ~67500. Val is higher (~0.16) because the mixed distribution is harder
    # than pure synth, but this is the model that has actually SEEN real audio.
    "piano":   "/workspace/retone_poly/ckpt/piano/latest.pt",
    # Strings best.pt is genuinely the best — val 0.116 at step ~64000.
    "strings": "/workspace/retone_poly/ckpt/strings/best.pt",
}

ARRANGERS = {
    "piano":   arrange_for_piano,
    "strings": arrange_for_strings,
}

SONGS = [
    ("furelise_synth",   "/workspace/retone_poly/test/seg/furelise_0-12.wav",           12),
    ("furelise_real",    "/workspace/retone_poly/test/real_furelise_piano.ogg",         15),
    ("bohemian_piano",   "/workspace/retone_poly/test/seg/bohemian_piano_20-32.wav",    12),
    ("georgia_strings",  "/workspace/retone_poly/test/seg/georgia_strings_0-10.wav",    10),
    ("dontstop",         "/workspace/retone_poly/test/Knightsbridge_-_Don_t_Stop_Me_Now_Instrumenta.mp3", 15),
]

BP = dict(onset_threshold=0.65, frame_threshold=0.5, minimum_note_length=100)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
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


@torch.no_grad()
def render(pm, model, vocoder, seconds, chunk=1024, overlap=64):
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < seconds]
        for n in inst.notes:
            n.end = min(n.end, seconds)
        inst.control_changes = [c for c in inst.control_changes if c.time < seconds]
    end = max((n.end for i in pm.instruments for n in i.notes), default=0.0)
    if end <= 0:
        return None, None
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
    return ((audio / peak * 0.9).astype(np.float32) if peak > 0 else audio), roll


def copy_input(src, dst, seconds):
    y, _ = librosa.load(src, sr=44100, mono=True, duration=seconds)
    peak = float(np.max(np.abs(y)) or 1)
    sf.write(dst, (y / peak * 0.9).astype(np.float32), 44100)


def poly_stats(roll):
    poly = roll[1].sum(axis=0)
    poly = poly[poly > 0]
    return (float(poly.mean()), int(poly.max())) if len(poly) else (0.0, 0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("loading BigVGAN ...")
    voc = load_bigvgan()

    print("loading models ...")
    models = {}
    for inst, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  {inst}: MISSING {path}"); continue
        m, step, val = load_model(path)
        val_str = f"val{val:.4f}" if isinstance(val, float) else "valNA"
        models[inst] = (m, step, val_str)
        print(f"  {inst}: step {step}, {val_str}, arranger={ARRANGERS[inst].__name__}")

    for song, path, seconds in SONGS:
        if not os.path.exists(path):
            print(f"\n[{song}] MISSING"); continue
        print(f"\n═══ [{song}] ({seconds}s) ═══")

        copy_input(path, os.path.join(OUTDIR, f"{song}__INPUT.wav"), seconds)

        t0 = time.time()
        pm = transcribe(path, **BP)
        dp.apply_sustain(pm)
        n_transcribed = sum(len(i.notes) for i in pm.instruments)
        print(f"  transcribed: {n_transcribed} notes in {time.time()-t0:.0f}s")
        if n_transcribed == 0:
            continue

        for inst, (model, step, val_str) in models.items():
            arranger = ARRANGERS[inst]
            for mode, pm_use in (("PLAIN", copy.deepcopy(pm)),
                                 ("ARRANGED", arranger(copy.deepcopy(pm)))):
                n_notes = sum(len(i.notes) for i in pm_use.instruments)
                t0 = time.time()
                audio, roll = render(pm_use, model, voc, seconds)
                if audio is None:
                    print(f"  {inst} {mode}: nothing to render"); continue
                mean_poly, max_poly = poly_stats(roll)
                out = os.path.join(OUTDIR, f"{song}__AS_{inst}_{mode}_step{step}.wav")
                sf.write(out, audio, 44100)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                peak = float(np.abs(audio).max())
                print(f"  {inst} {mode:8s}: {n_notes:5d} notes | poly {mean_poly:.1f}/{max_poly:2d} "
                      f"| render {time.time()-t0:.1f}s | peak {peak:.3f} rms {rms:.4f}")
                del audio, roll

    print("\nDONE.")


if __name__ == "__main__":
    main()
