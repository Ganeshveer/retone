"""End-to-end: polyphonic audio (or MIDI) -> target instrument audio.

  audio --[Basic Pitch]--> notes --[roll]--> [trained model] --> mel --[BigVGAN]--> audio

Pass --midi to skip transcription (upper bound on quality: no Stage-1 error).
"""
import argparse, os, sys, time
import numpy as np, torch, librosa, soundfile as sf, pretty_midi

sys.path.insert(0, "/workspace/retone_poly")
sys.path.insert(0, "/workspace/retone_poly/lib")
sys.path.insert(0, "/workspace/retone_poly/BigVGAN")

import train_lib as T
from train_lib import PianoRollToMel, MEL_MEAN, MEL_STD
import dataprep as dp
from vocoder import load_bigvgan


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("cfg", T.CFG)
    m = PianoRollToMel(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    print("  model @ step %s (val %.4f)" % (ck.get("step", "?"), ck.get("val", float("nan"))))
    return m


def transcribe(audio_path, out_midi="/tmp/t.mid"):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(audio_path, ICASSP_2022_MODEL_PATH,
                         onset_threshold=0.5, frame_threshold=0.3,
                         minimum_note_length=58)
    midi.write(out_midi)
    return midi


@torch.no_grad()
def render(pm, model, vocoder, device="cuda", chunk=1024, overlap=64):
    # pm.get_end_time() reflects the LAST EVENT of any kind (including sustain-pedal
    # CCs), not the last note. Trimming notes to 15s but leaving CCs at t=311s made
    # this render 311s of mostly silence. Derive the length from the notes themselves.
    end = max((n.end for i in pm.instruments for n in i.notes), default=0.0)
    if end <= 0:
        raise ValueError("no notes to render")
    roll = dp.midi_to_roll(pm, int(end * dp.FRAME_RATE) + 1)
    x = torch.from_numpy(roll).unsqueeze(0).to(device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="audio file, or .mid with --midi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--midi", action="store_true", help="input is MIDI (skip transcription)")
    ap.add_argument("--seconds", type=float, default=30)
    a = ap.parse_args()

    print("loading vocoder + model ...")
    voc = load_bigvgan()
    model = load_model(a.ckpt)

    if a.midi:
        pm = dp.apply_sustain(pretty_midi.PrettyMIDI(a.input))
        print("  MIDI input (no Stage-1 error)")
    else:
        print("  transcribing %s ..." % os.path.basename(a.input))
        t0 = time.time()
        pm = transcribe(a.input)
        print("  transcribed in %.0fs" % (time.time() - t0))

    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < a.seconds]
        for n in inst.notes:
            n.end = min(n.end, a.seconds)
        inst.control_changes = [c for c in inst.control_changes if c.time < a.seconds]
    n_notes = sum(len(i.notes) for i in pm.instruments)
    print("  notes: %d over %.1fs" % (n_notes, a.seconds))
    if n_notes == 0:
        print("  NO NOTES - aborting"); return

    t0 = time.time()
    audio, roll = render(pm, model, voc)
    poly = roll[1].sum(axis=0); act = poly[poly > 0]
    print("  rendered %.1fs in %.1fs | peak %.3f | rms %.4f" % (
        len(audio) / 44100, time.time() - t0, np.abs(audio).max(),
        np.sqrt(np.mean(audio ** 2))))
    print("  polyphony: mean %.1f max %d" % (act.mean() if len(act) else 0, int(poly.max())))
    sf.write(a.out, audio, 44100)
    print("  -> %s" % a.out)


if __name__ == "__main__":
    main()
