"""Build aligned (piano_roll, mel) training pairs.

Source performances come from MAESTRO MIDI — real human playing with true velocity
and sustain pedal. Each performance is rendered through a GM soundfont twice (once
per target instrument), so both models learn from IDENTICAL performances and the
only variable is timbre.

Alignment is exact by construction: the audio is rendered FROM the MIDI.
"""
import os, sys, math, subprocess, tempfile, pathlib, random
import numpy as np, torch, pretty_midi, librosa, soundfile as sf

sys.path.insert(0, "/workspace/retone_poly/BigVGAN")
from meldataset import mel_spectrogram

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# GM program numbers per target instrument
PROGRAMS = {
    "piano":   0,    # Acoustic Grand Piano
    "strings": 48,   # String Ensemble 1
    "guitar":  24,   # Acoustic Guitar (nylon)
    "rhodes":  4,    # Electric Piano 1
    "organ":   19,   # Church Organ
}

MEL = dict(n_fft=2048, num_mels=128, sampling_rate=44100,
           hop_size=512, win_size=2048, fmin=0, fmax=None)
FRAME_RATE = MEL["sampling_rate"] / MEL["hop_size"]   # 86.13 fps


def apply_sustain(pm):
    """Extend note offsets to sustain-pedal release (CC64).

    fluidsynth honours the pedal when rendering, so the AUDIO has extended sustain.
    If the conditioning roll shows note-off while the audio is still ringing, the
    model is trained on a lie. Onsets & Frames does exactly this transform.
    """
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        pedal = [(c.time, c.value >= 64) for c in inst.control_changes if c.number == 64]
        if not pedal:
            continue
        pedal.sort()
        # Build the intervals during which the pedal is DOWN.
        downs, start = [], None
        for t, on in pedal:
            if on and start is None:
                start = t
            elif not on and start is not None:
                downs.append((start, t)); start = None
        if start is not None:
            downs.append((start, 1e9))
        for n in inst.notes:
            for a, b in downs:
                # A note released while the pedal is held rings until pedal-up.
                if a <= n.end <= b:
                    n.end = max(n.end, b)
                    break
    return pm


def render(midi_path, instrument, out_wav, max_seconds=None):
    """MIDI -> audio through the GM soundfont, forced to one instrument program."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if max_seconds:
        for inst in pm.instruments:
            inst.notes = [n for n in inst.notes if n.start < max_seconds]
            for n in inst.notes:
                n.end = min(n.end, max_seconds)
            # Trim control changes too. Notes alone is not enough: fluidsynth renders
            # until the LAST event of any kind, so leftover pedal CCs out at t=969s
            # produce 16 minutes of silence for a 20-second excerpt.
            inst.control_changes = [c for c in inst.control_changes if c.time < max_seconds]
    for inst in pm.instruments:
        if not inst.is_drum:
            inst.program = PROGRAMS[instrument]
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        tmp_mid = tf.name
    pm.write(tmp_mid)
    subprocess.run(["fluidsynth", "-ni", "-F", str(out_wav),
                    "-r", str(MEL["sampling_rate"]), "-g", "0.6", SF2, tmp_mid],
                   check=True, capture_output=True)
    os.unlink(tmp_mid)
    return out_wav


def midi_to_roll(pm, n_frames, n_pitches=128, onset_frames=3):
    """(3, 128, T): [onset ramp, sustain, velocity]. Pedal already folded into offsets."""
    roll = np.zeros((3, n_pitches, n_frames), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if not (0 <= n.pitch < n_pitches):
                continue
            s = max(0, min(int(round(n.start * FRAME_RATE)), n_frames - 1))
            e = max(s + 1, min(int(round(n.end * FRAME_RATE)), n_frames))
            roll[1, n.pitch, s:e] = 1.0
            roll[2, n.pitch, s:e] = n.velocity / 127.0
            # Ramped onset, not a 1-frame spike: at 86 fps one frame is 11.6 ms and a
            # single-frame target is a very sparse gradient signal.
            for k in range(min(onset_frames, e - s)):
                roll[0, n.pitch, s + k] = max(roll[0, n.pitch, s + k], 1.0 - k / onset_frames)
    return roll


def audio_to_mel(y):
    """BigVGAN-compatible log-mel. Peak-normalizes first — BigVGAN was trained on
    volume-normalized waveform and its own loader does exactly this."""
    y = librosa.util.normalize(y) * 0.95
    m = mel_spectrogram(torch.from_numpy(y.astype(np.float32)).unsqueeze(0),
                        MEL["n_fft"], MEL["num_mels"], MEL["sampling_rate"],
                        MEL["hop_size"], MEL["win_size"], MEL["fmin"], MEL["fmax"],
                        center=False)
    return m.squeeze(0).numpy()


def build(midi_files, instrument, out_dir, seconds=60, min_frames=512, verbose_every=25):
    """Render + cache (roll, mel) pairs for one target instrument."""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "r.wav")
        for i, mp in enumerate(midi_files):
            try:
                render(mp, instrument, wav, max_seconds=seconds)
                y, _ = librosa.load(wav, sr=MEL["sampling_rate"], mono=True)
                y = y[: int(seconds * MEL["sampling_rate"])]   # guarantee the length
                if len(y) < MEL["hop_size"] * min_frames:
                    n_skip += 1; continue
                mel = audio_to_mel(y)
                pm = apply_sustain(pretty_midi.PrettyMIDI(str(mp)))
                for inst in pm.instruments:
                    inst.notes = [n for n in inst.notes if n.start < seconds]
                roll = midi_to_roll(pm, mel.shape[1])
                T = min(roll.shape[2], mel.shape[1])
                if T < min_frames:
                    n_skip += 1; continue
                np.savez_compressed(out_dir / f"pair_{n_ok:05d}.npz",
                                    roll=roll[:, :, :T].astype(np.float16),
                                    mel=mel[:, :T].astype(np.float16))
                n_ok += 1
                if n_ok % verbose_every == 0:
                    print(f"  [{instrument}] {n_ok} pairs", flush=True)
            except Exception as e:
                n_skip += 1
                if n_skip <= 3:
                    print(f"  skip {pathlib.Path(mp).name}: {type(e).__name__}: {e}", flush=True)
    hours = n_ok * seconds / 3600
    print(f"  [{instrument}] DONE {n_ok} pairs (~{hours:.1f}h audio), {n_skip} skipped", flush=True)
    return n_ok
