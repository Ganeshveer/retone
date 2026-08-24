# Polyphonic instrument conversion — training code

Working implementation of the Stage-2 renderer from
[`notebooks/train_poly_instrument_conversion.ipynb`](../../notebooks/train_poly_instrument_conversion.ipynb).
These are the files as actually run on an A40 pod, not templates.

```
audio ──▶ [Stage 1: transcribe] ──▶ notes ──▶ [Stage 2: renderer] ──▶ mel ──▶ [BigVGAN] ──▶ audio
              pretrained                        WE TRAIN THIS              frozen, pretrained
```

| File | Role |
|---|---|
| `vocoder.py`  | Loads BigVGAN, bypassing the broken HF mixin (see below) |
| `dataprep.py` | MIDI → rendered audio → aligned `(piano_roll, mel)` pairs |
| `train_lib.py`| Config, dataset, on-the-fly augmentation, model (~64M params), loss |
| `train_poly.py`| Training loop — detached-friendly, resumable |
| `fetch_real_maestro.py` | HTTP-range extraction of real Disklavier audio from the 108 GB MAESTRO zip — pairs with the same MIDI we already render synthetically |
| `arrange.py` | Idiomatic MIDI re-voicing per target instrument (rolled chords / Alberti restrikes for piano, sostenuto for strings) — runs **between** transcription and render |
| `render_batch_v2.py` | Batch A/B renderer (transcribe → arrange → model → BigVGAN) |
| `infer.py`, `run_tests.py` | Single-input inference / on-pod test harness |

## Running

```bash
# 1. deps (torch already present on RunPod PyTorch images)
pip install librosa soundfile pretty_midi tqdm tensorboard basic-pitch
git clone https://github.com/NVIDIA/BigVGAN.git && pip install -r BigVGAN/requirements.txt
apt-get install -y ffmpeg fluidsynth fluid-soundfont-gm

# 2. data — MAESTRO MIDI rendered to the target instrument
python build_data.py            # see the notebook §12

# 3. train
RETONE_INSTRUMENT=piano nohup python train_poly.py > train.log 2>&1 &
```

## Verified configuration (A40, 48 GB)

- **44.1 kHz** end to end — matches `nvidia/bigvgan_v2_44khz_128band_512x`
- ~64M params (hidden 768, 8 layers), batch 64, bf16, TF32, `torch.compile`
- Frame rate 86.13 fps: piano roll and mel are the same length, frame for frame

## Problems hit, and the fixes

**BigVGAN `from_pretrained` raises TypeError.** Its `_from_pretrained` declares
keyword-only `proxies`/`resume_download`; `huggingface_hub` ≥ 1.x stopped passing
them. Rather than pin an old hub (which pulls other conflicts), `vocoder.py`
replicates the mixin: `hf_hub_download` the config + weights, build, `load_state_dict`.

**`nproc` lies on shared pods.** This A40 reports 96 cores; the real cgroup quota is
**7.65**. Sizing `num_workers` from `nproc` spawns ~94 workers and starves the GPU.
`_cpu_quota()` reads cgroup **v1 and v2** — this pod is v1
(`cpu.cfs_quota_us`), the previous one was v2 (`cpu.max`).

**fluidsynth renders past the requested length.** Trimming notes is not enough — it
renders to the last event *of any kind*, so leftover sustain-pedal CCs at t=969s
produced 16 minutes of silence for a 20-second excerpt. Trim `control_changes` too,
and hard-slice the audio as a guarantee.

**Sustain pedal must be folded into the roll.** fluidsynth honours CC64 when
rendering, so the audio has pedal-extended sustain. If the conditioning shows
note-off while the audio still rings, the model trains on a lie. `apply_sustain()`
extends offsets to pedal-release (the Onsets & Frames transform).

**BigVGAN needs volume-normalized input.** Its own loader does
`librosa.util.normalize(audio) * 0.95`. Skipping it puts training-target mels
outside the distribution the vocoder expects — degrades quality quietly, never errors.

## Judging the sanity overfit

Compare against a **constant-mean predictor**, not against the initial random loss.
Measured on real data: constant baseline **1.045**, model reaches **0.14–0.21** —
5–7× better on a target with std 1.21. A residual floor is *expected*: mel contains
noise-floor and reverb detail that note events genuinely do not determine, so
"loss → 0" is not the right bar and chasing it is a misread.

## Beating the "one soundfont, byte-identical every time" trap

`audio = FluidR3(MIDI)` is deterministic, so a model trained purely on synthetic
data learns to imitate one specific soundfont rather than a general instrument.
Two orthogonal countermeasures:

**On-the-fly mel-domain augmentation** (`train_lib._augment`) — perturbations
that vary in real recordings but never vary in soundfont renders:
level offset ±0.35 (≈±3 dB), spectral tilt, low-level Gaussian noise floor,
SpecAugment frequency masking, and velocity jitter on the CONDITIONING (so the
model doesn't treat the soundfont's velocity→timbre curve as gospel).

**Real Disklavier audio pairs** (`fetch_real_maestro.py`) — the MAESTRO audio
zip is 108 GB, but its central directory supports HTTP range requests, so we
extract only what fits in `--budget-gb` (default 8 GB → ~15 h of real acoustic
piano) via `remotezip` without ever downloading the monolith. Each performance
is chopped into overlapping windows and stored as `pair_real_*.npz` alongside
the synthetic `pair_*.npz` — training scans the union and preloads both into
RAM as a single distribution.

## Idiomatic arrangement (`arrange.py`)

The renderer will play exactly what the piano-roll says, note for note. If the
transcriber hands it a 4-note block chord sustained for 1 s and the target is a
piano, the render sounds like a block chord — because that's what it is.
Musicians would never play a sustained string bed that way on a piano.

`arrange_for_piano(pm)`:
- **Rolled onsets** — chords of ≥3 notes get staggered by 25 ms in
  low-to-high order. Ear reads it as motion instead of a strike.
- **Alberti restrikes** — clusters held > 400 ms get inner-voice restrikes
  every 300 ms at 60% velocity, in an alternating outer/inner pattern.
  Underneath the original notes still sustain their full length.

`arrange_for_strings(pm)`:
- **Sostenuto extension** — each note's END is extended to the next onset in
  its voice (nearest neighbor within ±3 semitones), capped at +2 s, with a
  40 ms legato overlap between consecutive notes in a voice.

Both are pure MIDI-to-MIDI transforms, run between Basic Pitch and the model.
Knobs are per-function kwargs — tune them on a per-song basis when needed.

## Bug found in `train_poly.py` resume

The original resume path only reloaded `step` from `latest.pt` and left
`best = float("inf")` in memory. On restart, the first val evaluation was
auto-crowned "new best" and overwrote `best.pt` with a worse model — silently.
Patched: after loading `latest.pt`, also read `best.pt` and seed `best` from
its `val` field. Future restarts protect the on-disk best.

**If you already got clobbered**: keep the pre-restart `best.pt` as
`best_pre_realdata.pt` (`cp` before killing) so the true best is preserved.
