# DDSP timbre-transfer checkpoints

This directory ships the trained decoder weights the RunPod worker loads for the DDSP tone-transfer task. Each instrument is one `<name>.pth` (state_dict) + a matching `<name>.yaml` (the exact training config — required for shape correctness at load time; see the "hard rule" below).

## What's here

| File | Instrument | Sample rate | Training | Val loss | License |
|---|---|---|---|---|---|
| `violin48.pth` + `violin48.yaml` | Violin (**default**) | 48 kHz | Own-trained on URMP violin isolated stems (~15 min, single performer per piece), ~34k gradient steps to best val | 8.94 (MSS, `valid_waveform_sec=6`) | CC0 training data; checkpoint free for any use |
| `violin.pth` + `violin.yaml` | Violin (legacy v2) | 16 kHz | Pretrained by [sweetcocoa/ddsp-pytorch](https://github.com/sweetcocoa/ddsp-pytorch), Apache-2.0 | — | Apache-2.0 (upstream) |

The **48 kHz** checkpoint is the default the engine routes to. The 16 kHz one is kept for direct A/B comparison and as a stable fallback while the 48 kHz recipe evolves.

## How the worker loads these

`runpod/ddsp_engine.py` has one dict that maps a public instrument name to a `(pth, yaml)` pair:

```python
MODELS = {
    "violin": ("violin48.pth", "violin48.yaml"),
    "violin_16k_v2": ("violin.pth", "violin.yaml"),
}
```

To ship a new instrument (e.g., a trained flute checkpoint):
1. Drop `flute48.pth` + `flute48.yaml` into this directory.
2. Add a whitelist line to `.gitignore`: `!runpod/ddsp_models/flute48.pth`.
3. Add one line to the `MODELS` dict.
4. Add the instrument name to `DDSP_INSTRUMENTS` in `backend/app/services/tone_transfer.py` so the UI offers it.
5. Commit + push. RunPod's GitHub integration rebuilds the worker image automatically.

## Hard rule: yaml must be byte-exact for the checkpoint's training config

`AutoEncoder(cfg).load_state_dict(state, strict=False)` **silently random-initializes** any layer whose shape doesn't match the checkpoint. If the shipped yaml disagrees with the training config on any capacity key (`n_harmonics`, `n_freq`, `gru_units`, `mlp_units`, `mlp_layers`, `use_z`, `sample_rate`, `frame_resolution`, `use_reverb`), you get **garbage output with no error and no warning** at inference. Never hand-edit these values; always ship the exact yaml produced by the training run.

`strict=False` is still correct here — one buffer (`loudness_extractor.smoothing_window`) is deterministically rebuilt at `__init__`, so a shape mismatch on it is harmless. That's the only legitimate mismatch.

## Rendering a stem locally (sanity check without hitting RunPod)

```python
import soundfile as sf, numpy as np, sys
sys.path.insert(0, "runpod")
import ddsp_engine as de

eng = de.DDSPEngine()      # picks GPU if available, else CPU
y, sr = sf.read("some_mono_source.wav")
out, out_sr = eng.render_mono(np.asarray(y, dtype=np.float32), sr, "violin")
sf.write("rendered.wav", out, out_sr)   # always 44,100 Hz mono
```

CPU renders take roughly real-time-x1 for the 48 kHz model on a modern CPU. On the RunPod worker's GPU, the same job runs in a few seconds even including cold-start.

## Training a new instrument

See [`notebooks/train_ddsp_48k.ipynb`](../../notebooks/train_ddsp_48k.ipynb) for the full training pipeline, verified live end-to-end. It clones sweetcocoa/ddsp-pytorch, applies 8 patches for modern torch/omegaconf/pandas/CUDA compatibility, downloads URMP or reads your own recordings, trains a decoder, and drops the trained `<name>48.pth`/`<name>48.yaml` right back into this directory.

## Violin48 training details (for reproducibility)

- **Data:** URMP dataset (Dryad DOI 10.5061/dryad.ng3r749) — 386 isolated solo stems, native 48 kHz mono WAV, CC0-licensed. Trained on `AuSep_*_vn_*.wav` only (~15 min after silence trimming and split; 90/10 train/test).
- **Preprocessing:** 48 kHz mono, `librosa.effects.trim(top_db=35)`, peak-normalize to −1 dBFS, CREPE-full f0 at 4 ms step.
- **Model capacity:** `n_harmonics=180`, `n_freq=128`, `gru_units=512`, `mlp_layers=3`, `mlp_units=512`, `use_z=False`, `use_reverb=True`.
- **Training:** Adam, lr 1e-3 with multiplicative decay 0.98, `batch_size=32`, `waveform_sec=1.0`, `validation_interval=1000`.
- **Convergence:** Best validation loss (MSS, 6 s window) = **8.94 at step ~34,000**. Continued training past this point did not improve val loss further across 50k+ additional steps — the model has converged. Training loss kept dropping (indicating the model has capacity headroom but the data does not have enough diversity to reward it, which is consistent with URMP's small solo set).
- **GPU used:** RunPod RTX 4000 Ada (20 GB VRAM). VRAM usage: ~7 GB peak.

## Known limitations

- **Monophonic input only.** Feeding a chord (piano, guitar) will pick out one pitch via CREPE and render that voice; the other notes are dropped. Use Engine 2 (AFTER diffusion) for polyphonic instruments — see the notebook's §8 for the plan.
- **Timbre bias.** Trained on URMP's specific violin performers, mics, and (small) rooms. Real-world violin transfer sounds like *that* violin, not a generic one — this is a DDSP-family property, not a bug.
- **f0 correctness sets the ceiling.** CREPE errors show up as audible pitch glitches in the render. On very breathy or noisy source stems, the engine's `f0_threshold=0.5` may fall back to unvoiced and produce silence for that frame — clean up the source's silence first for best results.
