# Polyphonic conversion — model checkpoints

Trained on RunPod A40 (48 GB), 44.1 kHz, ~64 M params (`PianoRollToMel`,
hidden 768 × 8 transformer layers). Val loss is mel L1 + 0.5 × Δ-L1 vs. the
frozen `nvidia/bigvgan_v2_44khz_128band_512x` vocoder's expected input.

| Instrument | File | Step | Val | What it is |
|---|---|---|---|---|
| **Piano** | `piano/best_pre_realdata.pt` | 14 000 | **0.1190** | Best pre-real-data checkpoint. Trained on 260 MAESTRO MIDI × FluidR3 GM patch 0 renders + full augmentation. Cleanest, most confident, but never saw real audio. |
| **Piano** | `piano/latest.pt` | ~70 000 | ~0.16 | Latest. Continued from step 20 000 on **mixed** real+synthetic data (260 synth + 1202 real Disklavier crops from MAESTRO's actual acoustic audio, fetched via `remotezip` from the 108 GB archive). Val is higher because the real-audio distribution is genuinely harder; the model has SEEN real acoustic piano. |
| **Strings** | `strings/best.pt` | 64 000 | **0.1163** | Best strings so far. Same architecture, trained on MAESTRO MIDI × FluidR3 GM patch 48 (strings ensemble). One-timbre training — the residual "shouting" comes from having never seen non-FluidR3 strings. **Next iteration must add multi-soundfont diversity + real strings.** |
| **Strings** | `strings/latest.pt` | ~65 000 | ~0.19 | Latest strings state. |

## Fetching

Weights are on Hugging Face — this repo doesn't ship them (each is 268–759 MB).

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Ganeshveer/retone-poly",
    filename="piano/best_pre_realdata.pt",   # or "strings/best.pt", etc.
)
```

Model repo URL: <https://huggingface.co/Ganeshveer/retone-poly>

## Rerunning training

```bash
# On a fresh pod:
git clone https://github.com/Ganeshveer/retone.git && cd retone/training/poly

# 1. deps (torch, bf16, TF32 all supported on Ampere+):
pip install librosa soundfile pretty_midi tqdm tensorboard basic-pitch remotezip
git clone https://github.com/NVIDIA/BigVGAN.git && pip install -r BigVGAN/requirements.txt
apt-get install -y ffmpeg fluidsynth fluid-soundfont-gm

# 2. data — one of:
#    a) Pure synthetic (fast, ~30 min): python dataprep.py  # renders MAESTRO MIDI through FluidR3
#    b) With real Disklavier audio mixed in: python fetch_real_maestro.py --budget-gb 8

# 3. resume from a published checkpoint:
mkdir -p ckpt/piano ckpt/strings
python -c "from huggingface_hub import hf_hub_download; \
           import shutil; \
           for f in ['piano/latest.pt', 'strings/latest.pt']: \
               p = hf_hub_download('Ganeshveer/retone-poly', f); shutil.copy(p, f'ckpt/{f}')"

# 4. train — resumes from ckpt/<inst>/latest.pt if present; the resume patch
#    (train_poly.py:45) also reads best.pt's val so it isn't clobbered.
RETONE_INSTRUMENT=piano   nohup python train_poly.py > train_piano.log   2>&1 &
RETONE_INSTRUMENT=strings nohup python train_poly.py > train_strings.log 2>&1 &

# 5. inference:
python render_batch_v3.py     # A/B PLAIN vs ARRANGED on 5 songs; see samples_v3/
```

Checkpoint sizes:
- `best.pt` ≈ 269 MB (model + cfg + step + val)
- `latest.pt` ≈ 759 MB (model + optimizer + LR-scheduler for resume)
