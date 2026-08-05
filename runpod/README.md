# ReTone RunPod worker

GPU stem-separation worker for RunPod Serverless. Pulls audio from R2, separates stems,
writes them back to R2.

## Files

| File | Purpose |
|---|---|
| `handler.py` | RunPod entrypoint; `ping` + `separate` tasks; R2 read/write |
| `separator_engine.py` | Wraps `audio-separator`; tier → model; maps outputs to stem names |
| `models.json` | Per-tier model config (env-overridable) |
| `download_models.py` | Bakes weights into the image at build |
| `Dockerfile` | CUDA + PyTorch + ffmpeg + deps |

## Model tiers (defaults)

| Tier | Default model | Stems | License |
|---|---|---|---|
| `2stem` | BS-RoFormer (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) | vocals, instrumental | model-specific |
| `4stem` | Demucs `htdemucs.yaml` | vocals, drums, bass, other | MIT |
| `6stem` | Demucs `htdemucs_6s.yaml` | vocals, drums, bass, guitar, piano, other | MIT |

Defaults are robust and runnable. To use the higher-quality research picks:
- **4-stem → SCNet XL IHF** and **6-stem → BS-ROFO-SW**: set `MODEL_4STEM` / `MODEL_6STEM`
  to the corresponding checkpoint filename (run `audio-separator --list_models` to see
  supported names), or swap `separator_engine.py` for a thin MSST `inference.py` call.
  🚩 BS-ROFO-SW is "unknown" license — prototype only; keep Demucs 6s as the launch fallback.

## Endpoint environment variables (set in the RunPod console)

```
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=retone
# optional overrides:
# R2_ENDPOINT_URL=https://<acct>.r2.cloudflarestorage.com
# MODEL_4STEM=...  MODEL_6STEM=...
```

## Deploy — Option A: RunPod GitHub integration (recommended)

1. Push this repo to GitHub.
2. RunPod console → Serverless → New Endpoint → **GitHub Repo** → select repo/branch,
   Dockerfile path `runpod/Dockerfile` (build context = repo root; the Dockerfile COPYs
   are `runpod/`-prefixed to match).
3. Pick a **24 GB GPU** (RTX 4090 / A5000). Set the env vars above.
4. RunPod builds + hosts the image; each push redeploys. Copy the **Endpoint ID** into
   `backend/.env` as `RUNPOD_ENDPOINT_ID`, and your account **API key** as `RUNPOD_API_KEY`.

## Deploy — Option B: custom image

```bash
# Run from the repo root (build context = repo root):
docker build --platform linux/amd64 -f runpod/Dockerfile -t <user>/retone-worker:v1 .
docker push <user>/retone-worker:v1
# RunPod console → New Endpoint → Import from Docker Registry → <user>/retone-worker:v1
```

## Local smoke test (no GPU; CPU is slow but proves the handler)

```bash
cd runpod
pip install -r requirements.txt
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=retone
python handler.py --rp_serve_api            # starts a local API mimicking RunPod
# then POST {"input":{"task":"ping"}} to the printed URL
```

## Job contract

Request `input`:
```json
{ "task": "separate", "audio_key": "uploads/<id>/song.mp3",
  "tier": "4stem", "output_prefix": "stems/<id>/", "output_format": "flac" }
```
Response `output`:
```json
{ "stems": [ {"name": "vocals", "key": "stems/<id>/vocals.flac"}, ... ], "tier": "4stem" }
```
