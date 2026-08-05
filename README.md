# ReTone

A browser-based DAW for **audio reprocessing** — stem separation, Melodyne/RipX-style
pitch editing, and instrument replacement. ML inference runs on **RunPod Serverless**;
audio lives in **Cloudflare R2**; the editor runs in the browser.

> Status: **M0–M1 foundation** — scaffold + upload → stem separation → multitrack player.
> Later passes add note-based pitch editing (M2), instrument change (M3), and vocal
> regeneration (Phase 2). See `../.claude/plans/` for the full plan.

## Architecture

```
Browser (React/Vite DAW)  ──presigned PUT──►  Cloudflare R2  ◄──pull/push──  RunPod worker (GPU)
        │                                          ▲                              ▲
        └──────► FastAPI backend ──────────────────┘   ──/run + poll /status──────┘
                 (presign, trigger jobs, persist project/stem metadata)
```

The backend never streams audio bytes through itself: the browser uploads directly to R2
via a presigned URL, and the RunPod worker reads/writes R2 directly. The backend only
signs URLs, triggers jobs, and stores metadata.

## Layout

| Path | What |
|---|---|
| `frontend/` | React + Vite + TypeScript DAW UI (WaveSurfer.js v7, Tone.js) |
| `backend/`  | Python FastAPI orchestration API (presign, RunPod, SQLite) |
| `runpod/`   | Serverless GPU worker: `handler.py`, `Dockerfile`, MSST stem separation |
| `infra/`    | R2 bucket/CORS setup + RunPod deploy notes |

## Quick start (local, mock mode — no credentials needed)

```bash
# 1) Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env          # MOCK_MODE=true by default
uvicorn app.main:app --reload --port 8000

# 2) Frontend (in a second terminal)
cd frontend
pnpm install        # or: npm install
pnpm dev            # http://localhost:5173
```

In mock mode the backend returns canned stems so you can exercise the full upload →
"separate" → multitrack-player flow without RunPod or R2.

## Going live (real inference)

1. Fill in `backend/.env` with your **Cloudflare R2** and **RunPod** credentials and set
   `MOCK_MODE=false`.
2. Deploy the worker: see `runpod/README.md` (RunPod GitHub integration).
3. Create the R2 bucket + CORS: see `infra/README.md`.

Cost is ~1–3¢ of GPU per ~4-minute song on a 24 GB serverless GPU; R2 has zero egress.

## Model & license notes

Stem separation runs under ZFTurbo's **MSST** framework:

| Tier | Model | License |
|---|---|---|
| 2-stem | MelBand RoFormer (`KimberleyJSN/melbandroformer`) | MIT ✅ |
| 4-stem | SCNet XL IHF (`scnet`) | verify on release |
| 6-stem | BS-ROFO-SW (`enerjazzer/BS-ROFO-SW-Fixed`) | ⛔ unknown — **prototype only** |

🚩 Before any public launch, resolve/replace BS-ROFO-SW (fallback: Demucs `htdemucs_6s`, MIT).
