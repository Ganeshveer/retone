# ReTone — Handover / Onboarding

Browser-based DAW (Melodyne/RipX-style) for stem separation, pitch editing, and ML timbre transfer. Repo: [github.com/Ganeshveer/retone](https://github.com/Ganeshveer/retone) (public).

## What it does

1. Upload a song → ML stem separation (2/4/6-stem, RunPod GPU) → open in a browser DAW.
2. Toggle/mute/solo stems with waveforms; Melodyne-style pitch editing (drag notes, correct-to-scale).
3. Change a stem's instrument — either replay its transcribed notes on a sampler, or (new) **re-voice the actual audio** as another instrument via ML timbre transfer, preserving the original performance.

Design: dark DAW canvas, amber accent, monospace numerics, colored waveform clips ("retro 2010s" aesthetic).

## Architecture

```
frontend/  React + Vite + TS — custom Web Audio engine + canvas waveforms (no WaveSurfer)
backend/   FastAPI — projects, presigned R2 uploads, RunPod job orchestration, note analysis
runpod/    Serverless GPU worker (Docker) — separation, transcription, DDSP timbre transfer
notebooks/ Standalone training notebook(s), run on a rented GPU pod, not part of the live app
```

Storage: Cloudflare R2 (S3-compatible). Inference: RunPod Serverless, autoscale-to-zero (`workersMax` set to N before a job, back to 0 after — this is the cost-control mechanism; don't leave `workersStandby` > 0, that's paid idle capacity).

## Current status

- **M0–M3 done and working**: upload → 6-stem separation (verified via RunPod, real GPU) → multitrack player → pitch editing (Signalsmith Stretch, formant-preserving) → polyphonic note transcription (Basic Pitch) → sampler-based instrument change (smplr).
- **DDSP timbre transfer v2 — deployed and verified live**: re-voices a mono stem (vocals/bass/lead) as **violin**, preserving the exact pitch curve/dynamics (not MIDI-replay). 16 kHz model (sweetcocoa/ddsp-pytorch, vendored + ported to `torch.fft`). Verified end-to-end on RunPod endpoint `tzmhl20ptjin95`: 79s render incl. cold start, output tracks source pitch (0.0 semitone offset, log-pitch corr 0.57), autoscales back to 0 workers after.
- **v3 in progress (user-driven)**: training higher-fidelity **48 kHz** models for violin/flute/saxophone/trumpet/bass, one-time, on a rented GPU pod, via `notebooks/train_ddsp_48k.ipynb`. Drop-in replacement — zero worker code change beyond the `MODELS` dict in `runpod/ddsp_engine.py`.
- **Not yet built**: instrument change for **piano/guitar** (polyphonic — DDSP-mono is architecturally wrong for these, see below) and **drums**.

## The 3-engine plan for timbre transfer (important — don't try to unify these)

Verified by research: one model can't be highest-quality across all stem types. Multi-voice DDSP is a dead end for struck/plucked strings (no sympathetic resonance) and percussion (no f0 to drive it).

| Engine | Stems | Model | Status |
|---|---|---|---|
| **1** | violin, flute, sax, trumpet, bass, mono vocals/lead | DDSP (sweetcocoa, harmonic+noise, audio-driven f0/loudness) | **Live** (v2, 16kHz); v3 (48kHz) training now |
| **2** | piano, guitar (polyphonic) | AFTER (ACIDS-IRCAM control-transfer diffusion; ships Maestro + GuitarSet checkpoints) | Not started — task #16 |
| **3** | drums | Inverse Drum Machine (onset+velocity analysis → one-shot kit swap) | Not started — task #17 |

All three share one philosophy (analyze input → resynthesize, timbre decoupled from performance) but are different models per family — this was a deliberate research-backed decision, not a shortcut.

## Running it locally

The preview/dev-server sandbox **cannot reach** `~/Downloads/retone` (permission denied) — run dev servers via a background `Bash` call, not the preview tool.

```bash
cd backend && source .venv/bin/activate && uvicorn app.main:app --reload   # http://localhost:8000
cd frontend && npm run dev                                                  # http://localhost:5173
```

Secrets live in `backend/.env` (gitignored, never commit): `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID` (`tzmhl20ptjin95`), `R2_ACCOUNT_ID`/`R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`/`R2_BUCKET`/`R2_ENDPOINT_URL`, `MOCK_MODE=false`. Without these the backend still runs in mock mode (`MOCK_MODE=true`, local disk, fake separation) — useful for pure frontend work.

## Key files

- `runpod/ddsp_engine.py` — DDSP inference (`render_mono`); `MODELS` dict maps instrument → `(pth, yaml)`. This is the only file v3 training changes.
- `runpod/vendor/ddsp_pytorch/` — vendored sweetcocoa/ddsp-pytorch, `torch.fft`-ported (pre-1.8 `torch.rfft`/`irfft` removed upstream).
- `runpod/handler.py` — task dispatch (`separate`, `analyze` implicitly via backend, `tone_transfer`).
- `backend/app/services/tone_transfer.py` — autoscale up → submit RunPod job → poll → presign → scale to 0.
- `backend/app/routers/projects.py` — REST routes, incl. `POST /stems/{stem}/tone-transfer`, `GET /meta/ddsp-instruments`.
- `frontend/src/components/NoteEditor.tsx` / `DawView.tsx` — instrument dropdown (sampler + DDSP optgroups), buffer-swap wiring.
- `notebooks/train_ddsp_48k.ipynb` — self-contained; clone the repo on a RunPod GPU pod and run top-to-bottom. Includes a 10-min sanity check that round-trips through the real worker engine before committing to a full training run.

## Known gotchas (learned the hard way — don't rediscover these)

- **RunPod cost control**: `workersStandby` is *paid idle capacity* — always 0 unless a job is actively running. `runpod_autoscale=True` in settings handles this automatically.
- **Worker `strict=False` footgun**: a shipped `.yaml` must be byte-identical to the training config on every capacity key (`n_harmonics`, `n_freq`, `gru_units`, etc.) — mismatches are *silently* ignored (random-init garbage, no error), never hand-edit capacity values.
- **Large-file git push (85 MB `crepe-full.pth` baked into the worker image)** intermittently 408s on some networks — fix: `git config http.version HTTP/1.1` (and if that still 408s on a force-push after a history rewrite, `git fetch origin` first so the push can build a thin delta instead of resending everything).
- **Git identity**: repo-local `user.name`/`user.email` is set to `Ganeshveer <ganeshveer.pattamsetty@gmail.com>` — history was once rewritten (`filter-branch` + force-push) to remove a different, accidentally-used work email. Don't let a different global git identity leak back in.
- **Dryad dataset downloads** (used for DDSP training data) are gated: the versioned API needs an OAuth bearer token, and the public web route sits behind an Anubis anti-bot JS challenge — plain `curl`/`wget` will silently fetch an HTML page, not the data. See the notebook's §2.0 for the two working routes.
- **Signalsmith Stretch pitch-render fix** (`ctx.suspend(0)` before `startRendering()` to avoid an AudioWorklet deadlock) — implemented but not re-verified since.

## Open tasks

- Re-verify the Signalsmith `suspend(0)` pitch-render fix (M2, in progress).
- **Engine 2**: integrate AFTER (piano/guitar) into the worker as a new `tone_transfer` mode — inference-only, pretrained checkpoints, no training needed.
- **Engine 3**: integrate Inverse Drum Machine (drums) similarly.
- v3: finish training + drop in 48kHz violin/flute/sax/trumpet/bass models once the user's GPU-pod training run completes.

## What someone picking this up needs access to

- RunPod account (for the endpoint + billing)
- Cloudflare R2 account (for the bucket)
- GitHub access to `Ganeshveer/retone`
- A free Dryad account only if continuing v3 training with URMP data (see notebook §2.0)
