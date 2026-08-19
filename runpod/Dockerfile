# ReTone RunPod Serverless worker.
# Base ships PyTorch 2.7.1 + CUDA 12.8 — has sm_120 kernels so it runs on Blackwell
# (RTX PRO 6000) GPUs as well as Ampere/Ada. audio-separator needs only torch>=2.3, so it
# installs on top WITHOUT downgrading torch (which would drop Blackwell support).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    SEP_OUTPUT_DIR=/tmp/retone_out

# ffmpeg: decode/encode. build-essential + python headers: the new base uses Python 3.11,
# and Demucs's `diffq` dependency has no 3.11 wheel so it compiles from source (needs gcc).
# git: lets RunPod capture the commit for the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /opt/conda/compiler_compat/ld   # conda ld shim breaks system-gcc C-ext builds

WORKDIR /app

# Build context is the repo root (RunPod GitHub integration). Paths are runpod/-prefixed.
COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# torchaudio matched to the base torch 2.7.1 (cu128) — used by the DDSP f0 extractor.
RUN pip install --no-cache-dir torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY runpod/models.json runpod/separator_engine.py runpod/download_models.py runpod/handler.py runpod/ddsp_engine.py runpod/after_engine.py ./
# DDSP timbre transfer: vendored model code + baked weights (violin + CREPE).
COPY runpod/vendor ./vendor
COPY runpod/ddsp_models ./ddsp_models
# AFTER polyphonic timbre transfer: reference audio clips (small — committed) + model
# weights (large .ts, ~1 GB total across 4 checkpoints — fetched at build via download_models.py).
COPY runpod/after_refs ./after_refs
COPY runpod/after_models ./after_models

# Bake separation + AFTER model weights into the image (best-effort; falls back to runtime download).
RUN python download_models.py

# RunPod invokes the handler; the SDK owns the process loop.
CMD ["python", "-u", "handler.py"]
