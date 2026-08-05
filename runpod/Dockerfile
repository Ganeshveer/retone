# ReTone RunPod Serverless worker.
# Base ships PyTorch 2.7.1 + CUDA 12.8 — has sm_120 kernels so it runs on Blackwell
# (RTX PRO 6000) GPUs as well as Ampere/Ada. audio-separator needs only torch>=2.3, so it
# installs on top WITHOUT downgrading torch (which would drop Blackwell support).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    SEP_OUTPUT_DIR=/tmp/retone_out

# ffmpeg is required by audio-separator for decode/encode.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the repo root (RunPod GitHub integration). Paths are runpod/-prefixed.
COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/models.json runpod/separator_engine.py runpod/download_models.py runpod/handler.py ./

# Bake model weights into the image (best-effort; falls back to runtime download).
RUN python download_models.py

# RunPod invokes the handler; the SDK owns the process loop.
CMD ["python", "-u", "handler.py"]
