# ReTone RunPod Serverless worker.
# Base image ships CUDA + PyTorch so audio-separator's GPU path works out of the box.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    SEP_OUTPUT_DIR=/tmp/retone_out

# ffmpeg is required by audio-separator for decode/encode.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.json separator_engine.py download_models.py handler.py ./

# Bake model weights into the image (best-effort; falls back to runtime download).
RUN python download_models.py

# RunPod invokes the handler; the SDK owns the process loop.
CMD ["python", "-u", "handler.py"]
