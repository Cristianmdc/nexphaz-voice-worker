# NexPhaz own-voice worker — Chatterbox TTS for RunPod Serverless.
# Based on the runpod pytorch base the reference worker proved out.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# --no-deps first, then the dep list: chatterbox-tts pins clash with the
# base image's torch; this is the install order the reference worker proved.
RUN python -m pip install --no-cache-dir --no-deps chatterbox-tts

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image so cold start skips the multi-GB
# download. snapshot_download needs no GPU — safe on CPU-only CI runners
# (the reference Dockerfile init'd on device='cuda', which only builds on a
# GPU box). If the repo id ever changes, the worker falls back to
# downloading at first boot — slower, not broken.
RUN python - <<'EOF' || echo "bake skipped — will download on first boot"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ResembleAI/chatterbox")
EOF

COPY rp_handler.py /
CMD ["python3", "-u", "rp_handler.py"]
