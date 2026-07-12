# SIDA inference service
# Wraps the official hzlsaber/SIDA (CVPR 2025) model behind a small FastAPI
# server. SIDA is a 7B-parameter multimodal LLM (LLaVA/LISA + SAM ViT-H) that
# does detection, pixel-level localization, AND a written explanation of why
# an image looks manipulated — unlike TruFor/HiFi-IFDL, which only score +
# localize.
#
# This image is CPU-only, like the other two services — see app.py's module
# docstring for how the code auto-detects CPU vs GPU and monkeypatches
# around the vendored upstream code's hardcoded .cuda() calls. If you have a
# Linux host with an NVIDIA GPU + the NVIDIA Container Toolkit and want real
# GPU speed (CPU inference is genuinely slow — see api/README.md), swap the
# base image for an nvidia/cuda:*-runtime image and install a CUDA-enabled
# torch/torchvision build (e.g. the official repo's cu117 pin, or a newer
# cuXXX build matching whichever torch version you land on — see the version
# note below) instead of the CPU wheels below; everything else in this file
# and app.py already supports both.

FROM python:3.10-slim

# build-essential is required here even though this is an inference-only
# service: pycocotools (a transitive requirement) compiles a C extension
# against numpy/Python headers at install time, and python:3.10-slim ships
# no compiler at all — without it, `pip install -r requirements.txt` fails
# with "Failed building wheel for pycocotools".
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN git clone --depth 1 https://github.com/hzlsaber/SIDA.git /app/SIDA

WORKDIR /app/SIDA

# --- CPU-only PyTorch. The official repo pins torch==1.13.1/torchvision==
# 0.14.1 (matched exactly in services/hifi and services/trufor, which stay
# on torch 1.11), but that specific pair has no Linux aarch64 wheels on
# either download.pytorch.org/whl/cpu or PyPI — meaning this fails outright
# on Apple Silicon under Docker Desktop (which builds linux/arm64 images
# natively, not x86_64). The failure mode isn't a clean "no wheel found"
# either: pip's `||` fallback below used to land on a torch/torchvision
# combination that both individually installed but weren't build-compatible
# with each other, crashing at import time with "cannot import name
# '_cast_Long' from torch.onnx.symbolic_opset9" — a version-mismatch error,
# not a SIDA-specific one. torch==2.1.2/torchvision==0.16.2 is the closest
# pin with confirmed manylinux2014_aarch64 wheels on PyPI, and both are well
# within the torch version range transformers==4.31.0, peft==0.4.0, and
# deepspeed==0.14.0 (all pinned in requirements.txt) support.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2 \
    || pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app.py /app/SIDA/serve_app.py

# Model weights (SIDA-7B-description) are pulled from Hugging Face Hub on
# first startup and cached in a named volume (see docker-compose.yml) so
# they survive container restarts/rebuilds — they are NOT baked into this
# image (it's a ~14GB download). No separate SAM weights download is
# needed — see the module docstring in app.py for why.

EXPOSE 8003
CMD ["uvicorn", "serve_app:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-keep-alive", "120"]
