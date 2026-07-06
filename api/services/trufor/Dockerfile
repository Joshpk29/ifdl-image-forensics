# TruFor inference service
# Wraps the official grip-unina/TruFor (CVPR 2023) model behind a small FastAPI server.
# Non-commercial license — see https://github.com/grip-unina/TruFor/blob/main/LICENSE.txt

FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Fetch official inference source (source only, no weights bundled in git) ---
RUN git clone --depth 1 https://github.com/grip-unina/TruFor.git /tmp/trufor_repo && \
    mkdir -p /app && \
    cp -r /tmp/trufor_repo/test_docker/src/* /app/ && \
    rm -rf /tmp/trufor_repo

# --- CPU-only PyTorch, pinned to the version TruFor was released/tested against ---
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==1.11.0+cpu torchvision==0.12.0+cpu \
    || pip install --no-cache-dir torch==1.11.0 torchvision==0.12.0

RUN pip install --no-cache-dir \
        tqdm "yacs>=0.1.8" "timm>=0.5.4" "numpy==1.21.5" \
        fastapi "uvicorn[standard]" python-multipart pillow

# --- Official pretrained weights (direct download, no auth/manual step required) ---
# Extracted to /weights, matching trufor.yaml's TEST.MODEL_FILE: ../weights/trufor.pth.tar
# grip.unina.it (an Italian university research server) occasionally rejects
# requests with no User-Agent or drops connections transiently, so this
# retries a few times with a browser-like UA before giving up.
RUN curl -fL --retry 5 --retry-delay 5 --connect-timeout 15 \
        -A "Mozilla/5.0 (X11; Linux x86_64) TruFor-docker-build" \
        https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip -o /tmp/w.zip && \
    unzip -q /tmp/w.zip -d / && \
    rm /tmp/w.zip

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
