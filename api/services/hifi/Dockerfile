# HiFi-IFDL inference service
# Wraps the official CHELSEA234/HiFi_IFDL (CVPR 2023) model behind a small FastAPI server.
# This is the model the project is named after: Hierarchical Fine-Grained Image
# Forgery Detection and Localization.
#
# The upstream repo hardcodes cuda:0 in several places (it was only ever released
# with GPU inference in mind). app.py patches around that at import time so it
# runs on CPU — see the comments in app.py for exactly what's patched and why.

FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/CHELSEA234/HiFi_IFDL.git /app/HiFi_IFDL

# --- CPU-only PyTorch pinned to the version in the official environment.yml ---
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==1.11.0+cpu torchvision==0.12.0+cpu \
    || pip install --no-cache-dir torch==1.11.0 torchvision==0.12.0

# numpy is pinned to 1.23.5, not the 1.21.5 in the official environment.yml.
# 1.21.5 still satisfies this codebase's np.int usage (removed in 1.24), but
# it's ABI-incompatible with this torch==1.11.0 wheel's compiled numpy C-API
# expectations — torch imports fine but any .numpy() call inside the model
# crashes with "RuntimeError: Numpy is not available" the first time you
# actually run inference. 1.23.5 satisfies both constraints; scipy/sklearn
# bumped accordingly since they enforce a numpy floor.
RUN pip install --no-cache-dir \
        "numpy==1.23.5" \
        "scipy==1.9.3" \
        "scikit-learn==1.1.3" \
        "matplotlib==3.5.3" \
        "einops==0.6.1" \
        "kmeans-pytorch==0.3" \
        "opencv-python-headless==4.8.1.78" \
        "imageio==2.9.0" \
        "tqdm" \
        "yacs>=0.1.8" \
        pillow \
        fastapi "uvicorn[standard]" python-multipart

WORKDIR /app
COPY app.py /app/app.py

# Model weights (HRNet + NLCDetection checkpoints) are NOT in the git repo —
# they're hosted on Google Drive and must be downloaded manually. Mounted at
# runtime via docker-compose so rebuilding the image isn't required after you
# grab them. See api/README.md for the exact download links and layout:
#   weights/HRNet/750001.pth
#   weights/NLCDetection/750001.pth

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
