"""
Orchestrator API — the single endpoint the Chrome extension talks to.

POST /analyze accepts either:
  - JSON: {"image_url": "https://..."}   (what the extension currently sends)
  - multipart/form-data with a `file` field (for direct testing, e.g. curl)

It fetches the image, sends it to both model services (TruFor and HiFi-IFDL)
in parallel, and combines their outputs into one response. If a model service
is unreachable or errors, its result is omitted rather than failing the whole
request — you still get a result from whichever model succeeded.
"""
import asyncio
import base64
import io
import os
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

TRUFOR_URL = os.environ.get("TRUFOR_URL", "http://trufor:8001")
HIFI_URL = os.environ.get("HIFI_URL", "http://hifi:8002")
MANIPULATED_THRESHOLD = float(os.environ.get("MANIPULATED_THRESHOLD", "0.5"))
REQUEST_TIMEOUT = float(os.environ.get("MODEL_REQUEST_TIMEOUT", "60"))

app = FastAPI(title="IFDL Image Forensics API")

# The extension calls this from content scripts running on arbitrary feed
# sites, so allow any origin. Nothing here is authenticated — see README for
# why that's an acceptable tradeoff for a local personal-use API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    image_url: str


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _fetch_image_bytes(image_url: str) -> bytes:
    # trust_env=False avoids httpx auto-detecting a proxy from the environment
    # (e.g. a SOCKS proxy), which would otherwise require the optional
    # `socksio` dependency just to be ignored. Feed images are plain public
    # CDN URLs, so a direct connection is the right default here.
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True, trust_env=False) as client:
        resp = await client.get(image_url)
        resp.raise_for_status()
        return resp.content


async def _call_model(client: httpx.AsyncClient, base_url: str, image_bytes: bytes) -> Optional[dict]:
    try:
        resp = await client.post(
            f"{base_url}/predict",
            files={"file": ("image.png", image_bytes, "application/octet-stream")},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — a down/slow model shouldn't fail the request
        return {"error": str(exc)}


def _decode_map(b64_png: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("L")
    return np.asarray(img).astype(np.float32) / 255.0


def _combine_localization_maps(results: list[dict]) -> Optional[str]:
    """Resize every model's map to a common size and average them into one
    overlay so the extension only has to render a single heatmap."""
    maps = []
    target_size = None
    for r in results:
        if "localization_map_png_base64" not in r:
            continue
        img = Image.open(io.BytesIO(base64.b64decode(r["localization_map_png_base64"]))).convert("L")
        if target_size is None:
            target_size = img.size
        else:
            img = img.resize(target_size)
        maps.append(np.asarray(img).astype(np.float32) / 255.0)

    if not maps:
        return None

    combined = np.mean(maps, axis=0)
    heat = np.clip(combined * 255.0, 0, 255).astype("uint8")
    out_img = Image.fromarray(heat, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _score_of(result: dict) -> Optional[float]:
    if "score" in result:
        return float(result["score"])
    return None


async def _analyze_bytes(image_bytes: bytes) -> dict:
    # trust_env=False: these are calls to internal Docker-network services
    # (trufor/hifi), so any HTTP_PROXY/SOCKS proxy set in the environment
    # should not apply here — and if it's a SOCKS proxy, httpx would otherwise
    # require the optional `socksio` dependency just to ignore it.
    async with httpx.AsyncClient(trust_env=False) as client:
        trufor_task = _call_model(client, TRUFOR_URL, image_bytes)
        hifi_task = _call_model(client, HIFI_URL, image_bytes)
        trufor_result, hifi_result = await asyncio.gather(trufor_task, hifi_task)

    per_model = {"trufor": trufor_result, "hifi_ifdl": hifi_result}

    ok_results = [r for r in (trufor_result, hifi_result) if r and "error" not in r]
    scores = [s for s in (_score_of(r) for r in ok_results) if s is not None]

    if not scores:
        raise HTTPException(
            status_code=502,
            detail={"message": "Both model services failed", "models": per_model},
        )

    confidence = float(sum(scores) / len(scores))
    manipulated = confidence >= MANIPULATED_THRESHOLD

    return {
        "manipulated": manipulated,
        "confidence": confidence,
        "localization_map_png_base64": _combine_localization_maps(ok_results),
        "models": per_model,
    }


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    try:
        image_bytes = await _fetch_image_bytes(payload.image_url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {exc}")

    return await _analyze_bytes(image_bytes)


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Convenience endpoint for testing directly, e.g.:
    curl -F file=@test.jpg http://localhost:8000/analyze/upload
    """
    image_bytes = await file.read()
    return await _analyze_bytes(image_bytes)
