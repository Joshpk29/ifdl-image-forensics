# IFDL Image Forensics API

A REST API the Chrome extension calls to check whether an image has been
manipulated, and if so, where — and, if SIDA is enabled, why. It runs up to
three independent forgery-detection models and combines their outputs:

- **[TruFor](https://github.com/grip-unina/TruFor)** (CVPR 2023, GRIP-UNINA) — transformer-based, combines RGB with a learned noise fingerprint.
- **[HiFi-IFDL](https://github.com/CHELSEA234/HiFi_IFDL)** (CVPR 2023, MSU) — the hierarchical fine-grained model this project is named after.
- **[SIDA](https://github.com/hzlsaber/SIDA)** (CVPR 2025, using the `SIDA-7B-description` checkpoint) — a multimodal LLM that classifies, localizes, *and writes an explanation* of why an image looks manipulated (what's off, and which visual cues gave it away: lighting, edges, resolution, shadows). This is the only one of the three that produces text, not just a score + map.

All three run on CPU — no GPU required for any of them, which matters if
you're on Docker Desktop for Mac/Windows, since neither has a path to CUDA
passthrough regardless of what hardware is underneath. SIDA *can* use a GPU
if one is available to Docker (see the comment at the top of
`services/sida/Dockerfile`), but auto-detects and falls back to CPU
otherwise — see "A note on SIDA's speed" below, because CPU inference for a
7B-parameter model is a real tradeoff, not a minor one. Neither TruFor nor
HiFi-IFDL ship as pip packages, so each is built as its own Docker service
(source code cloned from GitHub at build time) and wrapped with a small
FastAPI server; SIDA follows the same pattern. A fourth `orchestrator`
service is the one thing the extension actually talks to — it fans a request
out to whichever model services are running and merges the results.

> **Current status: TruFor is disabled by default.** Its pretrained weights
> are hosted on grip.unina.it, a university server that's currently
> unreachable (confirmed via direct browser test, not a Docker networking
> issue). The `trufor` service is gated behind a Docker Compose profile so it
> doesn't block `docker compose up`. Everything below works with HiFi-IFDL
> alone; see "Re-enabling TruFor" at the bottom once grip.unina.it is back.
> **SIDA is also disabled by default** — not because it needs a GPU (it
> doesn't strictly), but because of the ~14GB one-time model download and
> its CPU inference speed. See "SIDA setup".

```
extension  --POST /analyze-->  orchestrator (8000)
                                    |--> trufor  (8001, internal)
                                    |--> hifi    (8002, internal)
                                    |--> sida    (8003, internal, opt-in)
```

## Response format

`POST /analyze` with `{"image_url": "..."}` (this is exactly what
`background.js` already sends) returns:

```json
{
  "manipulated": true,
  "confidence": 0.665,
  "localization_map_png_base64": "iVBORw0KG...",
  "explanation": {
    "text": "The image is tampered. Type: part tampered. Areas: ...",
    "model": "sida",
    "class": "part_tampered"
  },
  "models": {
    "trufor": { "score": 0.72, "localization_map_png_base64": "..." },
    "hifi_ifdl": { "manipulated": true, "score": 0.61, "localization_map_png_base64": "..." },
    "sida": { "score": 0.85, "score_is_heuristic": true, "class": "part_tampered", "explanation": "...", "localization_map_png_base64": "..." }
  }
}
```

`confidence` is the average of whichever model(s) responded successfully — if
one or two models are down or error, you still get a result from whichever
did rather than a failed request. `localization_map_png_base64` is a
grayscale PNG (every responding model's map resized to a common size and
averaged) where brighter pixels = more likely manipulated. `explanation` is
`null` unless SIDA is enabled and responded — see "A note on SIDA's score"
below for why its `score` is marked `score_is_heuristic` rather than being a
calibrated probability like TruFor/HiFi-IFDL's.

There's also `POST /analyze/upload` (multipart file, for testing with curl)
and `GET /health` on all model services.

## 1. Prerequisites

- Docker and Docker Compose installed
- ~6 GB free disk space (two ML model images plus weights)
- No GPU needed — everything is pinned to CPU-only PyTorch builds, including
  SIDA (see "SIDA setup" below for the extra disk space and speed tradeoff
  that one involves)

## 2. Download the HiFi-IFDL weights (manual step — Google Drive, can't be scripted)

1. Go to the [HiFi_IFDL_weights_link](https://drive.google.com/drive/folders/1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm) linked from the [official README](https://github.com/CHELSEA234/HiFi_IFDL#usage-on-detecting-and-localization-for-the-general-forged-content-including-gan-and-diffusion-generated-images) and download it.
2. Arrange the files so you end up with exactly this layout:
   ```
   api/services/hifi/weights/HRNet/750001.pth
   api/services/hifi/weights/NLCDetection/750001.pth
   ```
   (`weights/` already exists in the repo with a placeholder — just add the
   two subfolders above.)

If you skip this step, the HiFi service still starts and responds — it just
falls back to randomly-initialized weights and logs `weight-loading fails`,
so predictions will be meaningless until the real weights are in place.

## 3. Build and run

```bash
cd api
docker compose up --build
```

This builds and starts `orchestrator` and `hifi` (TruFor is skipped — see the
status note above). First build takes a couple minutes cloning the HiFi_IFDL
repo and installing PyTorch. Subsequent starts are fast.

Once running:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 4. Test it

```bash
curl -F "file=@/path/to/some/image.jpg" http://localhost:8000/analyze/upload | python3 -m json.tool
```

Or with a URL, matching what the extension sends:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/some-image.jpg"}'
```

## 5. Point the extension at it

The extension already defaults to `http://localhost:8000/analyze` — nothing
to change unless you're running the API somewhere other than your own
machine. To change it: extension icon → popup → API endpoint field.

## SIDA setup (optional, CPU by default, slow)

SIDA is a 7B-parameter multimodal LLM — meaningfully heavier than the other
two models, both to run and to download. It's off by default and gated
behind its own Compose profile.

1. Start it:
   ```bash
   docker compose --profile sida up --build
   ```
   On first run this downloads the `saberzl/SIDA-7B-description` checkpoint
   from Hugging Face (~14GB) and caches it in a named Docker volume
   (`sida_hf_cache`), so subsequent restarts don't re-download it. No manual
   download step and no Hugging Face auth/token needed — the checkpoint is
   public. (The upstream SIDA README also lists a separate SAM ViT-H
   checkpoint download, but that's only needed for *training* SIDA from a
   base LISA checkpoint — the released SIDA-7B-description weights already
   include the trained SAM mask decoder, and the official inference scripts
   don't load it separately either. See the comment at the top of
   `services/sida/app.py` if you want the full reasoning.)

2. Once it's up, `orchestrator`'s responses will include a non-null
   `explanation` field, and `models.sida` in the response.

### A note on SIDA's speed

TruFor and HiFi-IFDL are small CNNs — sub-second per image on CPU. SIDA is a
7B-parameter LLM doing a full autoregressive generation pass per image, and
on CPU that's a different order of magnitude: expect low single-digit
**minutes** per image, not seconds, plus a slow first load while ~14GB of
weights come off disk into memory. That's real for a browser extension
scanning a feed of images live.

If you have a Linux machine (not Docker Desktop on Mac/Windows — those can't
do CUDA passthrough) with an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed, or access to a cloud GPU box, SIDA drops to single-digit seconds
per image instead. To switch this service to GPU: swap `services/sida/Dockerfile`'s
base image for an `nvidia/cuda:*-runtime` image, install the `cu117` torch
build from the official repo's `requirements.txt` instead of the CPU wheels
currently there, and add a GPU device reservation back to the `sida` service
in `docker-compose.yml` (`deploy.resources.reservations.devices`). `app.py`
already auto-detects and uses a GPU if `torch.cuda.is_available()` — no
Python changes needed, just the image/compose changes.

If SIDA's CPU speed turns out to be impractical for how you actually use the
extension, the simplest move is leaving `sida` off (the default) and running
just `docker compose up` — TruFor + HiFi-IFDL are fast and already give you
detection + localization, just not the written explanation.

### A note on SIDA's score

TruFor and HiFi-IFDL each return a calibrated `score` between 0 and 1 from
their model's own sigmoid/softmax output. SIDA's public inference API
doesn't expose that — it only returns generated text (e.g. "This image is
classified as tampered.") and, for the tampered/synthetic classes, a
predicted mask. So `services/sida/app.py` derives `score` heuristically from
SIDA's stated classification (real → 0.05, part-tampered → 0.85, fully
synthetic → 0.95) and flags it with `"score_is_heuristic": true` — it's on
the same 0–1 scale as the other two so the orchestrator can still average it
in, but treat SIDA's `class` and `explanation` fields as the trustworthy
part of its output, not `score`.

## Notes on the CPU port

All three upstream repos were released assuming a GPU. TruFor's own test
script already supports `-gpu -1` for CPU cleanly. HiFi-IFDL's and SIDA's
code both hardcode `cuda:0` in a few places (module-level `device` globals,
unconditional `.cuda()` calls, `torch.load` without `map_location`) —
`services/hifi/app.py` and `services/sida/app.py` patch around each of these
at import time, with comments explaining why each patch is needed. This was
verified by tracing every `cuda` reference in the repo and running a full
forward pass on CPU before wiring it into the API.

## License note

TruFor's license restricts use to "informational and nonprofit purposes."
SIDA-7B-description is released under the Llama 2 license (it's fine-tuned
from a Llama-2-based LISA checkpoint), which has its own acceptable-use
restrictions. All three models are used here for a personal, non-commercial
project — worth checking each license's terms if this ever becomes something
you'd ship or monetize.

## Re-enabling TruFor

Once `https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip` is
reachable again (try it directly in a browser first to confirm), bring it
back with:

```bash
docker compose --profile trufor up --build
```

That builds and starts `trufor` alongside the other two. No code changes
needed — the orchestrator already calls it whenever it's reachable and
folds its score into `confidence` and the combined localization map.

## Extending

To add another model to the ensemble: add a new folder under `services/`
with its own Dockerfile + `app.py` exposing `POST /predict` (return at least
`score`, and optionally `localization_map_png_base64` and `explanation`),
add it to `docker-compose.yml`, and add its URL to `orchestrator/main.py`'s
`asyncio.gather` call in `_analyze_bytes` (and, if it produces an
explanation, a branch in `_explanation_of`).
