# IFDL Image Forensics API

A REST API the Chrome extension calls to check whether an image has been
manipulated, and if so, where. It runs two independent forgery-detection
models and combines their outputs:

- **[TruFor](https://github.com/grip-unina/TruFor)** (CVPR 2023, GRIP-UNINA) — transformer-based, combines RGB with a learned noise fingerprint.
- **[HiFi-IFDL](https://github.com/CHELSEA234/HiFi_IFDL)** (CVPR 2023, MSU) — the hierarchical fine-grained model this project is named after.

Both run on CPU. Neither repo ships as a pip package, so each is built as its
own Docker service (source code cloned from GitHub at build time) and wrapped
with a small FastAPI server. A third `orchestrator` service is the one thing
the extension actually talks to — it fans a request out to both models and
merges the results.

> **Current status: TruFor is disabled by default.** Its pretrained weights
> are hosted on grip.unina.it, a university server that's currently
> unreachable (confirmed via direct browser test, not a Docker networking
> issue). The `trufor` service is gated behind a Docker Compose profile so it
> doesn't block `docker compose up`. Everything below works with HiFi-IFDL
> alone; see "Re-enabling TruFor" at the bottom once grip.unina.it is back.

```
extension  --POST /analyze-->  orchestrator (8000)
                                    |--> trufor  (8001, internal)
                                    |--> hifi    (8002, internal)
```

## Response format

`POST /analyze` with `{"image_url": "..."}` (this is exactly what
`background.js` already sends) returns:

```json
{
  "manipulated": true,
  "confidence": 0.665,
  "localization_map_png_base64": "iVBORw0KG...",
  "models": {
    "trufor": { "score": 0.72, "localization_map_png_base64": "..." },
    "hifi_ifdl": { "manipulated": true, "score": 0.61, "localization_map_png_base64": "..." }
  }
}
```

`confidence` is the average of whichever model(s) responded successfully — if
one model is down or errors, you still get a result from the other rather
than a failed request. `localization_map_png_base64` is a grayscale PNG (both
models' maps resized to a common size and averaged) where brighter pixels =
more likely manipulated.

There's also `POST /analyze/upload` (multipart file, for testing with curl)
and `GET /health` on all three services.

## 1. Prerequisites

- Docker and Docker Compose installed
- ~6 GB free disk space (two ML model images plus weights)
- No GPU needed — everything is pinned to CPU-only PyTorch builds

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

## Notes on the CPU port

Both upstream repos were released assuming a GPU. TruFor's own test script
already supports `-gpu -1` for CPU cleanly. HiFi-IFDL's code hardcodes
`cuda:0` in a few places (module-level `device` globals, unconditional
`.cuda()` calls, `torch.load` without `map_location`) — `services/hifi/app.py`
patches around each of these at import time, with comments explaining why
each patch is needed. This was verified by tracing every `cuda` reference in
the repo and running a full forward pass on CPU before wiring it into the API.

## License note

TruFor's license restricts use to "informational and nonprofit purposes."
Both models are used here for a personal, non-commercial project — worth
keeping in mind if this ever becomes something you'd ship or monetize.

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

To add a third model to the ensemble: add a new folder under `services/`
with its own Dockerfile + `app.py` exposing `POST /predict` (return at least
`score` and optionally `localization_map_png_base64`), add it to
`docker-compose.yml`, and add its URL to `orchestrator/main.py`'s
`asyncio.gather` call in `_analyze_bytes`.
