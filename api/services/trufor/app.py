"""
FastAPI wrapper around the official TruFor model (grip-unina/TruFor, CVPR 2023).

Loads the model once at startup and exposes a single /predict endpoint that
takes an image and returns a manipulation score plus a pixel-level
localization map. Preprocessing mirrors data_core.py from the official repo
exactly (img / 256.0, CHW, no extra normalization).
"""
import io
import base64

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from config import update_config, _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

DEVICE = "cpu"


class _Args:
    opts = None


update_config(config, _Args())

_model = None


def get_model():
    global _model
    if _model is None:
        model_file = config.TEST.MODEL_FILE
        if not model_file:
            raise RuntimeError("TEST.MODEL_FILE not set in trufor.yaml")
        checkpoint = torch.load(model_file, map_location=torch.device(DEVICE))
        model = confcmx(cfg=config)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def preprocess(image: Image.Image) -> torch.Tensor:
    """Matches data_core.myDataset.__getitem__ from the official repo."""
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 256.0
    return tensor.unsqueeze(0)


def run_inference(image: Image.Image):
    model = get_model()
    rgb = preprocess(image).to(DEVICE)
    with torch.no_grad():
        pred, conf, det, _npp = model(rgb)

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

        if det is not None:
            score = torch.sigmoid(det).item()
        else:
            score = float(pred.mean())

    return pred, score


def map_to_png_base64(pred: np.ndarray) -> str:
    heat = np.clip(pred * 255.0, 0, 255).astype("uint8")
    img = Image.fromarray(heat, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


app = FastAPI(title="TruFor inference service")


@app.on_event("startup")
def _load_on_startup():
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": "trufor"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data))

    pred, score = run_inference(image)

    return JSONResponse(
        {
            "model": "trufor",
            "score": float(score),
            "localization_map_png_base64": map_to_png_base64(pred),
            "map_width": int(pred.shape[1]),
            "map_height": int(pred.shape[0]),
        }
    )
