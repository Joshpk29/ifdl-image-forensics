"""
FastAPI wrapper around the official HiFi_IFDL model (CHELSEA234/HiFi_IFDL, CVPR 2023).

The upstream HiFi_Net.py hardcodes `torch.device('cuda:0')` and calls like
`.cuda()`, and utils/utils.py + utils/custom_loss.py hardcode a module-level
`device = torch.device('cuda:0')` used throughout inference. It also calls
`torch.load(...)` without `map_location`, which crashes on a CPU-only machine.

None of this is a training-vs-inference distinction we can just configure away
via an argument — it's baked into the module. So this file monkeypatches around
it in three places, applied BEFORE the upstream modules are imported:

  1. `torch.Tensor.cuda` / `torch.nn.Module.cuda` become no-ops when no GPU is
     present, so unconditional `.cuda()` calls in models/NLCDetection_api.py
     don't crash.
  2. `torch.load` defaults to `map_location='cpu'` when the caller doesn't
     specify one, so loading the bundled HRNet backbone weights doesn't crash.
  3. The `device` globals inside utils.utils and utils.custom_loss are
     reassigned to `torch.device('cpu')`. Their functions look up `device`
     from their own module's globals at call time, so this patch is picked up
     correctly even though HiFi_Net.py imports those functions via `from
     utils.utils import *`.

This was validated by tracing every `cuda`/`.cuda()` reference in the repo and
running the full FENet -> SegNet -> localization forward pass on CPU.
"""
import io
import os
import sys
import base64

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

CPU = torch.device("cpu")
REPO_DIR = "/app/HiFi_IFDL"

# --- 1 & 2: patch before importing anything from the upstream repo ---
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

_orig_torch_load = torch.load


def _cpu_safe_load(*args, **kwargs):
    kwargs.setdefault("map_location", CPU)
    return _orig_torch_load(*args, **kwargs)


torch.load = _cpu_safe_load

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)  # repo uses relative paths for weights/, center/, etc.

import utils.utils as _uutils  # noqa: E402
import utils.custom_loss as _closs  # noqa: E402

# --- 3: retarget the module-level device globals ---
_uutils.device = CPU
_closs.device = CPU

from models.seg_hrnet import get_seg_model  # noqa: E402
from models.seg_hrnet_config import get_cfg_defaults  # noqa: E402
from models.NLCDetection_api import NLCDetection  # noqa: E402
from utils.custom_loss import IsolatingLossFunction, load_center_radius_api  # noqa: E402
from utils.utils import one_hot_label_new, level_1_convert  # noqa: E402


def _load_weights(model, model_dir, initial_epoch):
    """Replaces upstream's restore_weight_helper.

    The released checkpoints were saved from a model wrapped in
    nn.DataParallel, so every key in state_dict['model'] has a "module."
    prefix (e.g. "module.conv1.weight"). We don't wrap in DataParallel here
    (there's only one CPU device, so it would do nothing), which means the
    prefix has to be stripped before load_state_dict — otherwise every key
    mismatches and loading fails silently (upstream's version swallows the
    exception with a bare except).
    """
    weight_path = f"{model_dir}/{initial_epoch}.pth"
    try:
        state_dict = torch.load(weight_path, map_location=CPU)["model"]
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        print(f"{model_dir} weight-loading succeeds: {weight_path}")
    except Exception as e:  # noqa: BLE001 — log and fall back to random init rather than crash the service
        print(f"{model_dir} weight-loading FAILS: {e!r}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_dir}_params: {total_params}")
    return model


class HiFiNetCPU:
    """CPU port of HiFi_Net (see module docstring for why this can't just
    subclass HiFi_Net directly — its __init__ hardcodes cuda:0)."""

    def __init__(self):
        FENet_cfg = get_cfg_defaults()
        FENet = get_seg_model(FENet_cfg).to(CPU)
        SegNet = NLCDetection().to(CPU)
        self.FENet = _load_weights(FENet, "weights/HRNet", 750001)
        self.SegNet = _load_weights(SegNet, "weights/NLCDetection", 750001)
        self.FENet.eval()
        self.SegNet.eval()

        center, radius = load_center_radius_api()
        self.LOSS_MAP = IsolatingLossFunction(center, radius).to(CPU)

    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((256, 256), resample=Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return torch.unsqueeze(tensor, 0)

    def detect(self, image: Image.Image):
        with torch.no_grad():
            img_input = self._transform_image(image)
            output = self.FENet(img_input)
            _mask1_fea, _mask1_binary, _out0, _out1, _out2, out3 = self.SegNet(output, img_input)
            res, prob = one_hot_label_new(out3)
            res = level_1_convert(res)[0]
            return res, prob[0]

    def localize(self, image: Image.Image):
        with torch.no_grad():
            img_input = self._transform_image(image)
            output = self.FENet(img_input)
            mask1_fea, _mask1_binary, _out0, _out1, _out2, _out3 = self.SegNet(output, img_input)
            pred_mask_score = self.LOSS_MAP.inference(mask1_fea)[1].cpu().numpy()
            # 2.3 is the threshold used upstream to separate real/fake pixels
            # in the hyper-sphere feature space (see custom_loss.py).
            binary_mask = (pred_mask_score >= 2.3).astype(np.float32)[0]
            return binary_mask


_model = None


def get_model():
    global _model
    if _model is None:
        _model = HiFiNetCPU()
    return _model


app = FastAPI(title="HiFi-IFDL inference service")


@app.on_event("startup")
def _load_on_startup():
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": "hifi_ifdl"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data))

    model = get_model()
    res, prob = model.detect(image)
    binary_mask = model.localize(image)

    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(
        {
            "model": "hifi_ifdl",
            "manipulated": bool(res),
            "score": float(prob),
            "localization_map_png_base64": mask_b64,
            "map_width": int(binary_mask.shape[1]),
            "map_height": int(binary_mask.shape[0]),
        }
    )
