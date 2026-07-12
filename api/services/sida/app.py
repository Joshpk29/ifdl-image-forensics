"""
FastAPI wrapper around the official hzlsaber/SIDA model (CVPR 2025):
https://github.com/hzlsaber/SIDA

SIDA is a large multimodal model (LLaVA/LISA backbone + a SAM ViT-H mask
decoder + a 3-way classification head) that, unlike TruFor/HiFi-IFDL, does
three things in one pass:
  1. Classifies the image as real / fully-synthetic / part-tampered.
  2. If not real, predicts a pixel-level mask of the manipulated region.
  3. Writes an explanation of *why* it thinks so (what looks wrong, and
     which visual cues gave it away — lighting, edges, resolution, shadows)
     — this is what the "-description" checkpoint variant adds over plain
     SIDA, and is the reason this project uses SIDA-7B-description rather
     than the base SIDA-7B model.

This file adapts the official `chat_description.py` demo — an interactive
REPL that loads the model once and then prompts a human for an image path
on every turn — into a single `/predict` endpoint that loads the model once
at startup and serves requests. The model-loading sequence and the
`model.evaluate(...)` call are kept as close to the original demo script as
possible (the only real departure is device/dtype handling, generalized
from the demo's hardcoded `.bfloat16().cuda()` to the DEVICE/TORCH_DTYPE
auto-detection below) specifically because this is real upstream
LLaVA/LISA-derived code with a lot of surface area for subtle breakage —
deviating from a script the authors actually shipped and tested is a bigger
risk than it looks.

Notably, this does NOT pass a separate `vision_pretrained=` (SAM ViT-H
checkpoint) argument, even though the SIDA repo's README lists downloading
`sam_vit_h_4b8939.pth` as a prerequisite. That flag only matters when
building SAM from a base LISA checkpoint for *training*; the released
SIDA-7B-description weights already contain the trained SAM mask-decoder
in their own state dict, and the official inference demo (both `chat.py`
and `chat_description.py`) never passes it either. Passing a stray SAM
checkpoint here would just be overwritten by `from_pretrained`'s state dict
load anyway.

This service auto-detects GPU vs CPU (`torch.cuda.is_available()`) rather
than requiring one. GPU is strongly preferred — a 7B-parameter model doing a
full autoregressive generate() pass is legitimately slow on CPU (think
minutes per image, not the sub-second/several-second numbers TruFor/HiFi-IFDL
and a GPU-backed SIDA give you). It still runs on CPU, though, for the same
reason HiFi-IFDL does: not everyone has a CUDA GPU available (e.g. Docker
Desktop on a Mac has no path to CUDA passthrough at all, regardless of host
hardware).

Running on CPU requires one more trick beyond just moving tensors to
`cpu`: `SIDAForCausalLM.evaluate()` and other methods in the *vendored*
model/SIDA_description.py (cloned from the upstream repo at build time, not
something this project maintains) hardcode `.cuda()` calls internally on
tensors they construct themselves — the exact same problem documented at
length in `services/hifi/app.py`. The fix is the same one used there:
monkeypatch `torch.Tensor.cuda`/`torch.nn.Module.cuda` into no-ops when no
GPU is present, applied before importing the upstream modules, so those
internal `.cuda()` calls silently keep tensors on CPU instead of crashing
with "Torch not compiled with CUDA enabled".
"""
import base64
import io
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

MODEL_ID = os.environ.get("SIDA_MODEL_ID", "saberzl/SIDA-7B-description")
VISION_TOWER = os.environ.get("SIDA_VISION_TOWER", "openai/clip-vit-large-patch14")
IMAGE_SIZE = int(os.environ.get("SIDA_IMAGE_SIZE", "1024"))
MAX_NEW_TOKENS = int(os.environ.get("SIDA_MAX_NEW_TOKENS", "512"))

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
# bf16 is what the paper/official demo use, but CPU bf16 op coverage for
# custom repos like this one (odd ops in the LLaVA/SAM code paths) is
# inconsistent across torch versions/platforms — fp32 is the safe,
# well-supported choice on CPU. On GPU, stick with bf16 (matches upstream,
# and halves memory/compute versus fp32).
TORCH_DTYPE = torch.bfloat16 if HAS_CUDA else torch.float32

# See the module docstring: the vendored upstream code hardcodes .cuda()
# calls on tensors it constructs internally (inside SIDAForCausalLM.evaluate
# and friends). Patching these to no-ops before importing that code is the
# same fix services/hifi/app.py uses for the identical problem.
if not HAS_CUDA:
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

from model.SIDA_description import SIDAForCausalLM  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from model.llava.mm_utils import tokenizer_image_token  # noqa: E402
from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
from utils.utils import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

# Matches the prompt structure demonstrated in the official README (the
# -description checkpoint was fine-tuned to respond to a prompt shaped like
# this — asking it something unrelated in structure gets much worse output).
PROMPT_TEXT = (
    "Please answer beginning with [CLS] for classification. If the image is "
    "tampered or fully synthetic, output a mask for the manipulated region "
    "and explain why, covering the type of tampering, the affected areas, "
    "the tampered content, and any visual inconsistencies in lighting, "
    "edges, resolution, or shadows."
)


def remove_repeated_tags(text: str) -> str:
    """Copied verbatim from the official chat_description.py post-processing
    step — the model occasionally repeats a <tag>content</tag> pair."""
    tag_content_pattern = r"<([^>]+)>([^<]+)"
    matches = re.findall(tag_content_pattern, text)
    if not matches:
        return text
    first_tag = f"<{matches[0][0]}>"
    parts = text.split(first_tag, 1)
    prefix = parts[0] if len(parts) > 1 else ""
    seen_pairs = set()
    unique_segments = []
    for tag, content in matches:
        key = (tag, content.strip())
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_segments.append(f"<{tag}>{content}")
    return prefix + "".join(unique_segments)


def _preprocess_sam(
    x: torch.Tensor,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=IMAGE_SIZE,
) -> torch.Tensor:
    """Matches preprocess() in the official chat_description.py: normalize
    with SAM's fixed pixel mean/std, then zero-pad to img_size x img_size."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


def classify_from_text(text_output: str, has_mask: bool) -> tuple[str, bool, float]:
    """SIDA's public evaluate() surfaces its classification only as
    generated text (e.g. "[CLS] This image is classified as tampered."),
    not as a returned probability — model/SIDA_description.py computes
    classification logits internally but evaluate() doesn't return them.
    So `score` here is a heuristic mapped from the stated class, kept on the
    same 0-1 "higher = more likely manipulated" scale as TruFor/HiFi-IFDL's
    real scores so the orchestrator can still average them, but it is NOT a
    calibrated probability the way theirs are. The `class` and `explanation`
    fields are the trustworthy part of SIDA's output.
    """
    lower = text_output.lower()
    if "full" in lower and "synthetic" in lower:
        return "full_synthetic", True, 0.95
    if "tampered" in lower or "manipulat" in lower or has_mask:
        return "part_tampered", True, 0.85
    if "real" in lower or "authentic" in lower:
        return "real", False, 0.05
    # Unparseable classification text — fall back on whether a mask was
    # produced at all (the model only emits [SEG]/a mask for its tampered
    # class), rather than silently defaulting to "real".
    if has_mask:
        return "part_tampered", True, 0.7
    return "unknown", False, 0.5


class SIDAService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir=None, model_max_length=1024, padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens("[END]")
        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        cls_token_idx = self.tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

        self.model = SIDAForCausalLM.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True,
            vision_tower=VISION_TOWER,
            seg_token_idx=seg_token_idx,
            cls_token_idx=cls_token_idx,
            torch_dtype=TORCH_DTYPE,
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(DEVICE)

        try:
            self.model.get_model().initialize_vision_modules(self.model.get_model().config)
            self.model.get_model().get_vision_tower().to(dtype=TORCH_DTYPE)
        except AttributeError:
            pass  # already initialized as part of this checkpoint's from_pretrained load

        self.model = self.model.to(dtype=TORCH_DTYPE, device=DEVICE)
        self.model.eval()

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(IMAGE_SIZE)

    def _build_prompt(self) -> str:
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT_TEXT
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        return conv.get_prompt()

    def predict(self, image_rgb: np.ndarray) -> dict:
        original_size_list = [image_rgb.shape[:2]]

        image_clip = (
            self.clip_image_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0)
            .to(device=DEVICE, dtype=TORCH_DTYPE)
        )

        image_sam = self.transform.apply_image(image_rgb)
        resize_list = [image_sam.shape[:2]]
        image_sam = (
            _preprocess_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(device=DEVICE, dtype=TORCH_DTYPE)
        )

        input_ids = tokenizer_image_token(self._build_prompt(), self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                image_clip,
                image_sam,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=MAX_NEW_TOKENS,
                tokenizer=self.tokenizer,
            )

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = remove_repeated_tags(text_output)
        explanation = (
            text_output.replace("[CLS]", "").replace("[SEG]", "")
            .replace("<s>", "").replace("</s>", "").strip()
        )

        mask_b64 = None
        map_width = map_height = None
        valid_mask = pred_masks and pred_masks[0].shape[0] > 0
        if valid_mask:
            mask = pred_masks[0].detach().float().cpu().numpy()[0] > 0
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            map_height, map_width = mask.shape

        cls_name, manipulated, score = classify_from_text(text_output, bool(valid_mask))

        return {
            "model": "sida",
            "manipulated": manipulated,
            "class": cls_name,
            "score": score,
            "score_is_heuristic": True,
            "explanation": explanation,
            "raw_text_output": text_output,
            "localization_map_png_base64": mask_b64,
            "map_width": map_width,
            "map_height": map_height,
        }


_service = None


def get_service() -> "SIDAService":
    global _service
    if _service is None:
        _service = SIDAService()
    return _service


app = FastAPI(title="SIDA inference service")


@app.on_event("startup")
def _load_on_startup():
    get_service()


@app.get("/health")
def health():
    return {"status": "ok", "model": "sida", "checkpoint": MODEL_ID, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image_rgb = np.array(image)

    result = get_service().predict(image_rgb)
    return JSONResponse(result)
