"""ADetailer (Bing-su/adetailer) re-implemented: detect faces / hands / people, then inpaint each detection.

  {"adetailer": {"units": [{"model": "face_yolov8n.pt", "prompt": "", "negative_prompt": "", "confidence": 0.3,
                            "mask_k_largest": 0, "mask_min_ratio": 0.0, "mask_max_ratio": 1.0, "x_offset": 0,
                            "y_offset": 0, "dilate_erode": 4, "mask_merge_invert": "None", "mask_blur": 4,
                            "denoising_strength": 0.4, "inpaint_only_masked": true, "inpaint_padding": 32,
                            "use_inpaint_width_height": false, "inpaint_width": 512, "inpaint_height": 512,
                            "use_steps": false, "steps": 28, "use_cfg_scale": false, "cfg_scale": 7.0,
                            "use_sampler": false, "sampler": "", "use_noise_multiplier": false}, ...]}}

Models: YOLOv8 detection / segmentation weights (ultralytics), given as a path or as a file name from the
Bingsu/adetailer Hugging Face repo (downloaded on first use, as the extension does), or "opencv_face" (OpenCV's
bundled Haar cascade, no download). `[PROMPT]` in a unit prompt is replaced by the main prompt; `[SEP]` splits the
prompt per detection, like the extension.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

UNIT_DEFAULTS = {"model": "face_yolov8n.pt", "prompt": "", "negative_prompt": "", "confidence": 0.3,
                 "mask_k_largest": 0, "mask_min_ratio": 0.0, "mask_max_ratio": 1.0, "x_offset": 0, "y_offset": 0,
                 "dilate_erode": 4, "mask_merge_invert": "None", "mask_blur": 4, "denoising_strength": 0.4,
                 "inpaint_only_masked": True, "inpaint_padding": 32, "use_inpaint_width_height": False,
                 "inpaint_width": 512, "inpaint_height": 512, "use_steps": False, "steps": 28,
                 "use_cfg_scale": False, "cfg_scale": 7.0, "use_sampler": False, "sampler": "",
                 "use_noise_multiplier": False, "noise_multiplier": 1.0, "color_correction": True}
HF_REPO = "Bingsu/adetailer"
HF_MODELS = ("face_yolov8n.pt", "face_yolov8s.pt", "face_yolov8m.pt", "face_yolov9c.pt", "hand_yolov8n.pt",
             "hand_yolov8s.pt", "hand_yolov9c.pt", "person_yolov8n-seg.pt", "person_yolov8s-seg.pt",
             "person_yolov8m-seg.pt", "deepfashion2_yolov8s-seg.pt")
_yolo_cache: dict = {}


@dataclass
class Detection:
    boxes: list          # [(x1, y1, x2, y2)]
    masks: list          # PIL "L" images, full size
    confidences: list


def normalize_units(cfg: dict) -> list[dict]:
    from .adapters import normalize
    units = cfg.get("units")
    top = {k: v for k, v in cfg.items() if k != "units"}
    if units is None:
        units, top = [top], {}
    out = []
    for u in units or []:
        if not isinstance(u, dict) or str(u.get("model", "")).lower() in ("", "none"):
            continue
        # top-level fields (X/Y/Z overrides) apply to the first unit
        out.append(normalize(dict(u, **top) if not out else u, UNIT_DEFAULTS))
    return out


# ----------------------------------------------------------------------------------------------------- detection
def resolve_model(name: str, model_dirs: Optional[list] = None) -> str:
    if name == "opencv_face" or os.path.isfile(name):
        return name
    for d in model_dirs or []:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    if name in HF_MODELS or name.endswith(".pt"):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(HF_REPO, name)
    raise FileNotFoundError(f"ADetailer model not found: {name}")


def _box_mask(size, boxes) -> list:
    from PIL import Image, ImageDraw
    out = []
    for b in boxes:
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).rectangle([int(v) for v in b], fill=255)
        out.append(m)
    return out


def detect(image, model: str, confidence: float, device: str = "cuda") -> Detection:
    if model == "opencv_face":
        from .extras import face_boxes
        boxes = [(x, y, x + w, y + h) for x, y, w, h in face_boxes(image)]
        return Detection(boxes, _box_mask(image.size, boxes), [1.0] * len(boxes))
    from PIL import Image
    from ultralytics import YOLO
    y = _yolo_cache.get(model)
    if y is None:
        y = _yolo_cache[model] = YOLO(model)
    res = y(image, conf=confidence, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return Detection([], [], [])
    boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
    confs = [float(c) for c in res.boxes.conf.cpu().numpy()]
    if res.masks is not None:
        import cv2
        masks = []
        for m in res.masks.data.cpu().numpy():
            m = cv2.resize((m > 0.5).astype(np.uint8) * 255, image.size, interpolation=cv2.INTER_NEAREST)
            masks.append(Image.fromarray(m, "L"))
    else:
        masks = _box_mask(image.size, boxes)
    return Detection(boxes, masks, confs)


# ------------------------------------------------------------------------------------------------------ masks
def filter_by_ratio(d: Detection, lo: float, hi: float, size) -> Detection:
    area = float(size[0] * size[1])
    keep = [i for i, b in enumerate(d.boxes) if lo <= (b[2] - b[0]) * (b[3] - b[1]) / area <= hi]
    return Detection([d.boxes[i] for i in keep], [d.masks[i] for i in keep], [d.confidences[i] for i in keep])


def filter_k_largest(d: Detection, k: int) -> Detection:
    if k <= 0 or len(d.boxes) <= k:
        return d
    order = sorted(range(len(d.boxes)), key=lambda i: -(d.boxes[i][2] - d.boxes[i][0]) * (d.boxes[i][3] - d.boxes[i][1]))
    keep = sorted(order[:k])
    return Detection([d.boxes[i] for i in keep], [d.masks[i] for i in keep], [d.confidences[i] for i in keep])


def sort_left_to_right(d: Detection) -> Detection:
    order = sorted(range(len(d.boxes)), key=lambda i: (d.boxes[i][0], d.boxes[i][1]))
    return Detection([d.boxes[i] for i in order], [d.masks[i] for i in order], [d.confidences[i] for i in order])


def offset_dilate(mask, x: int, y: int, dilate_erode: int):
    """ADetailer mask preprocessing: shift by (x, -y) (y up is positive), then dilate (>0) / erode (<0)."""
    import cv2
    from PIL import Image
    a = np.asarray(mask.convert("L"))
    if x or y:
        shifted = np.zeros_like(a)
        h, w = a.shape
        dx, dy = int(x), -int(y)
        xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
        ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
        cw, ch = w - abs(dx), h - abs(dy)
        if cw > 0 and ch > 0:
            shifted[yd:yd + ch, xd:xd + cw] = a[ys:ys + ch, xs:xs + cw]
        a = shifted
    if dilate_erode:
        k = np.ones((abs(int(dilate_erode)),) * 2, np.uint8)
        a = cv2.dilate(a, k, iterations=1) if dilate_erode > 0 else cv2.erode(a, k, iterations=1)
    return Image.fromarray(a, "L")


def merge_invert(masks: list, mode: str) -> list:
    from PIL import ImageChops, ImageOps
    mode = (mode or "None").lower()
    if not masks or mode == "none":
        return masks
    m = masks[0]
    for x in masks[1:]:
        m = ImageChops.lighter(m, x)
    if "invert" in mode:
        m = ImageOps.invert(m)
    return [m]


def unit_masks(image, u: dict, device: str, model_dirs=None) -> list:
    d = detect(image, resolve_model(u["model"], model_dirs), u["confidence"], device)
    d = filter_by_ratio(d, u["mask_min_ratio"], u["mask_max_ratio"], image.size)
    d = filter_k_largest(d, int(u["mask_k_largest"]))
    d = sort_left_to_right(d)
    masks = [offset_dilate(m, u["x_offset"], u["y_offset"], u["dilate_erode"]) for m in d.masks]
    return [m for m in merge_invert(masks, u["mask_merge_invert"]) if m.getbbox() is not None]


def match_colors(image, reference):
    """Per-channel LAB mean/std transfer from `reference` (A1111 img2img colour correction, Reinhard style).

    Inpainting re-encodes the crop through the VAE; with many checkpoints' baked VAEs that round trip alone loses
    contrast and shifts colours, so the redrawn region looks washed out next to the untouched pixels."""
    import cv2
    from PIL import Image
    a = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(np.asarray(reference.convert("RGB").resize(image.size)), cv2.COLOR_RGB2LAB).astype(np.float32)
    am, asd = a.mean((0, 1)), a.std((0, 1)) + 1e-6
    rm, rsd = r.mean((0, 1)), r.std((0, 1))
    out = (a - am) / asd * rsd + rm
    return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB), "RGB")


def unit_prompts(text: str, main: str, n: int) -> list[str]:
    """[PROMPT] -> main prompt; [SEP] splits per detection (the last part repeats)."""
    text = (text or "").strip() or main
    parts = [p.strip().replace("[PROMPT]", main) for p in text.split("[SEP]")]
    return [parts[min(i, len(parts) - 1)] for i in range(max(n, 1))]


__all__ = ["match_colors", "UNIT_DEFAULTS", "HF_MODELS", "normalize_units", "resolve_model", "detect", "unit_masks", "unit_prompts",
           "offset_dilate", "merge_invert", "filter_k_largest", "filter_by_ratio", "Detection"]
