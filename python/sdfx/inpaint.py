"""Inpainting pre/post-processing following A1111 (pure PIL, no torch).

- mask invert and gaussian blur
- masked content: 0 fill, 1 original, 2 latent noise (approximated in pixel space), 3 latent nothing (approx: gray)
- 'only masked' (inpaint_full_res): crop around the mask with padding, expanded to the processing aspect ratio,
  processed at width x height, then pasted back into the full image
- final overlay: the original is kept outside the (blurred) mask
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .images import HAVE_PIL, Image, ImageFilter, ImageOps, require_pil, resize_image


def get_crop_region(mask, pad: int = 0) -> Optional[tuple[int, int, int, int]]:
    """Bounding box (x1, y1, x2, y2) of non-zero mask pixels expanded by pad; None if mask is empty."""
    bbox = mask.point(lambda v: 255 if v > 0 else 0).getbbox()
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w, h = mask.size
    return max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)


def expand_crop_region(crop_region, processing_width: int, processing_height: int, image_width: int, image_height: int):
    """A1111 masking.expand_crop_region: grow the region to the processing aspect ratio."""
    x1, y1, x2, y2 = crop_region
    ratio_crop_region = (x2 - x1) / max(1, (y2 - y1))
    ratio_processing = processing_width / processing_height
    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        diff = int(desired_height - (y2 - y1))
        y1 -= diff // 2
        y2 += diff - diff // 2
        if y2 >= image_height:
            d = y2 - image_height
            y2 -= d
            y1 -= d
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        diff = int(desired_width - (x2 - x1))
        x1 -= diff // 2
        x2 += diff - diff // 2
        if x2 >= image_width:
            d = x2 - image_width
            x2 -= d
            x1 -= d
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width
    return x1, y1, x2, y2


def fill(image, mask):
    """A1111 masking.fill: fill the masked area with progressively blurred surrounding colours."""
    image_mod = Image.new("RGBA", (image.width, image.height))
    image_masked = Image.new("RGBa", (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))
    image_masked = image_masked.convert("RGBa")
    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    return image_mod.convert("RGB")


def noise_image(width: int, height: int, seed: int):
    return Image.frombytes("RGB", (width, height), random.Random(seed).randbytes(width * height * 3))


def blur_mask(mask, mask_blur: int):
    if mask_blur and mask_blur > 0:
        return mask.filter(ImageFilter.GaussianBlur(mask_blur))
    return mask


@dataclass
class InpaintJob:
    image: object            # PIL RGB, width x height, sent to the model
    mask: object             # PIL L, width x height, sent to the model (white = repaint)
    original: object         # PIL RGB, the original (possibly resized) image for the overlay
    overlay_mask: object     # PIL L, same size as original
    paste_box: Optional[tuple[int, int, int, int]] = None  # x1,y1,x2,y2 for only-masked mode

    def finalize(self, generated):
        """Paste/overlay a generated width x height image back, returning the final image."""
        gen = generated.convert("RGB")
        if self.paste_box is not None:
            x1, y1, x2, y2 = self.paste_box
            base = self.original.copy()
            base.paste(gen.resize((x2 - x1, y2 - y1), Image.LANCZOS), (x1, y1))
            gen = base
        elif gen.size != self.original.size:
            gen = gen.resize(self.original.size, Image.LANCZOS)
        return Image.composite(gen, self.original, self.overlay_mask)


def prepare(init_image, mask_image, width: int, height: int, resize_mode: int = 0, mask_blur: int = 4,
            inpainting_fill: int = 1, full_res: bool = False, full_res_padding: int = 32, invert: bool = False,
            seed: int = 0) -> InpaintJob:
    require_pil()
    mask = mask_image.convert("L")
    if invert:
        mask = ImageOps.invert(mask)
    if mask.size != init_image.size:
        mask = mask.resize(init_image.size, Image.LANCZOS)

    if full_res:
        blurred_full = blur_mask(mask, mask_blur)
        region = get_crop_region(blurred_full, full_res_padding)
        if region is not None:
            region = expand_crop_region(region, width, height, mask.width, mask.height)
            x1, y1, x2, y2 = region
            m = resize_image(2, blurred_full.crop(region), width, height)
            img = resize_image(2, init_image.crop(region), width, height)
            img = _apply_fill(img, m, inpainting_fill, seed)
            return InpaintJob(img, m, init_image.convert("RGB"), blurred_full, (x1, y1, x2, y2))

    img = resize_image(resize_mode, init_image, width, height)
    m = resize_image(resize_mode, mask, width, height)
    m_blur = blur_mask(m, mask_blur)
    model_img = _apply_fill(img, m_blur, inpainting_fill, seed)
    return InpaintJob(model_img, m_blur, img.convert("RGB"), m_blur, None)


def _apply_fill(img, mask, mode: int, seed: int):
    if mode == 0:
        return Image.composite(fill(img, mask), img, mask)
    if mode == 2:
        return Image.composite(noise_image(img.width, img.height, seed), img, mask.point(lambda v: 255 if v > 127 else 0))
    if mode == 3:
        return Image.composite(Image.new("RGB", img.size, (127, 127, 127)), img, mask.point(lambda v: 255 if v > 127 else 0))
    return img


__all__ = ["get_crop_region", "expand_crop_region", "fill", "prepare", "InpaintJob", "HAVE_PIL"]
