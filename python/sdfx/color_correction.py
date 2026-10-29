"""Reference-guided color transforms, independent of diffusion and model loading.

AdaIN is implemented from the channel mean/variance normalization definition.
The A1111 method follows its AGPL-3.0 processing pipeline, pinned in the
repository's color-correction investigation document.
"""
import math

import numpy as np
from PIL import Image


def a1111(image, reference, strength=1.0):
    """LAB histogram transfer followed by the original image's luminosity."""
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite and between 0 and 1')
    if strength == 0:
        return image.copy()
    import cv2
    from skimage.exposure import match_histograms
    from blendmodes.blend import blendLayers, BlendType
    original = image.convert('RGB')
    target_lab = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2LAB)
    reference_lab = cv2.cvtColor(np.asarray(reference.convert('RGB')), cv2.COLOR_RGB2LAB)
    matched = match_histograms(target_lab, reference_lab, channel_axis=2)
    # match_histograms retains uint8 for multichannel uint8 inputs. Keep that
    # representation: float LAB uses a different OpenCV range convention.
    corrected = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_LAB2RGB).astype(np.uint8))
    corrected = blendLayers(corrected, original, BlendType.LUMINOSITY).convert('RGB')
    return corrected if strength == 1 else Image.blend(original, corrected, strength)


def adain(image, reference, strength=1.0):
    """Transfer RGB channel statistics, resizing the reference to the target.

    Constant target channels map to the reference mean. Output is RGB; alpha is
    not processed. A zero strength bypass preserves the input mode and pixels.
    """
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite and between 0 and 1')
    if strength == 0:
        return image.copy()
    source = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
    sample = np.asarray(reference.convert('RGB').resize(image.size, Image.Resampling.BICUBIC), dtype=np.float64) / 255
    if np.array_equal(source, sample):
        return image.convert('RGB').copy()
    mean = source.mean(axis=(0, 1), keepdims=True)
    ddof = 1 if source.shape[0] * source.shape[1] > 1 else 0
    deviation = np.sqrt(source.var(axis=(0, 1), keepdims=True, ddof=ddof) + 1e-5)
    target_mean = sample.mean(axis=(0, 1), keepdims=True)
    target_deviation = np.sqrt(sample.var(axis=(0, 1), keepdims=True, ddof=ddof) + 1e-5)
    normalized = np.divide(source - mean, deviation, out=np.zeros_like(source), where=deviation > 1e-8)
    corrected = normalized * target_deviation + target_mean
    blended = source + strength * (corrected - source)
    return Image.fromarray(np.clip(blended * 255 + 1e-10, 0, 255).astype(np.uint8))


def wavelet(image, reference, strength=1.0):
    """Replace coarse RGB color while retaining target detail.

    Independent NumPy implementation of five dilated binomial smoothing levels;
    no StableSR code or model weights are included.
    """
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite and between 0 and 1')
    if strength == 0:
        return image.copy()
    original = np.asarray(image.convert('RGB'), dtype=np.float64)
    sample = np.asarray(reference.convert('RGB').resize(image.size, Image.Resampling.BICUBIC), dtype=np.float64)

    def coarse(array):
        for radius in (1, 2, 4, 8, 16):
            for axis in (0, 1):
                indices = np.arange(array.shape[axis])
                before = np.take(array, np.maximum(indices - radius, 0), axis=axis)
                after = np.take(array, np.minimum(indices + radius, array.shape[axis] - 1), axis=axis)
                array = (before + 2 * array + after) / 4
        return array

    result = original + strength * (coarse(sample) - coarse(original))
    return Image.fromarray(np.rint(np.clip(result, 0, 255)).astype(np.uint8))
