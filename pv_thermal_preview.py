"""Offline visual-review prototype. Intentionally absent from app.py loaders.

Relative display scalars only: no temperature or empirical accuracy claim.
Changes substrate synthesis, preserving every supplied module/frame pixel.
Run comparisons separately; adopting this aesthetic requires visual review.
"""
import cv2
import numpy as np


def _roof_material_scales(gray, module_mask, scale):
    """Filter rooftop evidence without sampling the dark PV faces/frames.

    Normalized convolution excludes the supplied physical module support at
    every spatial scale. Outside that support the denominator is nonzero,
    even at the boundary. Source roof seams and cast shadows remain evidence.
    """
    valid = np.float32(module_mask == 0)
    roof = gray * valid
    fields = []
    for sigma in (0.8, 4.0, 22.0):
        numerator = cv2.GaussianBlur(roof, (0, 0), sigma * scale)
        denominator = cv2.GaussianBlur(valid, (0, 0), sigma * scale)
        np.maximum(denominator, 1e-6, out=denominator)
        numerator /= denominator
        fields.append(numerator)
    return fields


def experimental_material_background(scalar, rgb, module_mask, material):
    source = np.asarray(scalar, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[:2] != source.shape or module_mask.shape != source.shape:
        raise ValueError("Review inputs must share the original native resolution")
    if material not in {"grass", "roof", "roof_gravel"}:
        raise ValueError("This bounded prototype supports only the reviewed materials")
    scale = max(1.0, max(source.shape) / 1600.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # All spatial evidence comes from the source substrate. No new stochastic
    # field, cell template, periodic checkerboard or large synthetic cloud.
    if material == "roof":
        # Unmasked medium-minus-broad filtering treated each dark module as
        # roof texture, producing a positive shoulder on its lit roof side.
        smooth, medium, broad = _roof_material_scales(gray, module_mask, scale)
    else:
        smooth = cv2.GaussianBlur(gray, (0, 0), 0.8 * scale)
        medium = cv2.GaussianBlur(gray, (0, 0), 4.0 * scale)
        broad = cv2.GaussianBlur(gray, (0, 0), 22.0 * scale)
        distance = cv2.distanceTransform(np.uint8(module_mask == 0), cv2.DIST_L2, 3)
    result = source.copy()
    for start in range(0, source.shape[0], 128):
        rows = slice(start, start + 128)
        texture = smooth[rows] - medium[rows]
        structure = medium[rows] - broad[rows]
        if material == "grass":
            target = 132.0 + medium[rows] * 0.12 + structure * 0.32 + texture * 0.18
        elif material == "roof":
            target = 145.0 + medium[rows] * 0.12 + structure * 0.55 + texture * 0.45
        else:
            target = 148.0 + medium[rows] * 0.11 + structure * 0.40 + texture * 0.14
        # Preserve visible source shade; do not invent illumination patches.
        target += np.minimum(medium[rows] - 62.0, 0) * 0.45
        if material == "roof":
            # The old 2-to-6-pixel collar retained the hotter previous roof
            # scalar against the replacement substrate. Evaluate the same roof
            # material right up to the physical boundary instead; no dilated
            # bright collar, dark outline, or alteration of module/frame pixels.
            outside = module_mask[rows] == 0
            result[rows][outside] = target[outside]
        else:
            # Preserve the existing grass/gravel appearance exactly.
            weight = np.clip((distance[rows] - 2.0 * scale) / (4.0 * scale), 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            result[rows] = source[rows] * (1 - weight) + target * weight
    return result
