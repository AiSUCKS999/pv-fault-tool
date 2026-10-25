"""Reference-driven healthy-scene proof of concept, also usable offline.

Signal units are observed reference grayscale codes, not degrees Celsius.
Geometry and production fault rendering are deliberately outside this module.
"""
import math
from pathlib import Path

import cv2
import numpy as np


def correlated_field(shape, sigma, rng):
    field = rng.normal(0, 1, shape).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), max(.5, float(sigma)))
    field -= float(field.mean())
    field /= max(1e-6, float(field.std()))
    return field


def array_groups(quads):
    """Read-only geometric adjacency for shared thermal conditions, no recovery."""
    centers = quads.mean(axis=1)
    spans = quads.max(axis=1) - quads.min(axis=1)
    parents = list(range(len(quads)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(quads)):
        delta = np.abs(centers[i + 1:] - centers[i])
        average = (spans[i + 1:] + spans[i]) * .5
        horizontal = (delta[:, 0] < average[:, 0] * 1.08) & (delta[:, 1] < average[:, 1] * .20)
        vertical = (delta[:, 1] < average[:, 1] * 1.08) & (delta[:, 0] < average[:, 0] * .20)
        for j in np.flatnonzero(horizontal | vertical) + i + 1:
            parents[find(int(j))] = find(i)
    roots = [find(i) for i in range(len(quads))]
    lookup = {value: index for index, value in enumerate(sorted(set(roots)))}
    return np.array([lookup[value] for value in roots]), centers


def coverage(shape, polygon):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon * 256).astype(np.int32), 255,
                      lineType=cv2.LINE_AA, shift=8)
    return mask.astype(np.float32) / 255.


def render_healthy(image, modules, profile, camera_size, seed=20261022, active_support=None, inferred_pv=None,
                   native_faces=False):
    """Evaluate on a bounded camera working plane, using per-module ROI buffers."""
    rng = np.random.default_rng(seed)
    width, height = camera_size
    supersample = 2
    work_size = (width * supersample, height * supersample)
    shape = (work_size[1], work_size[0])
    support = (cv2.resize(active_support, work_size, interpolation=cv2.INTER_NEAREST)
               if active_support is not None else None)
    scale = np.array([work_size[0] / image.width, work_size[1] / image.height], np.float32)
    rgb = np.asarray(image.resize(work_size), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    chroma = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2)
    ground = profile["materials"]["exposed_aisle"]
    smooth_ground = profile["materials"]["track_like_surface"]
    # RGB selects visible substrate classes and shade, never final temperature
    # detail. The reference does not establish botanical/material identities.
    roughness = cv2.GaussianBlur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2 * supersample)),
                                 (0, 0), 2 * supersample)
    track = np.clip((profile["source_material_classifier"]["low_chroma"] - chroma) / 12., 0, 1)
    track *= np.clip((profile["source_material_classifier"]["low_texture"] - roughness) / 3., 0, 1)
    track = cv2.GaussianBlur(track, (0, 0), supersample)
    shade = np.clip((55. - cv2.GaussianBlur(gray, (0, 0), .5 * supersample)) / 35., 0, 1)
    del rgb, gray, chroma, roughness
    base = ground["mean"] * (1 - track) + smooth_ground["mean"] * track
    substrate = correlated_field(shape, ground["correlation_px"] * supersample / math.sqrt(2), rng)
    texture_gain = ground["detrended_std"] * (1 - track) + smooth_ground["detrended_std"] * track
    base += substrate * texture_gain
    base = base * (1 - shade) + profile["materials"]["shade"]["mean"] * shade
    del substrate, texture_gain

    quads = np.array([m["outer_polygon_rgb"] for m in modules], np.float32).reshape(-1,4,2)
    groups, centers = array_groups(quads)
    count = int(groups.max()) + 1 if len(groups) else 0
    # Array offsets are samples of one shared spatial field. A second, shorter
    # correlation scale gives neighboring modules related departures.
    reference_short = profile["module_short_px"]
    shared = correlated_field((height, width), reference_short * 3., rng)
    local = correlated_field((height, width), reference_short, rng)
    camera_centers = centers * np.array([width / image.width, height / image.height])
    group_offsets = np.zeros(count, np.float32)
    for g in range(count):
        center = camera_centers[groups == g].mean(axis=0)
        x, y = np.clip(np.rint(center).astype(int), [0, 0], [width - 1, height - 1])
        group_offsets[g] = shared[y, x]
    if count:group_offsets -= group_offsets.mean()
    group_offsets /= max(1e-6, float(group_offsets.std()))
    group_offsets *= profile["array_mean_std"]
    if inferred_pv is not None:
        material=cv2.resize(np.uint8(inferred_pv>0),work_size,interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # Unknown instance boundaries consume measured-reference PV statistics,
        # never detector rectangles or RGB luminance copied as temperature.
        pv_field=profile["module_mean"]+cv2.resize(shared,work_size,interpolation=cv2.INTER_LINEAR)*profile["array_mean_std"]
        base=base*(1-material)+pv_field*material
        del material,pv_field
    del shared
    reference_faces = []
    if native_faces:
        # Keep the measured spatial face response instead of repeating its
        # fitted linear ramp. These are exemplar signals, not source temperatures.
        reference = cv2.imread(str(Path(profile["reference_path"])), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise ValueError("Healthy thermal reference is unavailable")
        for measurement in profile["face_measurements"]:
            left, top, right, bottom = measurement["box"]
            face = reference[top:bottom, left:right].astype(np.float32)
            reference_faces.append(face - float(face.mean()))
        # Only the background is evaluated on the bounded working plane.
        # Faces, source seams and physical rims use native per-module ROIs.
        base = cv2.resize(base, image.size, interpolation=cv2.INTER_LINEAR)
        work_size = image.size
        shape = (image.height, image.width)
        scale = np.ones(2, np.float32)
        support = active_support
    face_means = []
    physical = np.zeros(shape, np.uint8)
    for index, module in enumerate(modules):
        outer = quads[index] * scale
        active = np.asarray(module["active_polygon_rgb"], np.float32) * scale
        low = np.maximum(0, np.floor(outer.min(axis=0) - 2)).astype(int)
        high = np.minimum(work_size, np.ceil(outer.max(axis=0) + 3)).astype(int)
        x1, y1 = low
        x2, y2 = high
        if x2 <= x1 or y2 <= y1:
            continue
        roi_shape = (y2 - y1, x2 - x1)
        outer_alpha = coverage(roi_shape, outer - low)
        active_alpha = coverage(roi_shape, active - low)
        if support is not None:
            allowed = support[y1:y2, x1:x2] > 0
            outer_alpha = np.clip(outer_alpha - active_alpha * (1 - allowed), 0, 1)
            active_alpha *= allowed
        cx, cy = np.clip(np.rint(camera_centers[index]).astype(int), [0, 0], [width - 1, height - 1])
        module_mean = profile["module_mean"] + group_offsets[groups[index]]
        module_mean += local[cy, cx] * profile["module_mean_std"]
        face_means.append(float(module_mean))
        # Fixed physical frame support, never a distance-based cooling band.
        frame_level = profile["frame_signal"] + group_offsets[groups[index]]
        roi = base[y1:y2, x1:x2]
        roi[:] = roi * (1 - outer_alpha) + frame_level * outer_alpha
        if native_faces:
            field = reference_faces[index % len(reference_faces)] + module_mean
            if np.linalg.norm(active[1]-active[0]) > np.linalg.norm(active[3]-active[0]):
                field = np.rot90(field).copy()
            local_h, local_w = field.shape
        else:
            local_h, local_w = 48, 32
            yy, xx = np.mgrid[0:local_h, 0:local_w].astype(np.float32)
            u, v = xx / (local_w - 1) - .5, yy / (local_h - 1) - .5
            field = np.full((local_h, local_w), module_mean, np.float32)
            field += profile["module_gradient_short"] * u + profile["module_gradient_long"] * v
            if profile["intrinsic_texture_std"] > 0:
                field += correlated_field(field.shape, 4., rng) * profile["intrinsic_texture_std"]
        corners = np.float32([[0, 0], [local_w - 1, 0], [local_w - 1, local_h - 1], [0, local_h - 1]])
        transform = cv2.getPerspectiveTransform(corners, (active - low).astype(np.float32))
        warped = cv2.warpPerspective(field, transform, (x2 - x1, y2 - y1),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if native_faces:
            source = cv2.cvtColor(np.asarray(image.crop((x1,y1,x2,y2))), cv2.COLOR_RGB2GRAY).astype(np.float32)
            # Structural evidence only: restrained cool seams, never RGB
            # luminance interpreted as broad thermal response or hot faults.
            seam = np.maximum(source-cv2.GaussianBlur(source,(0,0),1.),0)
            warped -= np.clip(seam / max(2.,float(np.percentile(seam,96))),0,1) * profile["intrinsic_texture_std"]
        roi[:] = roi * (1 - active_alpha) + warped * active_alpha
        cv2.fillConvexPoly(physical, np.rint(outer * 256).astype(np.int32), 255, shift=8)
    # Reference-derived effective response, followed by one detector sampling
    # and one noise realization. Viewing/cropping does not regenerate signal.
    cv2.GaussianBlur(base, (0, 0), .55 if native_faces else profile["effective_sigma_px"] * supersample, dst=base)
    observed = base if native_faces else cv2.resize(base, camera_size, interpolation=cv2.INTER_AREA)
    for start in range(0, observed.shape[0],128):
        rows=slice(start,start+128)
        observed[rows] += rng.normal(0, profile["detector_noise_std"], observed[rows].shape).astype(np.float32)
    np.clip(observed, 0, 255, out=observed)
    physical = physical if native_faces else cv2.resize(physical, camera_size, interpolation=cv2.INTER_NEAREST)
    material_masks = {
        "exposed_aisle": cv2.resize(np.uint8((track < .5) & (shade < .1)), camera_size, interpolation=cv2.INTER_NEAREST),
        "track_like_surface": cv2.resize(np.uint8((track >= .5) & (shade < .1)), camera_size, interpolation=cv2.INTER_NEAREST),
        "shade": cv2.resize(np.uint8(shade >= .5), camera_size, interpolation=cv2.INTER_NEAREST),
    }
    return observed, {"working_shape": list(shape), "supersample": supersample,
                      "array_groups": count, "module_mean_std_realized": float(np.std(face_means)) if face_means else 0.,
                      "physical_mask": physical, "material_masks": material_masks}
