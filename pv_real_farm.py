# RF-1 real-farm RGB upload and manual module geometry workflow.
from datetime import datetime, timezone
import csv
import hashlib
import io
import shutil


REAL_FARM_ROOT = ROOT / "artifacts" / "real_farm"
REAL_FARM_UPLOAD_DIR = REAL_FARM_ROOT / "uploads"
REAL_FARM_PROJECT_DIR = REAL_FARM_ROOT / "projects"
REAL_FARM_EXPORT_DIR = REAL_FARM_ROOT / "exports"
REAL_FARM_VERSION = "RF-1"
REAL_FARM_SOURCE_MODE = "REAL_FARM_IMAGE"
REAL_FARM_CONFIDENCE_LEVEL = "LEVEL_1_MODULE_ONLY"


def real_farm_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def real_farm_safe_name(name):
    stem = Path(name or "real_farm_image").stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    return cleaned or "real_farm_image"


def real_farm_file_sha256(data):
    return hashlib.sha256(data).hexdigest()


def real_farm_supported_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def real_farm_load_uploaded_rgb(uploaded_file):
    if not real_farm_supported_image(uploaded_file):
        raise ValueError("Supported formats are PNG, JPG/JPEG, and TIFF.")
    uploaded_file.seek(0)
    data = uploaded_file.read()
    image = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    digest = real_farm_file_sha256(data)[:16]
    REAL_FARM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    target = REAL_FARM_UPLOAD_DIR / f"{real_farm_safe_name(uploaded_file.name)}_{digest}{suffix}"
    if not target.exists():
        target.write_bytes(data)
    return image, target, data, digest


def real_farm_default_state(image_path=None, image=None, filename=None, digest=None):
    width, height = image.size if image is not None else (0, 0)
    return {
        "schema_version": 1,
        "rf_phase": REAL_FARM_VERSION,
        "created_at": real_farm_now(),
        "updated_at": real_farm_now(),
        "base_source_mode": REAL_FARM_SOURCE_MODE,
        "source_rgb_path": str(image_path) if image_path else "",
        "original_rgb_filename": filename or (Path(image_path).name if image_path else ""),
        "source_rgb_sha256_prefix": digest or "",
        "image": {"width": int(width), "height": int(height)},
        "thermal_source": "NONE",
        "thermal_is_real": False,
        "thermal_is_radiometric": False,
        "thermal_mask_ready": False,
        "thermal_geometry_signature": "",
        "observations": [],
        "panel_detection_method": "MANUAL",
        "geometry_confidence_level": 1,
        "modules": [],
        "provenance": {
            "base_source_mode": REAL_FARM_SOURCE_MODE,
            "thermal_source": "NONE",
            "thermal_is_real": False,
            "thermal_is_radiometric": False,
            "panel_detection_method": "MANUAL",
            "geometry_confidence_level": 1,
            "fault_injection_enabled": False,
            "synthetic_thermal_enabled": False,
            "automatic_detection_enabled": False,
        },
    }


def real_farm_next_module_id(modules):
    used = {
        int(str(item.get("module_id", "M000"))[1:])
        for item in modules
        if str(item.get("module_id", "")).startswith("M")
        and str(item.get("module_id", ""))[1:].isdigit()
    }
    next_id = 1
    while next_id in used:
        next_id += 1
    return f"M{next_id:03d}"


def real_farm_order_quad(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    # Rotate so the first point is top-left-ish in image space.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    if real_farm_polygon_area(ordered) < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in ordered]


def real_farm_polygon_area(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def real_farm_point_in_convex_quad(point, quad):
    pt = np.asarray(point, dtype=np.float32)
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    signs = []
    for idx in range(4):
        a = q[idx]
        b = q[(idx + 1) % 4]
        cross = np.cross(b - a, pt - a)
        signs.append(cross)
    return all(v >= -1e-3 for v in signs) or all(v <= 1e-3 for v in signs)


def real_farm_inset_quad(points, inset_ratio=0.035):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    inset = max(0.0, min(float(inset_ratio), 0.30))
    active = center + (pts - center) * (1.0 - inset * 2.0)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in active]


def real_farm_orientation_metadata(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edge_top = float(np.linalg.norm(pts[1] - pts[0]))
    edge_bottom = float(np.linalg.norm(pts[2] - pts[3]))
    edge_right = float(np.linalg.norm(pts[2] - pts[1]))
    edge_left = float(np.linalg.norm(pts[3] - pts[0]))
    width = max(1e-6, (edge_top + edge_bottom) * 0.5)
    height = max(1e-6, (edge_right + edge_left) * 0.5)
    ratio = max(width, height) / max(1e-6, min(width, height))
    if ratio < 1.08:
        label = "unknown"
    else:
        label = "landscape" if width > height else "portrait"
    confidence = float(np.clip((ratio - 1.0) / 1.5, 0.0, 1.0))
    return label, round(confidence, 4), width, height


def real_farm_orientation(points):
    return real_farm_orientation_metadata(points)[0]


def real_farm_canonical_size(points, long_side=320):
    """Derive canonical dimensions from geometry; the label never gates use."""
    _, _, width, height = real_farm_orientation_metadata(points)
    scale = float(long_side) / max(width, height, 1e-6)
    canonical_width = int(np.clip(round(width * scale), 80, long_side))
    canonical_height = int(np.clip(round(height * scale), 80, long_side))
    return {"width": canonical_width, "height": canonical_height}


def real_farm_homography(points):
    dst = np.asarray(points, dtype=np.float32).reshape(4, 2)
    src = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    h_local_to_rgb = cv2.getPerspectiveTransform(src, dst)
    h_rgb_to_local = cv2.getPerspectiveTransform(dst, src)
    det = float(np.linalg.det(h_local_to_rgb))
    return h_local_to_rgb, h_rgb_to_local, det


def real_farm_quad_self_intersects(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersect(a, b, c, d):
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    return intersect(pts[0], pts[1], pts[2], pts[3]) or intersect(pts[1], pts[2], pts[3], pts[0])


def real_farm_mechanical_validation(points, image_size, modules=None, current_id=None):
    """Tiered geometry checks. Only mechanical failures block mask generation."""
    width, height = map(float, image_size)
    blockers, warnings, information = [], [], []
    try:
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    except Exception:
        return {
            "mechanically_valid": False,
            "blockers": [{"level": "error", "code": "invalid_point_count", "message": "A module requires exactly four two-dimensional corners."}],
            "warnings": [],
            "information": [],
            "geometry_confidence": 0.0,
        }
    if not np.isfinite(pts).all():
        blockers.append({"level": "error", "code": "nonfinite_geometry", "message": "Polygon contains NaN or infinite coordinates."})
        return {"mechanically_valid": False, "blockers": blockers, "warnings": warnings, "information": information, "geometry_confidence": 0.0}
    area = abs(real_farm_polygon_area(pts))
    img_area = max(1.0, width * height)
    if real_farm_quad_self_intersects(pts):
        blockers.append({"level": "error", "code": "self_intersecting_polygon", "message": "Polygon corners self-intersect. Reorder corners around the module boundary."})
    distances = [float(np.linalg.norm(pts[a] - pts[b])) for a in range(4) for b in range(a + 1, 4)]
    if min(distances, default=0.0) < 1.0:
        blockers.append({"level": "error", "code": "duplicate_or_degenerate_points", "message": "Two or more polygon corners are duplicated or degenerate."})
    if area < max(4.0, img_area * 0.000001):
        blockers.append({"level": "error", "code": "polygon_area_too_small", "message": "Polygon area is too small to create a stable module transform."})
    outside = sum(1 for x, y in pts if x < 0 or y < 0 or x > width or y > height)
    if outside:
        warnings.append({"level": "warning", "code": "near_boundary_or_partial_crop", "message": f"{outside} corner(s) are outside the image bounds; the visible portion will be used."})
    image_rect = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    try:
        ordered_for_clip = np.asarray(real_farm_order_quad(pts), dtype=np.float32)
        visible_area, _ = cv2.intersectConvexConvex(ordered_for_clip, image_rect)
        visible_fraction = float(visible_area) / max(area, 1e-6)
    except Exception:
        visible_fraction = 0.0
    if visible_fraction < 0.05:
        blockers.append({"level": "error", "code": "polygon_almost_completely_outside", "message": "Less than 5% of the module polygon lies inside the image."})
    try:
        h_local, h_inverse, det = real_farm_homography(pts)
        condition = max(float(np.linalg.cond(h_local)), float(np.linalg.cond(h_inverse)))
        if abs(det) < 1e-8 or not np.isfinite(condition) or condition > 1e12:
            blockers.append({"level": "error", "code": "impossible_homography", "message": "The module-to-image homography is numerically unstable."})
    except Exception:
        blockers.append({"level": "error", "code": "homography_failed", "message": "Could not compute a module-local homography."})
    edge_lengths = [float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx])) for idx in range(4)]
    if min(edge_lengths) <= 1:
        blockers.append({"level": "error", "code": "degenerate_edge", "message": "At least one polygon edge is almost zero length."})
    aspect = max(edge_lengths[0] + edge_lengths[2], edge_lengths[1] + edge_lengths[3]) / max(1e-6, min(edge_lengths[0] + edge_lengths[2], edge_lengths[1] + edge_lengths[3]))
    if aspect > 12:
        warnings.append({"level": "warning", "code": "unusual_aspect_ratio", "message": "The quadrilateral has an unusual aspect ratio, but generation is allowed."})
    opposite_ratio = max(edge_lengths[0] / max(edge_lengths[2], 1e-6), edge_lengths[2] / max(edge_lengths[0], 1e-6), edge_lengths[1] / max(edge_lengths[3], 1e-6), edge_lengths[3] / max(edge_lengths[1], 1e-6))
    if opposite_ratio > 3.5:
        warnings.append({"level": "warning", "code": "heavy_perspective", "message": "The module has strong perspective distortion; canonical rectification will still be used."})
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(pts)
    if orientation == "unknown":
        information.append({"level": "info", "code": "orientation_ambiguous", "message": "Orientation is ambiguous and is stored as unknown; generation is still allowed."})
    elif orientation_confidence < 0.3:
        information.append({"level": "info", "code": "orientation_low_confidence", "message": f"Orientation inferred as {orientation} with low confidence."})
    if modules:
        rounded = np.round(pts, 1)
        for item in modules:
            if item.get("module_id") == current_id:
                continue
            other = np.round(np.asarray(item.get("outer_polygon_rgb", []), dtype=np.float32), 1)
            if other.shape == (4, 2) and np.allclose(rounded, other, atol=1.5):
                warnings.append({"level": "warning", "code": "duplicate_polygon", "message": f"Polygon duplicates {item.get('module_id')}."})
                break
    confidence = float(np.clip(1.0 - 0.18 * len(warnings) - 0.08 * len(information), 0.0, 1.0)) if not blockers else 0.0
    return {
        "mechanically_valid": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "information": information,
        "geometry_confidence": round(confidence, 4),
    }


def real_farm_quality(points, image_size, modules=None, current_id=None):
    validation = real_farm_mechanical_validation(points, image_size, modules, current_id)
    return validation["blockers"] + validation["warnings"] + validation["information"]


def real_farm_module_record(module_id, points, image_size, array_id="", inset_ratio=0.035, provenance_note="manual"):
    raw_validation = real_farm_mechanical_validation(points, image_size)
    ordered = real_farm_order_quad(points)
    active = real_farm_inset_quad(ordered, inset_ratio)
    h_local, h_inv, det = real_farm_homography(ordered)
    ordered_validation = real_farm_mechanical_validation(ordered, image_size)
    validation = raw_validation if raw_validation["blockers"] else ordered_validation
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(ordered)
    quality = validation["blockers"] + validation["warnings"] + validation["information"]
    starter_roi = provenance_note in {"manual_add", "manual_edit", "manual_duplicate", "manual_repeat_row", "manual_repeat_grid"}
    return {
        "module_id": module_id,
        "array_id": array_id.strip() or None,
        "source_mode": REAL_FARM_SOURCE_MODE,
        "polygon_rgb": ordered,
        "coarse_roi_polygon_rgb": ordered,
        "outer_polygon_rgb": ordered,
        "active_polygon_rgb": active,
        "manual_final_polygon_rgb": None if starter_roi else active,
        "refined_pv_polygon_rgb": None,
        "refined_mask_contours_rgb": [],
        "mask_source": "SEARCH_ROI" if starter_roi else "MANUAL_FINAL_POLYGON",
        "refinement_status": "pending" if starter_roi else "accepted",
        "active_inset_ratio": round(float(inset_ratio), 4),
        "canonical_size": real_farm_canonical_size(ordered),
        "orientation": orientation,
        "orientation_label": orientation,
        "orientation_confidence": orientation_confidence,
        "geometry_confidence": validation["geometry_confidence"],
        "warnings": validation["warnings"] + validation["information"],
        "mechanically_valid": validation["mechanically_valid"],
        "visually_supported": None,
        "homography_module_to_rgb": np.round(h_local, 8).tolist(),
        "homography_rgb_to_module": np.round(h_inv, 8).tolist(),
        "cells": None,
        "electrical_sections": [],
        "confidence": {
            "level": 1,
            "label": REAL_FARM_CONFIDENCE_LEVEL,
            "module_polygon": "MANUAL",
            "active_polygon": "DERIVED_INSET",
            "cells": "NOT_AVAILABLE",
            "electrical_sections": "NOT_AVAILABLE",
            "homography_determinant": det,
            "quality_warnings": quality,
        },
        "provenance": {
            "created_by": "RF-1 manual module geometry workflow",
            "geometry_method": "MANUAL_QUADRILATERAL",
            "provenance_note": provenance_note,
            "updated_at": real_farm_now(),
        },
    }


def real_farm_update_module(module, image_size):
    return real_farm_module_record(
        module.get("module_id"),
        module.get("outer_polygon_rgb", []),
        image_size,
        module.get("array_id") or "",
        float(module.get("active_inset_ratio", 0.035)),
        module.get("provenance", {}).get("provenance_note", "manual_edit"),
    )


def real_farm_project_json(state):
    clone = json.loads(json.dumps(state))
    clone["updated_at"] = real_farm_now()
    return json.dumps(clone, indent=2, sort_keys=True)


def real_farm_save_project(state, name="rf1_project"):
    REAL_FARM_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    path = REAL_FARM_PROJECT_DIR / f"{real_farm_safe_name(name)}.json"
    path.write_text(real_farm_project_json(state), encoding="utf-8")
    return path


def real_farm_load_project(path_or_file):
    if hasattr(path_or_file, "read"):
        data = path_or_file.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    return json.loads(Path(path_or_file).read_text(encoding="utf-8"))


def real_farm_modules_csv(state):
    output = BytesIO()
    text = io.StringIO()
    fieldnames = ["module_id", "array_id", "mask_source", "refinement_status", "orientation_label", "orientation_confidence", "geometry_confidence", "mechanically_valid", "visually_supported", "canonical_size", "warnings", "coarse_roi_polygon_rgb", "refined_pv_polygon_rgb", "refined_mask_contours_rgb", "outer_polygon_rgb", "active_polygon_rgb"]
    writer = csv.DictWriter(text, fieldnames=fieldnames)
    writer.writeheader()
    for module in state.get("modules", []):
        writer.writerow({
            "module_id": module.get("module_id"),
            "array_id": module.get("array_id") or "",
            "mask_source": module.get("mask_source"),
            "refinement_status": module.get("refinement_status"),
            "orientation_label": module.get("orientation_label", module.get("orientation")),
            "orientation_confidence": module.get("orientation_confidence"),
            "geometry_confidence": module.get("geometry_confidence"),
            "mechanically_valid": module.get("mechanically_valid"),
            "visually_supported": module.get("visually_supported"),
            "canonical_size": json.dumps(module.get("canonical_size")),
            "warnings": json.dumps(module.get("warnings", [])),
            "coarse_roi_polygon_rgb": json.dumps(module.get("coarse_roi_polygon_rgb")),
            "refined_pv_polygon_rgb": json.dumps(module.get("refined_pv_polygon_rgb")),
            "refined_mask_contours_rgb": json.dumps(module.get("refined_mask_contours_rgb", [])),
            "outer_polygon_rgb": json.dumps(module.get("outer_polygon_rgb")),
            "active_polygon_rgb": json.dumps(module.get("active_polygon_rgb")),
        })
    output.write(text.getvalue().encode("utf-8"))
    return output.getvalue()


def real_farm_annotated_preview(image, modules, selected_id=None, max_width=1800):
    scale = min(1.0, float(max_width) / max(1, image.width))
    preview = image.copy()
    if scale < 1:
        preview = preview.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    overlay = preview.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    font = grid_font(14)
    for module in modules:
        color = (56, 189, 248, 235) if module.get("module_id") == selected_id else (34, 197, 94, 220)
        active_color = (250, 204, 21, 190) if module.get("module_id") == selected_id else (134, 239, 172, 160)
        outer = [(x * scale, y * scale) for x, y in module.get("outer_polygon_rgb", [])]
        active = [(x * scale, y * scale) for x, y in module.get("active_polygon_rgb", [])]
        if len(outer) == 4:
            draw.line(outer + [outer[0]], fill=color, width=3 if module.get("module_id") == selected_id else 2)
            for idx, point in enumerate(outer, start=1):
                x, y = point
                r = 4 if module.get("module_id") == selected_id else 3
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
                if module.get("module_id") == selected_id:
                    draw.text((x + 5, y + 2), str(idx), fill=(255, 255, 255, 255), font=font)
            cx = sum(x for x, _ in outer) / 4
            cy = sum(y for _, y in outer) / 4
            label = module.get("module_id", "M???")
            if module.get("array_id"):
                label = f"{module.get('array_id')}-{label}"
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.rounded_rectangle((cx - 4, cy - 3, cx + (bbox[2]-bbox[0]) + 8, cy + (bbox[3]-bbox[1]) + 5), radius=3, fill=(8, 13, 24, 190))
            draw.text((cx, cy), label, fill=(255, 255, 255, 255), font=font)
        if len(active) == 4:
            draw.line(active + [active[0]], fill=active_color, width=1)
    return overlay.convert("RGB")


def real_farm_rectified_crop(image, module, output_size=None):
    if output_size is None:
        size = module.get("canonical_size") or real_farm_canonical_size(module.get("outer_polygon_rgb"))
        scale = 520.0 / max(float(size["width"]), float(size["height"]), 1.0)
        output_size = (max(80, int(round(size["width"] * scale))), max(80, int(round(size["height"] * scale))))
    w, h = map(int, output_size)
    source = np.asarray(image.convert("RGB"))
    dst = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    src = np.asarray(module.get("outer_polygon_rgb"), dtype=np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(source, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(crop)


def real_farm_rectified_contact_sheet(image, modules, max_modules=24):
    selected = modules[:max_modules]
    if not selected:
        return Image.new("RGB", (720, 180), (245, 247, 250))
    thumb_w, thumb_h = 180, 140
    cols = 4
    rows = int(math.ceil(len(selected) / cols))
    pad = 16
    label_h = 28
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    font = grid_font(12)
    for idx, module in enumerate(selected):
        crop = real_farm_rectified_crop(image, module)
        crop.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_w, thumb_h), (226, 232, 240))
        cell.paste(crop, ((thumb_w - crop.width) // 2, (thumb_h - crop.height) // 2))
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + label_h + pad)
        sheet.paste(cell, (x, y))
        draw.text((x, y + thumb_h + 5), f"{module.get('module_id')} | {module.get('orientation')}", fill=(15, 23, 42), font=font)
    return sheet


def real_farm_viewport_image(image, zoom=1.0, pan_x=0.5, pan_y=0.5, max_size=(1200, 760)):
    zoom = max(1.0, min(float(zoom), 8.0))
    crop_w = max(1, int(image.width / zoom))
    crop_h = max(1, int(image.height / zoom))
    cx = int(image.width * max(0.0, min(float(pan_x), 1.0)))
    cy = int(image.height * max(0.0, min(float(pan_y), 1.0)))
    x1 = max(0, min(image.width - crop_w, cx - crop_w // 2))
    y1 = max(0, min(image.height - crop_h, cy - crop_h // 2))
    crop = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    crop.thumbnail(max_size, Image.Resampling.LANCZOS)
    return crop


def real_farm_push_history():
    state = st.session_state.get("real_farm_state")
    if state:
        st.session_state.setdefault("real_farm_undo_stack", []).append(json.loads(json.dumps(state)))
        st.session_state["real_farm_undo_stack"] = st.session_state["real_farm_undo_stack"][-30:]
        st.session_state["real_farm_redo_stack"] = []


def real_farm_restore(state):
    st.session_state["real_farm_state"] = json.loads(json.dumps(state))


def real_farm_geometry_signature(modules):
    geometry = [
        {
            "module_id": item.get("module_id"),
            "active_polygon_rgb": item.get("active_polygon_rgb", []),
            "mask_source": item.get("mask_source", "MANUAL_FINAL_POLYGON"),
            "refined_mask_contours_rgb": item.get("refined_mask_contours_rgb", []),
        }
        for item in modules
    ]
    payload = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def real_farm_invalidate_thermal(state):
    state["thermal_source"] = "NONE"
    state["thermal_is_real"] = False
    state["thermal_is_radiometric"] = False
    state["thermal_mask_ready"] = False
    state["thermal_geometry_signature"] = ""
    state["observations"] = []
    provenance = state.setdefault("provenance", {})
    provenance.update({
        "thermal_source": "NONE",
        "thermal_is_real": False,
        "thermal_is_radiometric": False,
        "fault_injection_enabled": False,
        "synthetic_thermal_enabled": False,
    })
    for key in ("real_farm_selected_observation", "real_farm_observation_picker", "real_farm_close_view"):
        st.session_state.pop(key, None)
    st.session_state.pop("real_farm_thermal_master_layers", None)
    st.session_state.pop("real_farm_thermal_master_signature", None)


def real_farm_module_mask(image_size, modules, active=True):
    width, height = map(int, image_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_key = "active_polygon_rgb" if active else "outer_polygon_rgb"
    for module in modules:
        if active and module.get("mask_source") == "AUTO_REFINED_RGB" and module.get("refinement_status") == "accepted":
            for contour in module.get("refined_mask_contours_rgb", []):
                points = np.rint(np.asarray(contour, dtype=np.float32)).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
            continue
        points = np.rint(np.asarray(module.get(polygon_key, []), dtype=np.float32)).astype(np.int32)
        if points.shape == (4, 2):
            cv2.fillConvexPoly(mask, points, 255, lineType=cv2.LINE_AA)
    return mask


def real_farm_roi_mask(image_size, module):
    width, height = map(int, image_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    points = module.get("coarse_roi_polygon_rgb") or module.get("outer_polygon_rgb", [])
    polygon = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    if polygon.shape == (4, 2):
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
    return mask


def real_farm_refined_overlay(image, module, result=None):
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    roi = module.get("coarse_roi_polygon_rgb") or module.get("outer_polygon_rgb", [])
    if len(roi) == 4:
        coarse = [tuple(map(float, point)) for point in roi]
        draw.line(coarse + [coarse[0]], fill=(56, 189, 248, 235), width=max(2, image.width // 900))
    contours = (result or {}).get("refined_mask_contours_rgb") or module.get("refined_mask_contours_rgb", [])
    for contour in contours:
        points = [tuple(map(float, point)) for point in contour]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(34, 197, 94, 255), width=max(3, image.width // 650))
    return overlay.convert("RGB")


def real_farm_refine_pv_support(image, module):
    """Refine a coarse search ROI into orientation-independent PV support."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    roi_mask = real_farm_roi_mask(image.size, module)
    roi_points = np.asarray(module.get("coarse_roi_polygon_rgb") or module.get("outer_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    x_min, y_min = np.floor(roi_points.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(roi_points.max(axis=0)).astype(int)
    span = max(x_max - x_min, y_max - y_min, 40)
    padding = int(max(80, span * 1.5))
    x1, y1 = max(0, x_min - padding), max(0, y_min - padding)
    x2, y2 = min(width, x_max + padding + 1), min(height, y_max + padding + 1)
    crop = rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    red, green, blue = [crop[:, :, index].astype(np.int16) for index in range(3)]
    hue, saturation, value = [hsv[:, :, index] for index in range(3)]
    blue_surface = (hue >= 78) & (hue <= 132) & (saturation >= 24) & (value >= 24)
    cool_dark_surface = (blue >= green - 4) & (blue >= red + 6) & (value <= 175) & (saturation >= 18)
    vegetation = green > ((red + blue) // 2 + 12)
    candidate_local = ((blue_surface | cool_dark_surface) & ~vegetation).astype(np.uint8) * 255
    raw_mask = np.zeros((height, width), dtype=np.uint8)
    raw_mask[y1:y2, x1:x2] = candidate_local
    scale = max(1, int(round(max(crop.shape[:2]) / 900)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, 7 * scale) | 1, max(5, 7 * scale) | 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, 3 * scale) | 1, max(3, 3 * scale) | 1))
    cleaned_local = cv2.morphologyEx(candidate_local, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned_local = cv2.morphologyEx(cleaned_local, cv2.MORPH_OPEN, open_kernel, iterations=1)
    roi_local = roi_mask[y1:y2, x1:x2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_local, connectivity=8)
    selections = []
    roi_area = max(1, int(np.count_nonzero(roi_local)))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        overlap = int(np.count_nonzero(component & (roi_local > 0)))
        if area < max(80, roi_area * 0.015) or overlap < max(12, roi_area * 0.002):
            continue
        component_values = value[component]
        component_hue = hue[component]
        panel_fraction = float(np.mean((component_hue >= 78) & (component_hue <= 132)))
        score = overlap / max(1.0, roi_area) + min(1.0, area / max(1.0, roi_area)) * 0.25 + panel_fraction * 0.5
        selections.append((score, label, area, overlap))
    if not selections:
        return {
            "success": False, "message": "No connected PV-material candidate intersects the coarse ROI.",
            "roi_bbox": [x1, y1, x2, y2], "raw_mask": raw_mask,
            "cleaned_mask": np.zeros_like(raw_mask), "final_mask": np.zeros_like(raw_mask),
            "refined_mask_contours_rgb": [], "refined_pv_polygon_rgb": None,
        }
    selections.sort(reverse=True)
    best_score = selections[0][0]
    selected_labels = [label for score, label, _, _ in selections if score >= max(0.18, best_score * 0.62)]
    selected_local = np.isin(labels, selected_labels).astype(np.uint8) * 255
    contours, _ = cv2.findContours(selected_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_local = np.zeros_like(selected_local)
    stored_contours = []
    largest_contour = None
    largest_area = 0.0
    for contour in contours:
        if cv2.contourArea(contour) < max(80, roi_area * 0.015):
            continue
        cv2.fillPoly(final_local, [contour], 255, lineType=cv2.LINE_AA)
        global_contour = contour.reshape(-1, 2).astype(np.float32) + np.asarray([x1, y1], dtype=np.float32)
        epsilon = max(1.5, cv2.arcLength(global_contour.reshape(-1, 1, 2), True) * 0.003)
        simplified = cv2.approxPolyDP(global_contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        stored_contours.append([[round(float(x), 2), round(float(y), 2)] for x, y in simplified])
        contour_area = float(cv2.contourArea(contour))
        if contour_area > largest_area:
            largest_area, largest_contour = contour_area, global_contour
    if largest_contour is None:
        return {
            "success": False, "message": "PV candidates were too small after cleanup.",
            "roi_bbox": [x1, y1, x2, y2], "raw_mask": raw_mask,
            "cleaned_mask": np.zeros_like(raw_mask), "final_mask": np.zeros_like(raw_mask),
            "refined_mask_contours_rgb": [], "refined_pv_polygon_rgb": None,
        }
    rect = cv2.minAreaRect(largest_contour.reshape(-1, 1, 2))
    quad = real_farm_order_quad(cv2.boxPoints(rect))
    cleaned_mask = np.zeros((height, width), dtype=np.uint8)
    cleaned_mask[y1:y2, x1:x2] = selected_local
    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_mask[y1:y2, x1:x2] = final_local
    roi_pixels = final_mask[roi_mask > 0]
    refined_fraction = float(np.mean(roi_pixels > 0)) if roi_pixels.size else 0.0
    return {
        "success": True,
        "message": f"Refined PV support covers {refined_fraction:.1%} of the coarse ROI; non-PV pixels remain background.",
        "roi_bbox": [x1, y1, x2, y2],
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "final_mask": final_mask,
        "refined_mask_contours_rgb": stored_contours,
        "refined_pv_polygon_rgb": quad,
        "refined_roi_fraction": round(refined_fraction, 6),
        "candidate_component_count": len(stored_contours),
    }


def real_farm_accept_refinement(module, result):
    if not result.get("success") or not result.get("refined_pv_polygon_rgb"):
        raise ValueError("A successful refinement result is required.")
    quad = real_farm_order_quad(result["refined_pv_polygon_rgb"])
    h_local, h_inverse, determinant = real_farm_homography(quad)
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(quad)
    module["coarse_roi_polygon_rgb"] = module.get("coarse_roi_polygon_rgb") or module.get("outer_polygon_rgb")
    module["refined_pv_polygon_rgb"] = quad
    module["refined_mask_contours_rgb"] = result["refined_mask_contours_rgb"]
    module["active_polygon_rgb"] = quad
    module["mask_source"] = "AUTO_REFINED_RGB"
    module["refinement_status"] = "accepted"
    module["refined_roi_fraction"] = result.get("refined_roi_fraction")
    module["canonical_size"] = real_farm_canonical_size(quad)
    module["orientation"] = orientation
    module["orientation_label"] = orientation
    module["orientation_confidence"] = orientation_confidence
    module["homography_module_to_rgb"] = np.round(h_local, 8).tolist()
    module["homography_rgb_to_module"] = np.round(h_inverse, 8).tolist()
    module.setdefault("confidence", {})["homography_determinant"] = determinant
    module.setdefault("provenance", {})["mask_refinement_method"] = "LOCAL_RGB_MULTI_CUE_CONNECTED_COMPONENT"
    module["provenance"]["refinement_accepted_at"] = real_farm_now()
    return module


def real_farm_use_manual_final_polygon(module):
    final_polygon = real_farm_inset_quad(module.get("outer_polygon_rgb", []), float(module.get("active_inset_ratio", 0.035)))
    h_local, h_inverse, determinant = real_farm_homography(final_polygon)
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(final_polygon)
    module["manual_final_polygon_rgb"] = final_polygon
    module["active_polygon_rgb"] = final_polygon
    module["refined_pv_polygon_rgb"] = None
    module["refined_mask_contours_rgb"] = []
    module["mask_source"] = "MANUAL_FINAL_POLYGON"
    module["refinement_status"] = "accepted"
    module["canonical_size"] = real_farm_canonical_size(final_polygon)
    module["orientation"] = orientation
    module["orientation_label"] = orientation
    module["orientation_confidence"] = orientation_confidence
    module["homography_module_to_rgb"] = np.round(h_local, 8).tolist()
    module["homography_rgb_to_module"] = np.round(h_inverse, 8).tolist()
    module.setdefault("confidence", {})["homography_determinant"] = determinant
    module.setdefault("provenance", {})["mask_refinement_method"] = "EXPLICIT_MANUAL_POLYGON_OVERRIDE"
    return module


def real_farm_local_to_rgb(module, u, v):
    matrix = np.asarray(module.get("homography_module_to_rgb"), dtype=np.float32).reshape(3, 3)
    source = np.asarray([[[float(u), float(v)]]], dtype=np.float32)
    point = cv2.perspectiveTransform(source, matrix)[0, 0]
    return float(point[0]), float(point[1])


def real_farm_module_radius(module):
    points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    edges = [float(np.linalg.norm(points[(idx + 1) % 4] - points[idx])) for idx in range(4)]
    return max(4, int(min(sum(edges[::2]) * 0.5, sum(edges[1::2]) * 0.5) * 0.16))


def real_farm_generate_observations(modules, count, seed):
    if not modules:
        return []
    rng = np.random.default_rng(int(seed))
    order = list(range(len(modules)))
    rng.shuffle(order)
    observations = []
    for idx in range(int(count)):
        if idx and idx % len(modules) == 0:
            rng.shuffle(order)
        module = modules[order[idx % len(modules)]]
        u = float(rng.uniform(0.18, 0.82))
        v = float(rng.uniform(0.18, 0.82))
        x, y = real_farm_local_to_rgb(module, u, v)
        observations.append({
            "observation_id": f"RFH{idx + 1:03d}",
            "observation_type": "SingleHotSpot",
            "module_id": module.get("module_id"),
            "local_uv": [round(u, 6), round(v, 6)],
            "center_rgb_px": [round(x, 2), round(y, 2)],
            "severity": int(rng.integers(5, 10)),
            "diagnostic_fault_type": None,
            "created_at": real_farm_now(),
        })
    return observations


def real_farm_legacy_temperature_field(image, modules, observations=None):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
    field = 54.0 + smooth.astype(np.float32) * 0.27
    for module in modules:
        module_mask = real_farm_module_mask(image.size, [module], active=True)
        module_seed = int(hashlib.sha256(str(module.get("module_id")).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(module_seed)
        variation = cv2.GaussianBlur(rng.normal(0, 1.8, field.shape).astype(np.float32), (0, 0), 1.2)
        module_temp = 166.0 + (smooth.astype(np.float32) - 128.0) * 0.055 + variation
        field[module_mask > 0] = module_temp[module_mask > 0]
    module_lookup = {item.get("module_id"): item for item in modules}
    for observation in observations or []:
        module = module_lookup.get(observation.get("module_id"))
        if not module:
            continue
        x, y = observation.get("center_rgb_px", (0, 0))
        radius = real_farm_module_radius(module)
        x1 = max(0, int(x) - radius * 3)
        y1 = max(0, int(y) - radius * 3)
        x2 = min(image.width, int(x) + radius * 3 + 1)
        y2 = min(image.height, int(y) + radius * 3 + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        sigma = max(2.0, radius * 0.72)
        bump = np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2.0 * sigma * sigma))
        local_mask = real_farm_module_mask(image.size, [module], active=True)[y1:y2, x1:x2] > 0
        field[y1:y2, x1:x2] += bump * (62.0 + float(observation.get("severity", 7)) * 2.1) * local_mask
    return np.clip(field, 0, 255).astype(np.uint8)


def real_farm_legacy_thermal_image(image, modules, observations=None):
    field = real_farm_legacy_temperature_field(image, modules, observations)
    return temperature_to_rgb(Image.fromarray(field, "L"))


REAL_FARM_SYNTHETIC_THERMAL = "SYNTHETIC_RELATIVE_PROXY"
REAL_FARM_REAL_THERMAL = "REAL_THERMAL"


def real_farm_module_visual_score(image, module):
    """Return a conservative RGB panel-likeness score for geometry QA."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    x1 = max(0, int(math.floor(float(points[:, 0].min()))))
    y1 = max(0, int(math.floor(float(points[:, 1].min()))))
    x2 = min(image.width, int(math.ceil(float(points[:, 0].max()))) + 1)
    y2 = min(image.height, int(math.ceil(float(points[:, 1].max()))) + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    local_points = np.rint(points - np.asarray([x1, y1], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_points, 255, lineType=cv2.LINE_AA)
    crop = rgb[y1:y2, x1:x2]
    pixels = crop[mask > 0].astype(np.float32)
    if pixels.shape[0] < 16:
        return 0.0
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    green_ground = float(np.mean(green > ((red + blue) * 0.5 + 12.0)))
    blue_panel = float(np.mean(blue > green + 4.0))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    dark_fraction = float(np.mean(gray[mask > 0] < 145))
    score = 0.55 * blue_panel + 0.25 * dark_fraction + 0.20 * (1.0 - green_ground) - 0.35 * green_ground
    return float(np.clip(score, 0.0, 1.0))


def real_farm_thermal_modules(image, modules):
    """Accept mechanically valid manual geometry; visual evidence is advisory."""
    accepted, rejected = [], []
    for module in modules:
        score = real_farm_module_visual_score(image, module)
        item = dict(module)
        item["thermal_geometry_score"] = round(score, 4)
        item["visually_supported"] = bool(score >= 0.43)
        validation = real_farm_mechanical_validation(item.get("outer_polygon_rgb", []), image.size)
        stored_validity = item.get("mechanically_valid")
        mechanically_valid = validation["mechanically_valid"] if stored_validity is None else bool(stored_validity and validation["mechanically_valid"])
        item["mechanically_valid"] = mechanically_valid
        item["geometry_confidence"] = validation["geometry_confidence"]
        notices = list(item.get("warnings") or [])
        notices.extend(validation["warnings"] + validation["information"])
        if not item["visually_supported"]:
            notices.append({
                "level": "warning",
                "code": "weak_pv_texture_confidence",
                "message": f"{item.get('module_id', 'Module')} has weak PV-texture confidence, but mask generation is allowed.",
            })
        deduplicated = {notice.get("code", str(index)): notice for index, notice in enumerate(notices)}
        item["warnings"] = list(deduplicated.values())
        item["mechanical_blockers"] = validation["blockers"]
        refinement_ready = item.get("refinement_status", "accepted") == "accepted" and item.get("mask_source", "MANUAL_FINAL_POLYGON") != "SEARCH_ROI"
        item["refinement_ready"] = refinement_ready
        if mechanically_valid and refinement_ready:
            accepted.append(item)
        else:
            if mechanically_valid and not refinement_ready:
                item["mechanical_blockers"] = []
                item["warnings"].append({
                    "level": "info", "code": "refinement_required",
                    "message": f"{item.get('module_id', 'Module')} is a coarse search ROI. Accept an auto-refined mask or choose the manual polygon override.",
                })
            rejected.append(item)
    return accepted, rejected


def real_farm_refinement_workflow(state, image, modules, selected_id):
    st.markdown('<div class="pv-card"><div class="pv-section-title">Refined PV Area</div>', unsafe_allow_html=True)
    st.caption("The blue polygon is only a coarse search ROI. Auto-Refine finds visible PV material; review the green result before accepting it as thermal support.")
    selected = next((item for item in modules if item.get("module_id") == selected_id), None)
    if selected is None:
        st.info("Add and select a coarse PV region first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    status = selected.get("refinement_status", "pending")
    source = selected.get("mask_source", "SEARCH_ROI")
    st.write(f"**{selected.get('module_id')}** · `{source}` · `{status}`")
    preview_key = f"real_farm_refinement_preview_{selected.get('module_id')}"
    action_cols = st.columns(3)
    if action_cols[0].button("Auto-Refine", type="primary", use_container_width=True, key=f"rf_refine_{selected.get('module_id')}"):
        st.session_state[preview_key] = real_farm_refine_pv_support(image, selected)
    preview = st.session_state.get(preview_key)
    if preview:
        if preview.get("success"):
            st.success(preview.get("message"))
            preview_cols = st.columns(2)
            preview_cols[0].image(real_farm_refined_overlay(image, selected, preview), caption="Blue: coarse ROI · Green: refined PV support", use_container_width=True)
            preview_cols[1].image(Image.fromarray(preview["final_mask"], "L"), caption="Proposed final binary PV mask", use_container_width=True, clamp=True)
            if action_cols[1].button("Accept Mask", type="primary", use_container_width=True, key=f"rf_accept_{selected.get('module_id')}"):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                real_farm_accept_refinement(selected, preview)
                state["updated_at"] = real_farm_now()
                st.session_state.pop(preview_key, None)
                st.rerun()
        else:
            st.warning(preview.get("message", "Auto-refinement did not find a usable PV region."))
            debug_cols = st.columns(2)
            debug_cols[0].image(Image.fromarray(preview["raw_mask"], "L"), caption="Raw PV candidates", use_container_width=True, clamp=True)
            debug_cols[1].image(Image.fromarray(preview["cleaned_mask"], "L"), caption="Cleaned candidates", use_container_width=True, clamp=True)
    if action_cols[2].button("Use Manual Polygon", use_container_width=True, key=f"rf_manual_{selected.get('module_id')}"):
        real_farm_push_history()
        real_farm_invalidate_thermal(state)
        real_farm_use_manual_final_polygon(selected)
        state["updated_at"] = real_farm_now()
        st.session_state.pop(preview_key, None)
        st.rerun()
    if source == "AUTO_REFINED_RGB" and status == "accepted":
        accepted_cols = st.columns(2)
        accepted_cols[0].image(real_farm_refined_overlay(image, selected), caption="Accepted refined outline", use_container_width=True)
        accepted_cols[1].image(real_farm_mask_preview(image.size, [selected]), caption="Accepted final binary PV mask", use_container_width=True, clamp=True)
    elif source == "MANUAL_FINAL_POLYGON" and status == "accepted":
        st.warning("Manual override is active. The edited polygon is final thermal support; use Auto-Refine to derive support from RGB material instead.")
    else:
        st.info("Thermal generation stays locked until this coarse ROI has an accepted mask.")
    st.markdown("</div>", unsafe_allow_html=True)


def real_farm_background_scalar(image):
    """Build a full-scene relative thermal background while retaining RGB structure."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = luminance.shape
    low_sigma = max(10.0, max(width, height) / 65.0)
    mid_sigma = max(2.0, max(width, height) / 420.0)
    low = cv2.GaussianBlur(luminance, (0, 0), low_sigma)
    mid = cv2.GaussianBlur(luminance, (0, 0), mid_sigma)
    detail = luminance - mid
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    vegetation = np.clip(green - (red + blue) * 0.5, 0, 80)
    base = 101.0 + (low - float(np.median(low))) * 0.27 + (mid - low) * 0.34 + detail * 0.22
    base -= vegetation * 0.13
    seed = seed_for(width, height, float(np.mean(luminance)), "real-farm-background")
    texture = smooth_unit_field((height, width), seed, max(5.0, max(width, height) / 300.0), octaves=3)
    base += (texture - 0.5) * 7.0
    return np.clip(base, 34, 190).astype(np.float32)


def real_farm_module_scalar_layer(image, modules, background_scalar=None):
    """Rasterize per-module relative thermal response without using any union bbox."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    scene_base = np.asarray(background_scalar, dtype=np.float32) if background_scalar is not None else real_farm_background_scalar(image)
    scalar_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    union_mask = real_farm_module_mask(image.size, modules, active=True)
    for module_index, module in enumerate(modules):
        points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
        canonical_size = module.get("canonical_size") or real_farm_canonical_size(points)
        local_width = int(np.clip(canonical_size["width"], 24, 720))
        local_height = int(np.clip(canonical_size["height"], 24, 960))
        destination = np.asarray([[0, 0], [local_width - 1, 0], [local_width - 1, local_height - 1], [0, local_height - 1]], dtype=np.float32)
        to_local = cv2.getPerspectiveTransform(points, destination)
        x1 = max(0, int(math.floor(float(points[:, 0].min()))) - 1)
        y1 = max(0, int(math.floor(float(points[:, 1].min()))) - 1)
        x2 = min(width, int(math.ceil(float(points[:, 0].max()))) + 2)
        y2 = min(height, int(math.ceil(float(points[:, 1].max()))) + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi_points = points - np.asarray([x1, y1], dtype=np.float32)
        to_roi = cv2.getPerspectiveTransform(destination, roi_points)
        local_gray = cv2.warpPerspective(gray, to_local, (local_width, local_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        local_scene_base = cv2.warpPerspective(scene_base, to_local, (local_width, local_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        local_low = cv2.GaussianBlur(local_gray, (0, 0), max(1.2, min(local_width, local_height) * 0.055))
        local_detail = local_gray - cv2.GaussianBlur(local_gray, (0, 0), 1.1)
        yy, xx = np.mgrid[0:local_height, 0:local_width].astype(np.float32)
        u = xx / max(1, local_width - 1)
        v = yy / max(1, local_height - 1)
        module_seed = seed_for(module.get("module_id", module_index), "real-farm-pv-layer")
        rng = np.random.default_rng(module_seed)
        module_bias = float(rng.normal(6.5, 1.4))
        gradient = (u - 0.5) * float(rng.uniform(-3.2, 3.2)) + (v - 0.5) * float(rng.uniform(-2.8, 2.8))
        mottle = smooth_unit_field((local_height, local_width), module_seed, max(2.0, min(local_width, local_height) * 0.10), octaves=3)
        row_bias = np.sin(v * math.pi * max(2, min(14, int(local_height / 24))) + float(rng.uniform(-1, 1))) * 0.7
        pv_local = local_scene_base + module_bias + gradient + (local_low - float(np.median(local_low))) * 0.025 + local_detail * 0.12
        pv_local += (mottle - 0.5) * 3.2 + row_bias
        local_support = np.ones((local_height, local_width), dtype=np.float32)
        roi_size = (x2 - x1, y2 - y1)
        warped_scalar = cv2.warpPerspective(pv_local.astype(np.float32), to_roi, roi_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_support = cv2.warpPerspective(local_support, to_roi, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        module_mask = np.zeros((roi_size[1], roi_size[0]), dtype=np.uint8)
        cv2.fillConvexPoly(module_mask, np.rint(roi_points).astype(np.int32), 255, lineType=cv2.LINE_AA)
        module_mask = module_mask.astype(np.float32) / 255.0
        support = np.clip(warped_support * module_mask, 0, 1)
        scalar_sum[y1:y2, x1:x2] += warped_scalar * support
        weight_sum[y1:y2, x1:x2] += support
    pv_scalar = np.zeros((height, width), dtype=np.float32)
    valid = weight_sum > 1e-5
    pv_scalar[valid] = scalar_sum[valid] / weight_sum[valid]
    pv_scalar[union_mask == 0] = 0
    return pv_scalar, union_mask


def real_farm_display_range(master_scalar):
    values = np.asarray(master_scalar, dtype=np.float32)
    low = float(np.percentile(values, 0.6)) - 5.0
    scene_high = float(np.percentile(values, 99.4))
    high = max(low + 118.0, scene_high + 58.0)
    return [round(low, 4), round(high, 4)]


def real_farm_build_thermal_master(image, modules, real_thermal=None):
    accepted, rejected = real_farm_thermal_modules(image, modules)
    mask = real_farm_module_mask(image.size, accepted, active=True)
    if real_thermal is not None:
        registered = real_thermal.convert("L").resize(image.size, Image.Resampling.BICUBIC)
        combined = np.asarray(registered, dtype=np.float32)
        background = combined.copy()
        pv_layer = np.where(mask > 0, combined, 0).astype(np.float32)
        mode = REAL_FARM_REAL_THERMAL
    else:
        background = real_farm_background_scalar(image)
        pv_layer, mask = real_farm_module_scalar_layer(image, accepted, background)
        inside = (mask > 0).astype(np.uint8)
        distance_inside = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        alpha = np.clip(distance_inside / 1.35, 0, 1).astype(np.float32)
        combined = background * (1.0 - alpha) + pv_layer * alpha
        sensor_seed = seed_for(image.width, image.height, real_farm_geometry_signature(accepted), "real-farm-sensor")
        sensor_rng = np.random.default_rng(sensor_seed)
        sensor_noise = sensor_rng.normal(0, 0.72, combined.shape).astype(np.float32)
        column_bias = sensor_rng.normal(0, 0.28, image.width).astype(np.float32)[None, :]
        combined = np.clip(combined + sensor_noise + column_bias, 0, 255).astype(np.float32)
        mode = REAL_FARM_SYNTHETIC_THERMAL
    return {
        "mode": mode,
        "background_scalar": background.astype(np.float32),
        "pv_scalar": pv_layer.astype(np.float32),
        "pv_module_union_mask": mask.astype(np.uint8),
        "master_scalar": combined.astype(np.float32),
        "display_range": real_farm_display_range(combined),
        "accepted_modules": accepted,
        "rejected_modules": rejected,
    }


def real_farm_palette_image(scalar, display_range):
    low, high = map(float, display_range)
    normalized = np.clip((np.asarray(scalar, dtype=np.float32) - low) / max(1e-6, high - low), 0, 1)
    sensor = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return temperature_to_rgb(Image.fromarray(sensor, "L"))


def real_farm_production_cell(module, observation, width, height):
    u, v = observation.get("local_uv", (0.5, 0.5))
    return {
        "id": f"{module.get('module_id')}:{observation.get('observation_id')}",
        "x1": 0,
        "y1": 0,
        "x2": int(width),
        "y2": int(height),
        "module_bbox_px": {"x1": 0, "y1": 0, "x2": int(width), "y2": int(height)},
        "active_module_bbox_px": {"x1": 1, "y1": 1, "x2": max(2, int(width) - 1), "y2": max(2, int(height) - 1)},
        "fault_origin_px": {"x": float(u) * width, "y": float(v) * height},
        "module_cell_col": int(np.clip(math.floor(float(u) * CUSTOM_MODULE_CELL_COLS) + 1, 1, CUSTOM_MODULE_CELL_COLS)),
        "module_cell_row": int(np.clip(math.floor(float(v) * CUSTOM_MODULE_CELL_ROWS) + 1, 1, CUSTOM_MODULE_CELL_ROWS)),
    }


def real_farm_apply_production_fault(master_scalar, module, observation, fault_type):
    """Run the unchanged production engine in module-local space, then homography-warp its scalar delta."""
    output = np.asarray(master_scalar, dtype=np.float32).copy()
    points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    stored_size = module.get("canonical_size") or real_farm_canonical_size(points)
    width = int(stored_size["width"])
    height = int(stored_size["height"])
    canonical_size = (width, height)
    canonical = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    to_local = cv2.getPerspectiveTransform(points, canonical)
    to_rgb = cv2.getPerspectiveTransform(canonical, points)
    base_local = cv2.warpPerspective(output, to_local, canonical_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    faulted_local = base_local.copy()
    cell = real_farm_production_cell(module, observation, width, height)
    apply_thermal_fault(faulted_local, cell, fault_type, int(observation.get("severity", 7)))
    delta_local = faulted_local - base_local
    height_rgb, width_rgb = output.shape
    delta_rgb = cv2.warpPerspective(delta_local, to_rgb, (width_rgb, height_rgb), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support_local = np.ones((height, width), dtype=np.float32)
    support_rgb = cv2.warpPerspective(support_local, to_rgb, (width_rgb, height_rgb), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    module_mask = real_farm_module_mask((width_rgb, height_rgb), [module], active=True).astype(np.float32) / 255.0
    support = np.clip(support_rgb * module_mask, 0, 1)
    output += delta_rgb * support
    return np.clip(output, 0, 255).astype(np.float32)


def real_farm_faulted_master(master_layers, observations):
    output = np.asarray(master_layers["master_scalar"], dtype=np.float32).copy()
    module_lookup = {item.get("module_id"): item for item in master_layers.get("accepted_modules", [])}
    for observation in observations or []:
        module = module_lookup.get(observation.get("module_id"))
        if not module:
            continue
        observation_type = observation.get("observation_type", "SingleHotSpot")
        if observation_type:
            output = real_farm_apply_production_fault(output, module, observation, observation_type)
        diagnostic = observation.get("diagnostic_fault_type")
        if diagnostic:
            output = real_farm_apply_production_fault(output, module, observation, diagnostic)
    return output


def real_farm_thermal_image(image, modules, observations=None, master_layers=None):
    layers = master_layers or real_farm_build_thermal_master(image, modules)
    faulted = real_farm_faulted_master(layers, observations or [])
    return real_farm_palette_image(faulted, layers["display_range"])


def real_farm_mask_preview(image_size, modules):
    mask = real_farm_module_mask(image_size, modules, active=True)
    return Image.fromarray(mask, "L")


def real_farm_observations_csv(observations):
    text = io.StringIO()
    fields = ["observation_id", "module_id", "observation_type", "diagnostic_fault_type", "severity", "center_rgb_px", "local_uv"]
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    for observation in observations or []:
        row = {key: observation.get(key) for key in fields}
        row["center_rgb_px"] = json.dumps(row["center_rgb_px"])
        row["local_uv"] = json.dumps(row["local_uv"])
        writer.writerow(row)
    return text.getvalue().encode("utf-8")


def real_farm_dataset_zip(state, image, thermal):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("project.json", real_farm_project_json(state))
        package.writestr("modules.csv", real_farm_modules_csv(state))
        package.writestr("hotspot_fault_report.csv", real_farm_observations_csv(state.get("observations", [])))
        package.writestr("hotspot_fault_report.json", json.dumps(state.get("observations", []), indent=2))
        package.writestr("thermal_mask.png", export_current_view_bytes(real_farm_mask_preview(image.size, state.get("modules", []))))
        package.writestr("thermal_scene.png", export_current_view_bytes(thermal))
        package.writestr("source_rgb.png", export_current_view_bytes(image))
    return output.getvalue()


def real_farm_get_thermal_master(state, image, modules):
    geometry_signature = real_farm_geometry_signature(modules)
    signature = f"{state.get('source_rgb_sha256_prefix', '')}:{image.width}x{image.height}:{geometry_signature}:{state.get('thermal_source', REAL_FARM_SYNTHETIC_THERMAL)}"
    cached_signature = st.session_state.get("real_farm_thermal_master_signature")
    cached = st.session_state.get("real_farm_thermal_master_layers")
    if cached is None or cached_signature != signature:
        cached = real_farm_build_thermal_master(image, modules)
        st.session_state["real_farm_thermal_master_layers"] = cached
        st.session_state["real_farm_thermal_master_signature"] = signature
    state["thermal_display_range"] = list(cached["display_range"])
    state["thermal_master_shape"] = list(cached["master_scalar"].shape)
    state["thermal_accepted_module_ids"] = [item.get("module_id") for item in cached.get("accepted_modules", [])]
    state["thermal_rejected_module_ids"] = [item.get("module_id") for item in cached.get("rejected_modules", [])]
    return cached


def real_farm_fault_workflow(state, image, modules):
    signature = real_farm_geometry_signature(modules)
    thermal_ready = bool(state.get("thermal_mask_ready"))
    if thermal_ready and state.get("thermal_geometry_signature") != signature:
        real_farm_invalidate_thermal(state)
        thermal_ready = False
        st.warning("Module geometry changed, so the previous thermal mask and hotspots were cleared. Generate the mask again.")

    st.markdown('<div class="pv-card"><div class="pv-section-title">Real-image Thermal Mask & Fault Workflow</div>', unsafe_allow_html=True)
    st.caption("The real RGB image supplies the scene. Only accepted refined PV support or an explicit manual final polygon supplies the thermal mask; a coarse search ROI is never thermalized.")
    valid_geometry, excluded_geometry = real_farm_thermal_modules(image, modules) if modules else ([], [])
    pending_geometry = [item for item in excluded_geometry if item.get("mechanically_valid") and not item.get("refinement_ready")]
    invalid_geometry = [item for item in excluded_geometry if not item.get("mechanically_valid")]
    if valid_geometry:
        st.success(f"{len(valid_geometry)} mechanically valid module polygon(s) available. Thermal mask can be generated.")
        weak_ids = [item.get("module_id", "unknown") for item in valid_geometry if not item.get("visually_supported")]
        if weak_ids:
            st.warning(f"Weak PV-texture confidence for {', '.join(weak_ids)}. Manual geometry is authoritative, so mask generation remains allowed.")
    if invalid_geometry:
        blocked_ids = ", ".join(item.get("module_id", "unknown") for item in invalid_geometry)
        st.error(f"Mechanically invalid polygon(s) are blocked: {blocked_ids}. Fix duplicate points, self-intersection, insufficient area, image coverage, or homography stability.")
    if pending_geometry:
        pending_ids = ", ".join(item.get("module_id", "unknown") for item in pending_geometry)
        st.info(f"Refinement required for coarse ROI(s): {pending_ids}. Preview and accept a refined mask, or explicitly use the manual polygon override.")
    button_label = "Regenerate Real Thermal Mask" if thermal_ready else "Generate Real Thermal Mask"
    if st.button(button_label, type="primary", use_container_width=True, disabled=not valid_geometry):
        layers = real_farm_build_thermal_master(image, modules)
        if not layers.get("accepted_modules"):
            st.error("No mechanically valid module polygon is available. Correct the polygon geometry, then generate again.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        assessed = {item.get("module_id"): item for item in layers.get("accepted_modules", []) + layers.get("rejected_modules", [])}
        for stored in state.get("modules", []):
            result = assessed.get(stored.get("module_id"))
            if result:
                for key in ("mechanically_valid", "visually_supported", "thermal_geometry_score", "geometry_confidence", "warnings", "canonical_size", "orientation_label", "orientation_confidence"):
                    if key in result:
                        stored[key] = result[key]
        state["thermal_source"] = REAL_FARM_SYNTHETIC_THERMAL
        state["thermal_is_real"] = False
        state["thermal_is_radiometric"] = False
        state["thermal_mask_ready"] = True
        state["thermal_geometry_signature"] = signature
        state["observations"] = []
        state["updated_at"] = real_farm_now()
        state.setdefault("provenance", {}).update({
            "thermal_source": REAL_FARM_SYNTHETIC_THERMAL,
            "thermal_is_real": False,
            "thermal_is_radiometric": False,
            "thermal_mask_geometry": "MANUAL_ACTIVE_MODULE_POLYGONS",
            "fault_injection_enabled": True,
            "synthetic_thermal_enabled": True,
        })
        st.session_state["real_farm_close_view"] = False
        st.session_state["real_farm_thermal_master_layers"] = layers
        cache_signature = f"{state.get('source_rgb_sha256_prefix', '')}:{image.width}x{image.height}:{signature}:{REAL_FARM_SYNTHETIC_THERMAL}"
        st.session_state["real_farm_thermal_master_signature"] = cache_signature
        st.session_state.pop("real_farm_selected_observation", None)
        st.session_state.pop("real_farm_observation_picker", None)
        st.rerun()
    if not modules:
        st.info("Add at least one module polygon before generating the thermal mask.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if not state.get("thermal_mask_ready"):
        st.info("Generate the mask to unlock hotspots, close-view fault assignment, the report, and dataset export.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if state.get("thermal_source") == "REAL_RGB_GEOMETRY_DERIVED":
        state["thermal_source"] = REAL_FARM_SYNTHETIC_THERMAL
        state.setdefault("provenance", {})["thermal_source"] = REAL_FARM_SYNTHETIC_THERMAL

    observations = state.setdefault("observations", [])
    master_layers = real_farm_get_thermal_master(state, image, modules)
    thermal_modules = master_layers.get("accepted_modules", [])
    rejected_modules = master_layers.get("rejected_modules", [])
    thermal = real_farm_thermal_image(image, thermal_modules, observations, master_layers=master_layers)
    view = st.radio("Thermal workspace", ["Thermal", "RGB"], horizontal=True, key="real_farm_thermal_view")
    if view == "RGB":
        st.image(real_farm_annotated_preview(image, modules), use_container_width=True)
    else:
        st.image(thermal, use_container_width=True)
    if rejected_modules:
        rejected_ids = ", ".join(item.get("module_id", "unknown") for item in rejected_modules)
        st.warning(f"Mechanically invalid geometry was excluded: {rejected_ids}.")
    weak_ids = [item.get("module_id", "unknown") for item in thermal_modules if not item.get("visually_supported")]
    if weak_ids:
        st.warning(f"Weak PV-texture confidence for {', '.join(weak_ids)}. The manually defined polygons remain active for masks and faults.")
    with st.expander("Developer thermal QA"):
        debug_show_mask = st.checkbox("debug_show_mask", value=False)
        if debug_show_mask:
            qa_cols = st.columns(3)
            qa_cols[0].image(image, caption="Uploaded RGB", use_container_width=True)
            qa_cols[1].image(real_farm_annotated_preview(image, thermal_modules), caption="Accepted module polygons", use_container_width=True)
            qa_cols[2].image(Image.fromarray(master_layers["pv_module_union_mask"], "L"), caption="Binary PV union mask", use_container_width=True, clamp=True)
            scalar_cols = st.columns(3)
            scalar_cols[0].image(real_farm_palette_image(master_layers["background_scalar"], master_layers["display_range"]), caption="Background scalar", use_container_width=True)
            scalar_cols[1].image(real_farm_palette_image(np.where(master_layers["pv_module_union_mask"] > 0, master_layers["pv_scalar"], master_layers["display_range"][0]), master_layers["display_range"]), caption="PV scalar field", use_container_width=True)
            scalar_cols[2].image(real_farm_palette_image(master_layers["master_scalar"], master_layers["display_range"]), caption="Combined scalar master", use_container_width=True)

    generate_col, clear_col = st.columns([0.7, 0.3])
    with generate_col:
        count_col, seed_col = st.columns(2)
        hotspot_count = count_col.number_input("Hotspots to generate", min_value=1, max_value=100, value=7, step=1)
        hotspot_seed = seed_col.number_input("Hotspot seed", min_value=0, max_value=999999, value=42, step=1)
        if st.button("Generate Hotspots", type="primary", use_container_width=True, disabled=not thermal_modules):
            state["observations"] = real_farm_generate_observations(thermal_modules, int(hotspot_count), int(hotspot_seed))
            state["updated_at"] = real_farm_now()
            selected = state["observations"][0]["observation_id"]
            st.session_state["real_farm_selected_observation"] = selected
            st.session_state["real_farm_observation_picker"] = selected
            st.session_state["real_farm_close_view"] = False
            st.rerun()
    with clear_col:
        st.write("")
        st.write("")
        if st.button("Clear Hotspots", use_container_width=True, disabled=not observations):
            state["observations"] = []
            st.session_state.pop("real_farm_selected_observation", None)
            st.session_state.pop("real_farm_observation_picker", None)
            st.session_state["real_farm_close_view"] = False
            st.rerun()

    observations = state.get("observations", [])
    if observations:
        observation_ids = [item["observation_id"] for item in observations]
        selected_id = st.session_state.get("real_farm_selected_observation")
        if selected_id not in observation_ids:
            selected_id = observation_ids[0]
            st.session_state["real_farm_selected_observation"] = selected_id
        if st.session_state.get("real_farm_observation_picker") not in observation_ids:
            st.session_state["real_farm_observation_picker"] = selected_id
        observation_lookup = {item["observation_id"]: item for item in observations}
        selected_id = st.selectbox(
            "Select generated hotspot",
            observation_ids,
            key="real_farm_observation_picker",
            format_func=lambda value: f"{value} | {observation_lookup[value]['module_id']} | {fault_display_name(observation_lookup[value].get('diagnostic_fault_type')) if observation_lookup[value].get('diagnostic_fault_type') else 'Unassigned'}",
        )
        st.session_state["real_farm_selected_observation"] = selected_id
        selected_index = observation_ids.index(selected_id)
        previous_col, close_col, next_col = st.columns(3)
        if previous_col.button("Previous Hotspot", use_container_width=True, disabled=len(observation_ids) < 2):
            target = observation_ids[(selected_index - 1) % len(observation_ids)]
            st.session_state["real_farm_selected_observation"] = target
            st.session_state["real_farm_observation_picker"] = target
            st.rerun()
        close_label = "Far View" if st.session_state.get("real_farm_close_view") else "Close View"
        if close_col.button(close_label, type="primary", use_container_width=True):
            st.session_state["real_farm_close_view"] = not st.session_state.get("real_farm_close_view", False)
            st.rerun()
        if next_col.button("Next Hotspot", use_container_width=True, disabled=len(observation_ids) < 2):
            target = observation_ids[(selected_index + 1) % len(observation_ids)]
            st.session_state["real_farm_selected_observation"] = target
            st.session_state["real_farm_observation_picker"] = target
            st.rerun()

        selected = observation_lookup[st.session_state.get("real_farm_selected_observation", selected_id)]
        selected_module = next(item for item in modules if item.get("module_id") == selected.get("module_id"))
        if st.session_state.get("real_farm_close_view"):
            st.markdown(f"#### Close View — {selected['observation_id']} on {selected['module_id']}")
            st.image(real_farm_rectified_crop(thermal, selected_module), use_container_width=True)
            diagnosis_col, severity_col = st.columns([0.72, 0.28])
            fault_options = list(CLOSE_DIAGNOSTIC_FAULT_TYPES)
            current_fault = selected.get("diagnostic_fault_type")
            default_index = fault_options.index(current_fault) if current_fault in fault_options else 0
            chosen_fault = diagnosis_col.selectbox(
                "Actual fault for this hotspot",
                fault_options,
                index=default_index,
                format_func=fault_display_name,
                key=f"real_farm_diagnosis_{selected['observation_id']}",
            )
            severity = severity_col.slider("Severity", 1, 10, int(selected.get("severity", 7)), key=f"real_farm_severity_{selected['observation_id']}")
            if st.button("Apply Fault To This Hotspot", type="primary", use_container_width=True):
                selected["diagnostic_fault_type"] = chosen_fault
                selected["severity"] = int(severity)
                selected["diagnosed_at"] = real_farm_now()
                state["updated_at"] = real_farm_now()
                st.success(f"{fault_display_name(chosen_fault)} assigned to {selected['observation_id']}. All other hotspots were preserved.")
                st.rerun()

        st.markdown("#### Review / Report")
        report_rows = [
            {
                "hotspot": item.get("observation_id"),
                "module": item.get("module_id"),
                "actual_fault": fault_display_name(item.get("diagnostic_fault_type")) if item.get("diagnostic_fault_type") else "Unassigned",
                "severity": item.get("severity"),
                "x_px": item.get("center_rgb_px", [None, None])[0],
                "y_px": item.get("center_rgb_px", [None, None])[1],
            }
            for item in observations
        ]
        st.dataframe(report_rows, use_container_width=True, hide_index=True)
        diagnosed = sum(bool(item.get("diagnostic_fault_type")) for item in observations)
        st.caption(f"{len(observations)} hotspots retained · {diagnosed} diagnosed · {len(observations) - diagnosed} unassigned")
        project_name = real_farm_safe_name(state.get("original_rgb_filename") or "real_farm")
        export_a, export_b, export_c = st.columns(3)
        export_a.download_button("Download Thermal PNG", export_current_view_bytes(thermal), file_name=f"{project_name}_thermal.png", mime="image/png", use_container_width=True)
        export_b.download_button("Download Fault Report", real_farm_observations_csv(observations), file_name=f"{project_name}_fault_report.csv", mime="text/csv", use_container_width=True)
        export_c.download_button("Export Dataset ZIP", real_farm_dataset_zip(state, image, thermal), file_name=f"{project_name}_thermal_fault_dataset.zip", mime="application/zip", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def real_farm_main():
    st.markdown("## Real PV Farm Image")
    st.caption("Upload a real RGB PV image, map the real modules, generate a geometry-aligned thermal mask, then create and diagnose hotspots through the normal fault workflow.")
    st.session_state.setdefault("real_farm_undo_stack", [])
    st.session_state.setdefault("real_farm_redo_stack", [])

    with st.sidebar:
        st.divider()
        st.caption("RF-1 Geometry")
        if st.button("New Real Farm Project", use_container_width=True):
            for key in ("real_farm_state", "real_farm_image", "real_farm_image_path", "real_farm_selected_module", "real_farm_selected_observation", "real_farm_observation_picker", "real_farm_close_view"):
                st.session_state.pop(key, None)
            st.rerun()
        undo_col, redo_col = st.columns(2)
        with undo_col:
            if st.button("Undo Geometry", use_container_width=True, disabled=not st.session_state.get("real_farm_undo_stack")):
                current = st.session_state.get("real_farm_state")
                if current:
                    st.session_state.setdefault("real_farm_redo_stack", []).append(json.loads(json.dumps(current)))
                real_farm_restore(st.session_state["real_farm_undo_stack"].pop())
                st.rerun()
        with redo_col:
            if st.button("Redo Geometry", use_container_width=True, disabled=not st.session_state.get("real_farm_redo_stack")):
                current = st.session_state.get("real_farm_state")
                if current:
                    st.session_state.setdefault("real_farm_undo_stack", []).append(json.loads(json.dumps(current)))
                real_farm_restore(st.session_state["real_farm_redo_stack"].pop())
                st.rerun()

    upload_col, load_col = st.columns([0.62, 0.38])
    with upload_col:
        uploaded = st.file_uploader("Upload one real RGB aerial/drone image", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    with load_col:
        project_upload = st.file_uploader("Load RF-1 geometry JSON", type=["json"], accept_multiple_files=False)

    if uploaded is not None and ("real_farm_uploaded_name" not in st.session_state or st.session_state["real_farm_uploaded_name"] != uploaded.name):
        try:
            image, path, _, digest = real_farm_load_uploaded_rgb(uploaded)
            st.session_state["real_farm_uploaded_name"] = uploaded.name
            st.session_state["real_farm_image"] = image
            st.session_state["real_farm_image_path"] = str(path)
            st.session_state["real_farm_state"] = real_farm_default_state(path, image, uploaded.name, digest)
            st.session_state["real_farm_selected_module"] = None
            st.session_state["real_farm_undo_stack"] = []
            st.session_state["real_farm_redo_stack"] = []
            st.success(f"Loaded {uploaded.name} at {image.width} x {image.height}px")
        except Exception as exc:
            st.error(f"Could not load image: {exc}")

    if project_upload is not None:
        try:
            loaded = real_farm_load_project(project_upload)
            image_path = Path(loaded.get("source_rgb_path", ""))
            if image_path.exists():
                image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
                st.session_state["real_farm_image"] = image
                st.session_state["real_farm_image_path"] = str(image_path)
                st.session_state["real_farm_state"] = loaded
                st.session_state["real_farm_selected_module"] = loaded.get("modules", [{}])[0].get("module_id") if loaded.get("modules") else None
                st.success("Reloaded RF-1 project exactly from JSON.")
            else:
                st.error("Project JSON loaded, but the source RGB path does not exist on this machine.")
        except Exception as exc:
            st.error(f"Could not load RF-1 project JSON: {exc}")

    state = st.session_state.get("real_farm_state")
    image = st.session_state.get("real_farm_image")
    if not state or image is None:
        st.info("Upload a real RGB image to start RF-1 manual module geometry.")
        return

    modules = state.setdefault("modules", [])
    for module in modules:
        if "mask_source" not in module:
            module["coarse_roi_polygon_rgb"] = module.get("outer_polygon_rgb", [])
            module["mask_source"] = "SEARCH_ROI"
            module["refinement_status"] = "pending"
            module["refined_mask_contours_rgb"] = []
            module["refined_pv_polygon_rgb"] = None
    image_size = (int(state["image"]["width"]), int(state["image"]["height"]))
    selected_id = st.session_state.get("real_farm_selected_module")
    if selected_id not in {item.get("module_id") for item in modules}:
        selected_id = modules[0]["module_id"] if modules else None
        st.session_state["real_farm_selected_module"] = selected_id

    stats = st.columns(5)
    stats[0].metric("Source", "Real RGB")
    stats[1].metric("Dimensions", f"{image.width} x {image.height}")
    stats[2].metric("Modules", len(modules))
    stats[3].metric("Thermal", "Mask ready" if state.get("thermal_mask_ready") else "Not generated")
    stats[4].metric("Detection", "Manual")

    editor_col, inspector_col = st.columns([0.68, 0.32])
    with editor_col:
        st.markdown('<div class="pv-card"><div class="pv-section-title">Manual Geometry Workspace</div>', unsafe_allow_html=True)
        view_mode = st.radio("Preview", ["Annotated Geometry", "Clean Original", "Rectified Crops"], horizontal=True)
        zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.25)
        pan_cols = st.columns(2)
        pan_x = pan_cols[0].slider("Pan X", 0.0, 1.0, 0.5, 0.01)
        pan_y = pan_cols[1].slider("Pan Y", 0.0, 1.0, 0.5, 0.01)
        if view_mode == "Clean Original":
            st.image(real_farm_viewport_image(image, zoom, pan_x, pan_y), use_container_width=True)
        elif view_mode == "Rectified Crops":
            st.image(real_farm_rectified_contact_sheet(image, modules), use_container_width=True)
        else:
            st.image(real_farm_viewport_image(real_farm_annotated_preview(image, modules, selected_id), zoom, pan_x, pan_y), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with inspector_col:
        st.markdown('<div class="pv-card"><div class="pv-section-title">Module Geometry</div>', unsafe_allow_html=True)
        action = st.selectbox("Action", ["Add polygon", "Edit selected", "Duplicate", "Repeat row", "Repeat grid", "Delete selected"])
        inset = st.slider("Active inset", 0.0, 0.16, 0.035, 0.005)
        array_id = st.text_input("Array/group ID optional", value="")

        if action == "Add polygon":
            default_w = max(20, image.width * 0.12)
            default_h = max(20, image.height * 0.08)
            cx = image.width * 0.5
            cy = image.height * 0.5
            coords = [
                [cx - default_w / 2, cy - default_h / 2],
                [cx + default_w / 2, cy - default_h / 2],
                [cx + default_w / 2, cy + default_h / 2],
                [cx - default_w / 2, cy + default_h / 2],
            ]
            st.caption("Create a starter quadrilateral, then edit each corner numerically. Coordinates are original-image pixels.")
            if st.button("Add Module Polygon", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                module_id = real_farm_next_module_id(modules)
                module = real_farm_module_record(module_id, coords, image_size, array_id, inset, "manual_add")
                module["confidence"]["quality_warnings"] = real_farm_quality(module["outer_polygon_rgb"], image_size, modules, module_id)
                modules.append(module)
                state["updated_at"] = real_farm_now()
                st.session_state["real_farm_selected_module"] = module_id
                st.rerun()

        selected_module = next((item for item in modules if item.get("module_id") == selected_id), None)
        if modules:
            module_ids = [item["module_id"] for item in modules]
            chosen_id = st.selectbox("Selected module", module_ids, index=module_ids.index(selected_id) if selected_id in module_ids else 0)
            if chosen_id != selected_id:
                st.session_state["real_farm_selected_module"] = chosen_id
                st.rerun()
            selected_module = next((item for item in modules if item.get("module_id") == chosen_id), None)

        if selected_module and action == "Edit selected":
            st.caption("Corner order is module-local: top-left, top-right, bottom-right, bottom-left.")
            edited = []
            for idx, (x, y) in enumerate(selected_module.get("outer_polygon_rgb", []), start=1):
                cols = st.columns(2)
                edited_x = cols[0].number_input(f"P{idx} x", value=float(x), step=1.0, key=f"rf1_{selected_id}_{idx}_x")
                edited_y = cols[1].number_input(f"P{idx} y", value=float(y), step=1.0, key=f"rf1_{selected_id}_{idx}_y")
                edited.append([edited_x, edited_y])
            new_array = st.text_input("Selected module array/group", value=selected_module.get("array_id") or "", key=f"rf1_array_{selected_id}")
            if st.button("Save Module Edits", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                selected_module["outer_polygon_rgb"] = real_farm_order_quad(edited)
                selected_module["array_id"] = new_array.strip() or None
                selected_module["active_inset_ratio"] = inset
                updated = real_farm_update_module(selected_module, image_size)
                updated["confidence"]["quality_warnings"] = real_farm_quality(updated["outer_polygon_rgb"], image_size, modules, selected_id)
                modules[modules.index(selected_module)] = updated
                state["updated_at"] = real_farm_now()
                st.rerun()

        if selected_module and action == "Duplicate":
            dx = st.number_input("Translate X px", value=40.0, step=1.0)
            dy = st.number_input("Translate Y px", value=0.0, step=1.0)
            if st.button("Duplicate Selected Module", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                new_id = real_farm_next_module_id(modules)
                shifted = [[x + dx, y + dy] for x, y in selected_module["outer_polygon_rgb"]]
                new_module = real_farm_module_record(new_id, shifted, image_size, selected_module.get("array_id") or array_id, selected_module.get("active_inset_ratio", inset), "manual_duplicate")
                modules.append(new_module)
                state["updated_at"] = real_farm_now()
                st.session_state["real_farm_selected_module"] = new_id
                st.rerun()

        if selected_module and action == "Repeat row":
            count = st.number_input("Copies to create", min_value=1, max_value=80, value=5, step=1)
            dx = st.number_input("Row spacing X px", value=45.0, step=1.0)
            dy = st.number_input("Row spacing Y px", value=0.0, step=1.0)
            if st.button("Repeat Selected Module In Row", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                base_pts = selected_module["outer_polygon_rgb"]
                for step in range(1, int(count) + 1):
                    new_id = real_farm_next_module_id(modules)
                    shifted = [[x + dx * step, y + dy * step] for x, y in base_pts]
                    modules.append(real_farm_module_record(new_id, shifted, image_size, selected_module.get("array_id") or array_id, inset, "manual_repeat_row"))
                state["updated_at"] = real_farm_now()
                st.rerun()

        if selected_module and action == "Repeat grid":
            cols = st.number_input("Grid columns", min_value=1, max_value=40, value=4, step=1)
            rows = st.number_input("Grid rows", min_value=1, max_value=40, value=3, step=1)
            dx = st.number_input("Column spacing X px", value=45.0, step=1.0)
            dy = st.number_input("Column spacing Y px", value=0.0, step=1.0)
            row_dx = st.number_input("Row spacing X px", value=0.0, step=1.0)
            row_dy = st.number_input("Row spacing Y px", value=45.0, step=1.0)
            if st.button("Repeat Selected Module As Grid", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                base_pts = selected_module["outer_polygon_rgb"]
                for row_idx in range(int(rows)):
                    for col_idx in range(int(cols)):
                        if row_idx == 0 and col_idx == 0:
                            continue
                        new_id = real_farm_next_module_id(modules)
                        shifted = [[x + dx * col_idx + row_dx * row_idx, y + dy * col_idx + row_dy * row_idx] for x, y in base_pts]
                        modules.append(real_farm_module_record(new_id, shifted, image_size, selected_module.get("array_id") or array_id, inset, "manual_repeat_grid"))
                state["updated_at"] = real_farm_now()
                st.rerun()

        if selected_module and action == "Delete selected":
            if st.button("Delete Selected Module", type="primary", use_container_width=True):
                real_farm_push_history()
                real_farm_invalidate_thermal(state)
                modules[:] = [item for item in modules if item.get("module_id") != selected_module.get("module_id")]
                state["updated_at"] = real_farm_now()
                st.session_state["real_farm_selected_module"] = modules[0]["module_id"] if modules else None
                st.rerun()

        if selected_module:
            warnings = selected_module.get("confidence", {}).get("quality_warnings", [])
            if warnings:
                for item in warnings:
                    st.warning(f"{item['code']}: {item['message']}")
            else:
                st.success("Selected module geometry passes RF-1 mechanical checks.")
        st.markdown("</div>", unsafe_allow_html=True)

    real_farm_refinement_workflow(state, image, modules, st.session_state.get("real_farm_selected_module"))
    real_farm_fault_workflow(state, image, modules)

    export_col, data_col = st.columns([0.46, 0.54])
    project_name = real_farm_safe_name(state.get("original_rgb_filename") or "real_farm_rf1")
    with export_col:
        st.markdown('<div class="pv-card"><div class="pv-section-title">Project & Geometry Export</div>', unsafe_allow_html=True)
        project_json = real_farm_project_json(state)
        st.download_button("Download Geometry JSON", project_json, file_name=f"{project_name}_rf1_geometry.json", mime="application/json", use_container_width=True)
        st.download_button("Download Module CSV", real_farm_modules_csv(state), file_name=f"{project_name}_rf1_modules.csv", mime="text/csv", use_container_width=True)
        annotated = real_farm_annotated_preview(image, modules, selected_id, max_width=2400)
        st.download_button("Download Annotated Preview", export_current_view_bytes(annotated), file_name=f"{project_name}_rf1_annotated.png", mime="image/png", use_container_width=True)
        st.download_button("Download Clean Original Preview", export_current_view_bytes(image), file_name=f"{project_name}_clean_original.png", mime="image/png", use_container_width=True)
        if st.button("Save Project JSON Locally", use_container_width=True):
            path = real_farm_save_project(state, project_name)
            st.success(f"Saved: {path}")
        st.markdown("</div>", unsafe_allow_html=True)
    with data_col:
        st.markdown('<div class="pv-card"><div class="pv-section-title">ModuleGeometry Records</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "module_id": item.get("module_id"),
                    "array_id": item.get("array_id") or "",
                    "mask_source": item.get("mask_source", "SEARCH_ROI"),
                    "refinement": item.get("refinement_status", "pending"),
                    "orientation": item.get("orientation"),
                    "warnings": len(item.get("confidence", {}).get("quality_warnings", [])),
                    "confidence": item.get("confidence", {}).get("label"),
                }
                for item in modules
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Project provenance"):
            st.json(state.get("provenance", {}))
        st.markdown("</div>", unsafe_allow_html=True)
