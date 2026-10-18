# RF-1 real-farm RGB upload and manual module geometry workflow.
from datetime import datetime, timezone
import csv
import hashlib
import io
import shutil
import plotly.graph_objects as go


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
    starter_roi = provenance_note in {
        "manual_add", "manual_edit", "manual_duplicate", "manual_repeat_row", "manual_repeat_grid",
        "automatic_full_farm_search",
    }
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
        "refined_mask_holes_rgb": [],
        "module_instances": [],
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
    fieldnames = ["module_id", "array_id", "mask_source", "refinement_status", "orientation_label", "orientation_confidence", "geometry_confidence", "mechanically_valid", "visually_supported", "canonical_size", "warnings", "coarse_roi_polygon_rgb", "refined_pv_polygon_rgb", "refined_mask_contours_rgb", "refined_mask_holes_rgb", "outer_polygon_rgb", "active_polygon_rgb"]
    writer = csv.DictWriter(text, fieldnames=fieldnames)
    writer.writeheader()
    image_record = state.get("image", {})
    image_size = (int(image_record.get("width", 0)), int(image_record.get("height", 0)))
    exported_modules = real_farm_recovered_modules(
        state.get("modules", []), image_size=image_size if all(image_size) else None
    )
    for module in exported_modules:
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
            "refined_mask_holes_rgb": json.dumps(module.get("refined_mask_holes_rgb", [])),
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


def real_farm_module_selection_chart(image, modules, selected_id=None):
    """Return a click-selectable full-scene module map backed by stable module IDs."""
    max_width = 1400
    scale = min(1.0, max_width / max(1, image.width))
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    preview = image.resize((width, height), Image.Resampling.LANCZOS) if scale < 1 else image.copy()
    centers, ids, sizes, colors, hover = [], [], [], [], []
    figure = go.Figure()
    for module in modules:
        points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2) * scale
        module_id = str(module.get("module_id"))
        center = points.mean(axis=0)
        edges = [float(np.linalg.norm(points[(index + 1) % 4] - points[index])) for index in range(4)]
        centers.append(center)
        ids.append(module_id)
        sizes.append(float(np.clip(min(edges) * 0.9, 12, 34)))
        colors.append("rgba(56,189,248,0.65)" if module_id == selected_id else "rgba(34,197,94,0.14)")
        hover.append(
            f"{module_id}<br>Orientation: {module.get('orientation_label') or module.get('orientation') or 'unknown'}"
            f"<br>Confidence: {module.get('geometry_confidence', 0):.2f}"
        )
        closed = np.vstack([points, points[0]])
        figure.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
            line={"color": "#38bdf8" if module_id == selected_id else "rgba(34,197,94,0.72)", "width": 3 if module_id == selected_id else 1.5},
        ))
    if centers:
        values = np.asarray(centers, dtype=np.float32)
        figure.add_trace(go.Scatter(
            x=values[:, 0], y=values[:, 1], mode="markers", customdata=np.asarray(ids)[:, None], text=hover,
            hovertemplate="%{text}<extra></extra>", showlegend=False,
            marker={"size": sizes, "color": colors, "line": {"color": "rgba(255,255,255,0.55)", "width": 1}},
        ))
    figure.add_layout_image(source=preview, x=0, y=0, sizex=width, sizey=height, xref="x", yref="y", sizing="stretch", layer="below")
    figure.update_xaxes(range=[0, width], visible=False, fixedrange=True)
    figure.update_yaxes(range=[height, 0], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    figure.update_layout(
        height=max(360, min(760, int(height * min(1.0, 1180 / max(1, width))))),
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor="#030712", plot_bgcolor="#030712",
        dragmode=False, clickmode="event+select", showlegend=False,
    )
    return figure


def real_farm_geometry_qa_summary(image, modules):
    accepted_parents, excluded = real_farm_thermal_modules(image, modules) if modules else ([], [])
    accepted = real_farm_recovered_modules(accepted_parents, image_size=image.size)
    detected_modules = real_farm_recovered_modules(modules, include_unaccepted=True, image_size=image.size)
    warning_count = sum(
        1 for item in detected_modules
        if any(notice.get("level") == "warning" for notice in item.get("warnings", []))
    )
    rejected = sum(
        max(1, len(item.get("module_instances") or []))
        for item in excluded if not item.get("mechanically_valid", False)
    )
    return {
        "detected": len(detected_modules), "accepted": len(accepted),
        "warnings": int(warning_count), "rejected": int(rejected),
    }


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
            "refined_mask_holes_rgb": item.get("refined_mask_holes_rgb", []),
            "module_instances": item.get("module_instances", []),
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
        if active and module.get("refined_mask_contours_rgb") and module.get("refinement_status") == "accepted":
            for contour in module.get("refined_mask_contours_rgb", []):
                points = np.rint(np.asarray(contour, dtype=np.float32)).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_8)
            for contour in module.get("refined_mask_holes_rgb", []):
                points = np.rint(np.asarray(contour, dtype=np.float32)).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 0, lineType=cv2.LINE_8)
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
    holes = (result or {}).get("refined_mask_holes_rgb") or module.get("refined_mask_holes_rgb", [])
    for contour in holes:
        points = [tuple(map(float, point)) for point in contour]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(250, 204, 21, 240), width=max(2, image.width // 1000))
    return overlay.convert("RGB")


def real_farm_instance_overlay(image, module, result=None):
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    instances = (result or {}).get("module_instances") or module.get("module_instances", [])
    line_width = max(2, image.width // 1200)
    for instance in instances:
        points = [tuple(map(float, point)) for point in instance.get("outer_polygon_rgb", [])]
        if len(points) != 4:
            continue
        draw.line(points + [points[0]], fill=(34, 211, 238, 235), width=line_width)
        draw.text(points[0], instance.get("module_id", "M"), fill=(255, 255, 255, 245))
    return overlay.convert("RGB")


def real_farm_refine_pv_support(image, module):
    """Refine a coarse ROI while preserving aligned PV components and frame gaps."""
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

    # Small, single-pass cleanup keeps the actual table outline and meaningful
    # module/frame gaps. The former scale-dependent 15x15 close (twice) fused
    # separate evidence into the smooth bar reported in the live UI.
    cleaned_local = cv2.morphologyEx(
        candidate_local, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    cleaned_local = cv2.morphologyEx(
        cleaned_local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    roi_local = roi_mask[y1:y2, x1:x2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_local, connectivity=8)
    roi_area = max(1, int(np.count_nonzero(roi_local)))
    components = []
    seeds = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(64, roi_area * 0.0015):
            continue
        component = labels == label
        overlap = int(np.count_nonzero(component & (roi_local > 0)))
        component_hue = hue[component]
        panel_fraction = float(np.mean((component_hue >= 78) & (component_hue <= 132)))
        if panel_fraction < 0.10:
            continue
        left, top, comp_width, comp_height = [int(stats[label, key]) for key in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
        )]
        component_points = cv2.findNonZero(component.astype(np.uint8))
        component_box = cv2.boxPoints(cv2.minAreaRect(component_points)).astype(np.float32)
        component_edges = [component_box[(index + 1) % 4] - component_box[index] for index in range(4)]
        component_lengths = [float(np.linalg.norm(edge)) for edge in component_edges]
        long_edge_index = int(np.argmax(component_lengths))
        long_axis = component_edges[long_edge_index] / max(1e-6, component_lengths[long_edge_index])
        data = {
            "label": label,
            "area": area,
            "overlap": overlap,
            "panel_fraction": panel_fraction,
            "centroid": np.asarray([left + comp_width * 0.5, top + comp_height * 0.5], dtype=np.float32),
            "long_axis": long_axis.astype(np.float32),
            "short_extent": float(min(component_lengths)),
        }
        components.append(data)
        if overlap >= max(12, roi_area * 0.0015):
            score = overlap / float(roi_area) + min(1.0, area / float(roi_area)) * 0.25 + panel_fraction * 0.5
            seeds.append((score, data))
    if not seeds:
        return {
            "success": False, "message": "No connected PV-material candidate intersects the coarse ROI.",
            "roi_bbox": [x1, y1, x2, y2], "raw_mask": raw_mask,
            "cleaned_mask": np.zeros_like(raw_mask), "final_mask": np.zeros_like(raw_mask),
            "refined_mask_contours_rgb": [], "refined_mask_holes_rgb": [], "refined_pv_polygon_rgb": None,
        }

    seeds.sort(key=lambda item: item[0], reverse=True)
    best_score = seeds[0][0]
    seeds = [data for score, data in seeds if score >= max(0.12, best_score * 0.52)]
    selected_labels = set(seed["label"] for seed in seeds)
    # Include disconnected modules aligned with each ROI seed's dominant axis.
    # This recovers the intended row extent without admitting neighboring rows.
    for candidate in components:
        for seed in seeds:
            axis = seed["long_axis"]
            normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
            cross_distance = abs(float(np.dot(candidate["centroid"] - seed["centroid"], normal)))
            corridor = max(18.0, 0.36 * (seed["short_extent"] + candidate["short_extent"]))
            same_orientation = abs(float(np.dot(axis, candidate["long_axis"]))) >= 0.70
            if same_orientation and cross_distance <= corridor:
                selected_labels.add(candidate["label"])
                break
    selected_local = np.isin(labels, list(selected_labels)).astype(np.uint8) * 255

    # Fill only tiny pinholes. Long grid/frame separations remain in the final mask.
    final_local = selected_local.copy()
    inverse = cv2.bitwise_not(final_local)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    tiny_hole_limit = max(20, int(roi_area * 0.00035))
    for hole_label in range(1, hole_count):
        if int(hole_stats[hole_label, cv2.CC_STAT_AREA]) <= tiny_hole_limit:
            final_local[hole_labels == hole_label] = 255

    contours, hierarchy = cv2.findContours(final_local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    stored_contours, stored_holes, exterior_points = [], [], []
    for contour_index, contour in enumerate(contours):
        if abs(float(cv2.contourArea(contour))) < max(16, roi_area * 0.00015):
            continue
        global_contour = contour.reshape(-1, 2).astype(np.float32) + np.asarray([x1, y1], dtype=np.float32)
        epsilon = max(0.8, cv2.arcLength(global_contour.reshape(-1, 1, 2), True) * 0.0012)
        simplified = cv2.approxPolyDP(global_contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        points_json = [[round(float(x), 2), round(float(y), 2)] for x, y in simplified]
        if len(hierarchy) and int(hierarchy[contour_index][3]) >= 0:
            stored_holes.append(points_json)
        else:
            stored_contours.append(points_json)
            exterior_points.append(global_contour)
    if not exterior_points:
        return {
            "success": False, "message": "PV candidates were too small after cleanup.",
            "roi_bbox": [x1, y1, x2, y2], "raw_mask": raw_mask,
            "cleaned_mask": np.zeros_like(raw_mask), "final_mask": np.zeros_like(raw_mask),
            "refined_mask_contours_rgb": [], "refined_mask_holes_rgb": [], "refined_pv_polygon_rgb": None,
        }

    quad = real_farm_order_quad(cv2.boxPoints(cv2.minAreaRect(np.concatenate(exterior_points).reshape(-1, 1, 2))))
    cleaned_mask = np.zeros((height, width), dtype=np.uint8)
    cleaned_mask[y1:y2, x1:x2] = selected_local
    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_mask[y1:y2, x1:x2] = final_local
    roi_pixels = final_mask[roi_mask > 0]
    refined_fraction = float(np.mean(roi_pixels > 0)) if roi_pixels.size else 0.0
    result = {
        "success": True,
        "message": f"Refined PV support covers {refined_fraction:.1%} of the coarse ROI; module/frame gaps remain separate.",
        "roi_bbox": [x1, y1, x2, y2],
        "processing_crop_rgb": crop,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "final_mask": final_mask,
        "connected_component_labels": labels.astype(np.int32),
        "refined_mask_contours_rgb": stored_contours,
        "refined_mask_holes_rgb": stored_holes,
        "refined_pv_polygon_rgb": quad,
        "refined_roi_fraction": round(refined_fraction, 6),
        "candidate_component_count": len(components),
        "selected_component_count": len(selected_labels),
    }
    result["module_instances"] = real_farm_detect_module_instances(image, module, final_mask, quad)
    result["module_instance_count"] = len(result["module_instances"])
    return result


def real_farm_attach_instance_support(record, source_mask, confidence=None):
    """Attach the real PV pixels inside one recovered module to its geometry record."""
    source_mask = (np.asarray(source_mask) > 0).astype(np.uint8) * 255
    instance_polygon = np.rint(np.asarray(record.get("active_polygon_rgb", []), dtype=np.float32)).astype(np.int32)
    polygon_mask = np.zeros_like(source_mask)
    cv2.fillConvexPoly(polygon_mask, instance_polygon, 255, lineType=cv2.LINE_8)
    support_mask = cv2.bitwise_and(source_mask, polygon_mask)
    polygon_area = max(1, int(np.count_nonzero(polygon_mask)))
    support_area = int(np.count_nonzero(support_mask))
    support_fraction = support_area / float(polygon_area)
    contours, hierarchy = cv2.findContours(support_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    exterior, holes = [], []
    minimum_contour_area = max(4.0, polygon_area * 0.002)
    for contour_index, contour in enumerate(contours):
        if abs(float(cv2.contourArea(contour))) < minimum_contour_area:
            continue
        contour_points = contour.reshape(-1, 2)
        stored = [[round(float(x), 2), round(float(y), 2)] for x, y in contour_points]
        if len(stored) < 3:
            continue
        if len(hierarchy) and int(hierarchy[contour_index][3]) >= 0:
            holes.append(stored)
        else:
            exterior.append(stored)
    base_confidence = float(
        record.get("base_instance_confidence", record.get("instance_confidence", 1.0))
        if confidence is None else confidence
    )
    record.update({
        "mask_source": "AUTO_REFINED_MODULE_INSTANCE",
        "refinement_status": "accepted" if exterior and support_fraction >= 0.35 else "rejected",
        "refined_mask_contours_rgb": exterior,
        "refined_mask_holes_rgb": holes,
        "pv_support_fraction": round(float(support_fraction), 6),
        "base_instance_confidence": round(float(np.clip(base_confidence, 0, 1)), 4),
        "instance_confidence": round(float(np.clip(base_confidence * support_fraction, 0, 1)), 4),
        "qa_status": "accepted" if exterior and support_fraction >= 0.55 else "warning",
    })
    return record


def real_farm_detect_module_instances(image, parent_module, union_mask, parent_quad):
    """Split each disconnected PV table with rectified RGB frame evidence."""
    source_mask = (np.asarray(union_mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(120.0, float(np.count_nonzero(source_mask)) * 0.035)
    table_contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    table_contours.sort(key=lambda contour: cv2.boundingRect(contour)[1])
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    parent_id = parent_module.get("module_id", "M")
    instances = []

    def attach_refined_support(record, confidence):
        return real_farm_attach_instance_support(record, source_mask, confidence)

    def peak_separators(score, minimum_spacing, edge_margin):
        vector = np.asarray(score, dtype=np.float32).reshape(-1, 1)
        sigma = max(1.0, minimum_spacing * 0.055)
        smooth = cv2.GaussianBlur(vector, (0, 0), sigma).ravel()
        if smooth.size < 7:
            return []
        threshold = max(float(np.percentile(smooth, 74)), float(np.median(smooth) + np.std(smooth) * 0.50))
        candidates = [
            index for index in range(2, len(smooth) - 2)
            if smooth[index] >= threshold and smooth[index] == max(smooth[index - 2:index + 3])
            and edge_margin <= index <= len(smooth) - edge_margin
        ]
        selected = []
        for index in sorted(candidates, key=lambda item: float(smooth[item]), reverse=True):
            if all(abs(index - previous) >= minimum_spacing for previous in selected):
                selected.append(index)
        return sorted(selected)

    for contour in table_contours:
        table_quad = np.asarray(real_farm_order_quad(cv2.boxPoints(cv2.minAreaRect(contour)).tolist()), dtype=np.float32)
        size = real_farm_canonical_size(table_quad)
        local_w = int(np.clip(size["width"], 80, 720))
        local_h = int(np.clip(size["height"], 80, 1800))
        destination = np.asarray([[0, 0], [local_w - 1, 0], [local_w - 1, local_h - 1], [0, local_h - 1]], dtype=np.float32)
        to_local = cv2.getPerspectiveTransform(table_quad, destination)
        to_rgb = cv2.getPerspectiveTransform(destination, table_quad)
        local_rgb = cv2.warpPerspective(rgb, to_local, (local_w, local_h), flags=cv2.INTER_CUBIC)
        local_mask = cv2.warpPerspective(source_mask, to_local, (local_w, local_h), flags=cv2.INTER_NEAREST)
        support = local_mask > 0
        if not np.any(support):
            continue
        gray = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        bright = np.clip((gray - 120.0) / 100.0, 0, 1)
        support_f = support.astype(np.float32)
        x_score = (sobel_x * support_f).sum(axis=0) / np.maximum(1.0, support_f.sum(axis=0)) + bright.mean(axis=0) * 85.0
        y_score = (sobel_y * support_f).sum(axis=1) / np.maximum(1.0, support_f.sum(axis=1)) + bright.mean(axis=1) * 85.0
        x_separators = peak_separators(x_score, max(16, int(local_w * 0.28)), max(8, int(local_w * 0.10)))
        y_separators = peak_separators(y_score, max(12, int(local_w * 0.21)), max(8, int(local_w * 0.10)))
        x_bounds = sorted(set([0, local_w - 1] + x_separators))
        y_bounds = sorted(set([0, local_h - 1] + y_separators))
        for y_index in range(len(y_bounds) - 1):
            for x_index in range(len(x_bounds) - 1):
                lx1, lx2 = x_bounds[x_index], x_bounds[x_index + 1]
                ly1, ly2 = y_bounds[y_index], y_bounds[y_index + 1]
                if lx2 - lx1 < max(12, local_w * 0.20) or ly2 - ly1 < max(10, local_w * 0.13):
                    continue
                cell_support = support[ly1:ly2 + 1, lx1:lx2 + 1]
                coverage = float(np.mean(cell_support)) if cell_support.size else 0.0
                if coverage < 0.62:
                    continue
                inset_x = max(1.0, (lx2 - lx1) * 0.018)
                inset_y = max(1.0, (ly2 - ly1) * 0.025)
                local_quad = np.asarray([
                    [lx1 + inset_x, ly1 + inset_y], [lx2 - inset_x, ly1 + inset_y],
                    [lx2 - inset_x, ly2 - inset_y], [lx1 + inset_x, ly2 - inset_y],
                ], dtype=np.float32)
                rgb_quad = cv2.perspectiveTransform(local_quad.reshape(1, -1, 2), to_rgb)[0]
                automatic_search = parent_module.get("provenance", {}).get("provenance_note") == "automatic_full_farm_search"
                module_id = f"M{len(instances) + 1:03d}" if automatic_search else f"{parent_id}.{len(instances) + 1:03d}"
                record = real_farm_module_record(module_id, rgb_quad.tolist(), image.size, parent_module.get("array_id") or parent_id, 0.018, "auto_instance")
                record.update({
                    "source_mode": REAL_FARM_SOURCE_MODE, "source": "REAL_FARM",
                    "parent_module_id": parent_id,
                })
                attach_refined_support(record, coverage)
                record["warnings"] = [] if coverage >= 0.78 else [{
                    "level": "warning", "code": "low_instance_support",
                    "message": f"{module_id} has incomplete PV support and may need manual adjustment.",
                }]
                instances.append(record)
    if not instances:
        quad = np.asarray(real_farm_order_quad(parent_quad), dtype=np.float32)
        automatic_search = parent_module.get("provenance", {}).get("provenance_note") == "automatic_full_farm_search"
        fallback_id = "M001" if automatic_search else f"{parent_id}.001"
        fallback = real_farm_module_record(fallback_id, quad.tolist(), image.size, parent_id, 0.02, "auto_instance")
        fallback.update({"source": "REAL_FARM", "parent_module_id": parent_id})
        attach_refined_support(fallback, 0.35)
        fallback["warnings"] = [{
            "level": "warning", "code": "unsplit_recovered_region",
            "message": f"{fallback.get('module_id')} could not be split confidently and remains reviewable.",
        }]
        instances = [fallback]
    return instances


def real_farm_accept_refinement(module, result):
    if not result.get("success") or not result.get("refined_pv_polygon_rgb"):
        raise ValueError("A successful refinement result is required.")
    quad = real_farm_order_quad(result["refined_pv_polygon_rgb"])
    h_local, h_inverse, determinant = real_farm_homography(quad)
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(quad)
    module["coarse_roi_polygon_rgb"] = module.get("coarse_roi_polygon_rgb") or module.get("outer_polygon_rgb")
    module["refined_pv_polygon_rgb"] = quad
    module["refined_mask_contours_rgb"] = result["refined_mask_contours_rgb"]
    module["refined_mask_holes_rgb"] = result.get("refined_mask_holes_rgb", [])
    module["module_instances"] = result.get("module_instances", [])
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


def real_farm_recover_full_farm(state, image):
    """Recover the complete farm and promote child instances to canonical geometry.

    The full-image region is retained only as search provenance.  It must never
    remain in ``state['modules']`` because that collection is consumed by the
    mask, counters, selection, thermal renderer, review and export paths.
    """
    margin_x = max(1.0, image.width * 0.005)
    margin_y = max(1.0, image.height * 0.005)
    search_polygon = [
        [margin_x, margin_y], [image.width - 1 - margin_x, margin_y],
        [image.width - 1 - margin_x, image.height - 1 - margin_y],
        [margin_x, image.height - 1 - margin_y],
    ]
    search_region = real_farm_module_record(
        "SEARCH001", search_polygon, image.size, "FULL_FARM", 0.0, "automatic_full_farm_search"
    )
    result = real_farm_refine_pv_support(image, search_region)
    if not result.get("success"):
        return result
    real_farm_accept_refinement(search_region, result)
    recovered = [dict(item) for item in result.get("module_instances", [])]
    if not recovered:
        return {"success": False, "message": "Full-image refinement did not recover any module instances."}

    # This is the one authoritative downstream geometry collection.  The
    # SEARCH001 parent is deliberately excluded: it is a search hint, not a PV
    # module and not thermal support.
    state["modules"] = recovered
    search_provenance = {
        key: value for key, value in search_region.items()
        if key not in {"module_instances", "refined_mask_contours_rgb", "refined_mask_holes_rgb"}
    }
    search_provenance["mask_source"] = "SEARCH_ROI"
    search_provenance["refinement_status"] = "search_only"
    state["search_roi"] = search_provenance
    state["panel_detection_method"] = "AUTO_LOCAL_RGB_REFINEMENT"
    state["recovered_module_ids"] = [item.get("module_id") for item in recovered]
    state.setdefault("provenance", {}).update({
        "panel_detection_method": "AUTO_LOCAL_RGB_REFINEMENT",
        "automatic_detection_enabled": True,
        "search_roi_is_thermal_support": False,
    })
    state["updated_at"] = real_farm_now()
    result["search_region"] = search_provenance
    result["module_instances"] = recovered
    result["module_instance_count"] = len(recovered)
    return result


def real_farm_refresh_recovered_support(state, image):
    """Upgrade every already-recovered child to an instance-local refined mask."""
    detected = 0
    accepted = 0
    for parent in state.get("modules", []):
        instances = parent.get("module_instances") or []
        if not instances:
            continue
        parent_support = real_farm_module_mask(image.size, [parent], active=True)
        for instance in instances:
            detected += 1
            real_farm_attach_instance_support(instance, parent_support)
            if instance.get("refinement_status") == "accepted":
                accepted += 1
    if detected == 0:
        return {"success": False, "message": "No recovered module instances are available to refine."}
    state["recovered_module_ids"] = [
        item.get("module_id") for item in real_farm_recovered_modules(
            state.get("modules", []), include_unaccepted=True, image_size=image.size
        )
    ]
    state["panel_detection_method"] = "AUTO_LOCAL_RGB_REFINEMENT"
    state.setdefault("provenance", {}).update({
        "panel_detection_method": "AUTO_LOCAL_RGB_REFINEMENT",
        "automatic_detection_enabled": True,
        "search_roi_is_thermal_support": False,
    })
    state["updated_at"] = real_farm_now()
    return {
        "success": accepted > 0,
        "message": f"Refined {accepted} of {detected} recovered modules into the full-farm PV support mask.",
        "detected": detected,
        "accepted": accepted,
    }


def real_farm_use_manual_final_polygon(module):
    final_polygon = real_farm_inset_quad(module.get("outer_polygon_rgb", []), float(module.get("active_inset_ratio", 0.035)))
    h_local, h_inverse, determinant = real_farm_homography(final_polygon)
    orientation, orientation_confidence, _, _ = real_farm_orientation_metadata(final_polygon)
    module["manual_final_polygon_rgb"] = final_polygon
    module["active_polygon_rgb"] = final_polygon
    module["refined_pv_polygon_rgb"] = None
    module["refined_mask_contours_rgb"] = []
    module["refined_mask_holes_rgb"] = []
    module["module_instances"] = []
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
            "fault_id": f"RFH{idx + 1:03d}",
            "observation_id": f"RFH{idx + 1:03d}",
            "observation_type": "SingleHotSpot",
            "module_id": module.get("module_id"),
            "local_uv": [round(u, 6), round(v, 6)],
            "center_rgb_px": [round(x, 2), round(y, 2)],
            "severity": int(rng.integers(5, 10)),
            "seed": int(seed),
            "diagnostic_fault_type": None,
            "production_reference_ids": {},
            "applied_faults": {},
            "created_at": real_farm_now(),
        })
    return observations


def real_farm_add_module_observation(modules, observations, module_id, severity, seed):
    """Create one Far View hotspot observation on the explicitly selected module."""
    module = next((item for item in modules if item.get("module_id") == module_id), None)
    if module is None:
        raise ValueError(f"Unknown Real Farm module: {module_id}")
    rng = np.random.default_rng(seed_for(module_id, int(seed), len(observations), "real-farm-ui-hotspot"))
    u = float(rng.uniform(0.28, 0.72))
    v = float(rng.uniform(0.28, 0.72))
    x, y = real_farm_local_to_rgb(module, u, v)
    used = {
        int(str(item.get("observation_id", "RFH000"))[3:])
        for item in observations
        if str(item.get("observation_id", "")).startswith("RFH")
        and str(item.get("observation_id", ""))[3:].isdigit()
    }
    next_id = 1
    while next_id in used:
        next_id += 1
    observation_id = f"RFH{next_id:03d}"
    return {
        "fault_id": observation_id,
        "observation_id": observation_id,
        "observation_type": "SingleHotSpot",
        "module_id": module_id,
        "local_uv": [round(u, 6), round(v, 6)],
        "center_rgb_px": [round(x, 2), round(y, 2)],
        "severity": int(severity),
        "seed": int(seed),
        "diagnostic_fault_type": None,
        "production_reference_ids": {},
        "applied_faults": {},
        "created_at": real_farm_now(),
    }


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


def real_farm_expand_module_instances(modules):
    expanded = []
    for parent in modules:
        instances = parent.get("module_instances") or []
        if instances:
            for instance in instances:
                item = dict(instance)
                item["visually_supported"] = parent.get("visually_supported")
                item["thermal_geometry_score"] = parent.get("thermal_geometry_score")
                expanded.append(item)
        else:
            expanded.append(dict(parent))
    return expanded


def real_farm_recovered_modules(modules, include_unaccepted=False, image_size=None):
    """Return the flat module-instance collection used by masks, counters and faults.

    Parent records are search/refinement regions only when they contain recovered
    instances. They are never counted or rasterized as modules in that case.
    """
    recovered = []
    for parent in modules or []:
        instances = parent.get("module_instances") or []
        candidates = instances if instances else [parent]
        parent_support = None
        if image_size and instances and parent.get("refined_mask_contours_rgb"):
            parent_support = real_farm_module_mask(image_size, [parent], active=True)
        for candidate in candidates:
            item = dict(candidate)
            if instances:
                item.setdefault("parent_search_roi_id", parent.get("module_id"))
                item.setdefault("visually_supported", parent.get("visually_supported"))
                item.setdefault("thermal_geometry_score", parent.get("thermal_geometry_score"))
                if parent_support is not None and not item.get("refined_mask_contours_rgb"):
                    real_farm_attach_instance_support(
                        item, parent_support, item.get("base_instance_confidence", item.get("instance_confidence", 1.0))
                    )
            accepted = (
                item.get("mechanically_valid", True)
                and item.get("refinement_status", "accepted") == "accepted"
                and item.get("mask_source", "MANUAL_FINAL_POLYGON") != "SEARCH_ROI"
            )
            if include_unaccepted or accepted:
                recovered.append(item)
    return recovered


def real_farm_sync_live_geometry(state, image):
    """Publish one canonical module collection and its exact union mask.

    Every live Streamlit consumer reads these objects.  Search ROIs and the
    selected module are intentionally absent from this state.
    """
    recovered = real_farm_recovered_modules(
        state.get("modules", []), include_unaccepted=True, image_size=image.size
    )
    accepted = real_farm_recovered_modules(
        state.get("modules", []), include_unaccepted=False, image_size=image.size
    )
    final_mask = real_farm_module_mask(image.size, accepted, active=True)
    nonzero_masks = sum(
        bool(module.get("refined_mask_contours_rgb"))
        or len(module.get("active_polygon_rgb", [])) >= 3
        for module in accepted
    )
    st.session_state["real_farm_modules"] = recovered
    st.session_state["real_farm_final_pv_mask"] = final_mask
    st.session_state["real_farm_accepted_modules"] = accepted
    st.session_state["real_farm_nonzero_masks_unioned"] = int(nonzero_masks)
    state["recovered_module_ids"] = [item.get("module_id") for item in recovered]
    state["accepted_module_ids"] = [item.get("module_id") for item in accepted]
    state["final_pv_mask_pixels"] = int(np.count_nonzero(final_mask))
    return recovered, accepted, final_mask, int(nonzero_masks)


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
            preview_cols = st.columns(3)
            preview_instances = preview.get("module_instances", [])
            preview_union_mask = real_farm_module_mask(image.size, preview_instances, active=True)
            preview_cols[0].image(real_farm_refined_overlay(image, selected, preview), caption="Blue: coarse ROI · Green: refined PV support", use_container_width=True)
            preview_cols[1].image(
                Image.fromarray(preview_union_mask, "L"),
                caption=f"Proposed module-union mask · {len(preview_instances)} instances",
                use_container_width=True, clamp=True,
            )
            preview_cols[2].image(real_farm_instance_overlay(image, selected, preview), caption=f"{preview.get('module_instance_count', 0)} recovered module instances", use_container_width=True)
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
        accepted_cols = st.columns(3)
        recovered = real_farm_recovered_modules(modules, include_unaccepted=True, image_size=image.size)
        accepted_cols[0].image(image, caption="Original RGB", use_container_width=True)
        accepted_cols[1].image(
            real_farm_annotated_preview(image, recovered, max_width=2400),
            caption=f"All {len(recovered)} recovered module outlines", use_container_width=True,
        )
        accepted_cols[2].image(
            real_farm_mask_preview(image.size, modules),
            caption="Final full-farm PV support mask", use_container_width=True, clamp=True,
        )
    elif source == "MANUAL_FINAL_POLYGON" and status == "accepted":
        st.warning("Manual override is active. The edited polygon is final thermal support; use Auto-Refine to derive support from RGB material instead.")
    else:
        st.info("Thermal generation stays locked until this coarse ROI has an accepted mask.")
    st.markdown("</div>", unsafe_allow_html=True)


def real_farm_material_masks(image):
    """Infer relative material classes; RGB guides class/structure, never final color."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    red, green, blue = [rgb[:, :, index].astype(np.int16) for index in range(3)]
    hue, saturation, value = [hsv[:, :, index] for index in range(3)]
    pv_candidate = (blue > green + 7) & (blue > red + 13) & (value > 44) & (value < 205)
    pv_candidate = cv2.morphologyEx(pv_candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    vegetation = (green > red + 8) & (green > blue + 5) & (saturation > 34)
    soil = (((hue < 34) | (hue > 168)) & (saturation > 28) & ~vegetation)
    concrete = (saturation < 48) & (value > 82)
    shadow = value < 58
    edges = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.1), cv2.CV_32F))
    metal = (edges > np.percentile(edges, 91)) & (saturation < 82) & ~vegetation
    unknown = ~(vegetation | soil | concrete | shadow | metal)
    return {
        "pv_scene_candidate": pv_candidate.astype(np.uint8) * 255,
        "vegetation": vegetation.astype(np.uint8) * 255,
        "soil": soil.astype(np.uint8) * 255,
        "roof_concrete": concrete.astype(np.uint8) * 255,
        "metal_infrastructure": metal.astype(np.uint8) * 255,
        "shadow": shadow.astype(np.uint8) * 255,
        "unknown": unknown.astype(np.uint8) * 255,
    }


def real_farm_background_scalar(image):
    """Build a new material-relative thermal scalar scene from RGB structure."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    masks = real_farm_material_masks(image)
    base = np.full((height, width), 84.0, dtype=np.float32)
    class_values = {
        "vegetation": 67.0, "soil": 109.0, "roof_concrete": 96.0,
        "metal_infrastructure": 102.0, "shadow": 48.0, "unknown": 82.0,
        "pv_scene_candidate": 158.0,
    }
    for name, scalar_value in class_values.items():
        base[masks[name] > 0] = scalar_value

    seed = seed_for(width, height, "real-farm-common-thermal-base-v2")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene_gradient = (xx / max(1, width - 1) - 0.5) * 7.0 + (0.5 - yy / max(1, height - 1)) * 5.0
    broad = thermal_noise(base.shape, np.random.default_rng(seed), 8.5, max(12.0, max(width, height) / 95.0))
    mottle = thermal_noise(base.shape, np.random.default_rng(seed_for(seed, "mottle")), 3.4, max(2.2, max(width, height) / 720.0))
    structure = gray - cv2.GaussianBlur(gray, (0, 0), max(3.0, max(width, height) / 430.0))
    edge = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.2), cv2.CV_32F))
    base += scene_gradient + broad + mottle + np.clip(structure, -32, 32) * 0.09 + np.clip(edge, 0, 24) * 0.10
    # The generated-farm multiplier is intentionally reused, but its Gaussian
    # working set grows sharply on multi-megapixel uploads. Evaluate the same
    # model on a bounded scalar grid and restore it to the thermal-master size.
    environment_seed = seed_for(seed, "thermal-cloud-sun")
    maximum_axis = max(height, width)
    if maximum_axis > 1280:
        scale = 1280.0 / maximum_axis
        working_shape = (max(32, int(round(height * scale))), max(32, int(round(width * scale))))
        environment = pv_environment_multiplier(working_shape, environment_seed, strength=0.055)
        environment = cv2.resize(environment, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        environment = pv_environment_multiplier(base.shape, environment_seed, strength=0.055)
    base *= environment
    return np.clip(base, 24, 148).astype(np.float32)


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
        module_seed = seed_for(module.get("module_id", module_index), "real-farm-pv-layer")
        pv_local = smooth_thermal_array_field((local_width, local_height), module_seed, 164, 196)
        pv_local += pv_cell_temperature_field((local_height, local_width), seed_for(module_seed, "cell-field"), 1, strength=2.6)
        local_gray = cv2.warpPerspective(gray, to_local, (local_width, local_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        seam = np.abs(cv2.Laplacian(cv2.GaussianBlur(local_gray, (0, 0), 0.8), cv2.CV_32F))
        seam /= max(1.0, float(np.percentile(seam, 96)))
        pv_local -= np.clip(seam, 0, 1) * 5.5
        local_support = np.ones((local_height, local_width), dtype=np.float32)
        roi_size = (x2 - x1, y2 - y1)
        warped_scalar = cv2.warpPerspective(pv_local.astype(np.float32), to_roi, roi_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_support = cv2.warpPerspective(local_support, to_roi, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        full_module_mask = real_farm_module_mask(image.size, [module], active=True)
        module_mask = full_module_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
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


def real_farm_build_thermal_master(image, modules, real_thermal=None, final_pv_mask=None):
    accepted_parents, rejected = real_farm_thermal_modules(image, modules)
    accepted = real_farm_recovered_modules(accepted_parents, image_size=image.size)
    mask = real_farm_module_mask(image.size, accepted, active=True)
    if final_pv_mask is not None:
        supplied_mask = np.asarray(final_pv_mask, dtype=np.uint8)
        expected_shape = (image.height, image.width)
        if supplied_mask.shape != expected_shape:
            raise ValueError(f"Canonical final PV mask has shape {supplied_mask.shape}; expected {expected_shape}.")
        mask = np.where(supplied_mask > 0, 255, 0).astype(np.uint8)
    if real_thermal is not None:
        registered = real_thermal.convert("L").resize(image.size, Image.Resampling.BICUBIC)
        combined = np.asarray(registered, dtype=np.float32)
        background = combined.copy()
        pv_layer = np.where(mask > 0, combined, 0).astype(np.float32)
        mode = REAL_FARM_REAL_THERMAL
    else:
        background = real_farm_background_scalar(image)
        pv_layer, rendered_mask = real_farm_module_scalar_layer(image, accepted, background)
        if final_pv_mask is None:
            mask = rendered_mask
        else:
            # The scalar field and the displayed mask consume the same canonical
            # live object. No selected module or search region can replace it.
            pv_layer[mask == 0] = 0
        inside = (mask > 0).astype(np.uint8)
        distance_inside = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        alpha = np.clip(distance_inside / 1.35, 0, 1).astype(np.float32)
        combined = background * (1.0 - alpha) + pv_layer * alpha
        combined = np.clip(combined, 0, 255).astype(np.float32)
        mode = REAL_FARM_SYNTHETIC_THERMAL
    return {
        "mode": mode,
        "background_scalar": background.astype(np.float32),
        "pv_scalar": pv_layer.astype(np.float32),
        "pv_module_union_mask": mask.astype(np.uint8),
        "master_scalar": combined.astype(np.float32),
        "display_range": real_farm_display_range(combined),
        "accepted_modules": accepted,
        "accepted_parent_regions": accepted_parents,
        "rejected_modules": rejected,
        "material_masks": real_farm_material_masks(image),
    }


def real_farm_palette_image(scalar, display_range=None, scene_distance="far"):
    return thermal_display_rgb_from_scalar(np.asarray(scalar, dtype=np.float32), "real_farm", scene_distance)


def real_farm_production_cell(module, observation, width, height):
    u, v = observation.get("local_uv", (0.5, 0.5))
    column_index = int(np.clip(math.floor(float(u) * CUSTOM_MODULE_CELL_COLS), 0, CUSTOM_MODULE_CELL_COLS - 1))
    row_index = int(np.clip(math.floor(float(v) * CUSTOM_MODULE_CELL_ROWS), 0, CUSTOM_MODULE_CELL_ROWS - 1))
    cell_x1 = int(round(column_index * width / CUSTOM_MODULE_CELL_COLS))
    cell_y1 = int(round(row_index * height / CUSTOM_MODULE_CELL_ROWS))
    cell_x2 = int(round((column_index + 1) * width / CUSTOM_MODULE_CELL_COLS))
    cell_y2 = int(round((row_index + 1) * height / CUSTOM_MODULE_CELL_ROWS))
    return {
        "id": f"{module.get('module_id')}:{observation.get('observation_id')}",
        "x1": cell_x1,
        "y1": cell_y1,
        "x2": max(cell_x1 + 1, cell_x2),
        "y2": max(cell_y1 + 1, cell_y2),
        "module_bbox_px": {"x1": 0, "y1": 0, "x2": int(width), "y2": int(height)},
        "active_module_bbox_px": {"x1": 1, "y1": 1, "x2": max(2, int(width) - 1), "y2": max(2, int(height) - 1)},
        "fault_origin_px": {"x": float(u) * width, "y": float(v) * height},
        "module_cell_col": column_index + 1,
        "module_cell_row": row_index + 1,
    }


def real_farm_fault_realism_compositor(base_local, production_delta, fault_type, severity, seed):
    """Blend an unchanged production signal into the local Real Farm thermal field."""
    delta = np.asarray(production_delta, dtype=np.float32)
    if not np.any(np.abs(delta) > 1e-5):
        return delta
    minimum_extent = max(8, min(delta.shape))
    if fault_type == "SingleHotSpot":
        core_sigma = max(0.75, minimum_extent * 0.0045)
        halo_sigma = max(2.0, minimum_extent * 0.026)
        core_weight, halo_weight = 0.78, 0.22
    else:
        core_sigma = max(0.55, minimum_extent * 0.0025)
        halo_sigma = max(1.4, minimum_extent * 0.014)
        core_weight, halo_weight = 0.88, 0.12
    core = cv2.GaussianBlur(delta, (0, 0), core_sigma)
    halo = cv2.GaussianBlur(delta, (0, 0), halo_sigma)
    blended = core * core_weight + halo * halo_weight

    # A deterministic low-frequency field breaks the digitally perfect edge while
    # retaining the engine's diagnostic morphology and module-local coordinates.
    rng = np.random.default_rng(int(seed))
    irregular = rng.normal(0, 1, delta.shape).astype(np.float32)
    irregular = cv2.GaussianBlur(irregular, (0, 0), max(1.2, minimum_extent * 0.035))
    irregular /= max(1e-5, float(np.std(irregular)))
    local_detail = np.asarray(base_local, dtype=np.float32) - cv2.GaussianBlur(
        np.asarray(base_local, dtype=np.float32), (0, 0), max(0.8, minimum_extent * 0.006)
    )
    texture_factor = 1.0 + np.clip(local_detail / 18.0, -0.08, 0.08)
    blended *= np.clip(1.0 + irregular * 0.075, 0.78, 1.18) * texture_factor

    # The soft knee avoids flat white clipping at ordinary severities. Very high
    # severity can still reach a hot core without becoming a pasted uniform blob.
    if fault_type == "SingleHotSpot":
        soft_limit = 21.0 + float(np.clip(severity, 1, 10)) * 3.2
    else:
        soft_limit = 25.0 + float(np.clip(severity, 1, 10)) * 3.6
    return (soft_limit * np.tanh(blended / max(1.0, soft_limit))).astype(np.float32)


def real_farm_apply_production_fault(master_scalar, module, observation, fault_type, realism=False, return_debug=False):
    """Run the unchanged production engine in module-local space, then homography-warp its scalar delta."""
    output = np.asarray(master_scalar, dtype=np.float32).copy()
    points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    stored_size = module.get("canonical_size") or real_farm_canonical_size(points)
    width = int(stored_size["width"])
    height = int(stored_size["height"])
    # Production residual banks were approved on a resolved canonical module
    # plane. Drone pixels can make a real module only tens of pixels wide;
    # upsample the canonical plane before running the unchanged engine, then
    # homography-warp the scalar residual back to the original module.
    short_side = max(1, min(width, height))
    canonical_scale = max(1.0, 256.0 / short_side)
    if max(width, height) * canonical_scale > 720.0:
        canonical_scale = 720.0 / max(width, height)
    width = max(24, int(round(width * canonical_scale)))
    height = max(24, int(round(height * canonical_scale)))
    canonical_size = (width, height)
    canonical = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    to_local = cv2.getPerspectiveTransform(points, canonical)
    to_rgb = cv2.getPerspectiveTransform(canonical, points)
    base_local = cv2.warpPerspective(output, to_local, canonical_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    faulted_local = base_local.copy()
    cell = real_farm_production_cell(module, observation, width, height)
    trace = {}
    reference_ids = observation.setdefault("production_reference_ids", {})
    production_reference_id = reference_ids.get(fault_type)
    if production_reference_id is None:
        # QA and imported records may explicitly request one reference. Do not
        # overwrite this compatibility field: one observation can contain both
        # a hotspot and a diagnostic fault from different production banks.
        production_reference_id = observation.get("production_reference_id")
    production_only = fault_type in {"SingleHotSpot", "SingleDiode", "SingleByPassed", "StringOpenCircuit"}
    try:
        apply_thermal_fault(
            faulted_local,
            cell,
            fault_type,
            int(observation.get("severity", 7)),
            production_reference_id=production_reference_id,
            production_trace=trace,
            require_production=production_only,
        )
        if fault_type == "SingleHotSpot" and production_reference_id is None:
            # Resolve the bank choice before displaying the first render.
            # A later pinned-reference call skips the bank-selection RNG draw,
            # so displaying the unpinned probe would change hotspot morphology
            # on the next UI rerun. Keep the production engine unchanged.
            resolved_reference = trace.get("reference_id")
            if resolved_reference:
                faulted_local = base_local.copy()
                trace = {}
                apply_thermal_fault(
                    faulted_local, cell, fault_type,
                    int(observation.get("severity", 7)),
                    production_reference_id=resolved_reference,
                    production_trace=trace, require_production=production_only,
                )
    except FaultGeometryError:
        failure = {
            "fault_type": fault_type,
            "target_module_id": module.get("module_id"),
            "production_fault_render_failed": True,
            "fallback_used": False,
            "failure_reason": trace.get("failure_reason", "production_reference_unavailable"),
            "production_model": trace.get("production_model"),
            "reference_bank_path": trace.get("reference_bank_path"),
            "reference_quality_status": trace.get("reference_quality_status"),
            "approved_reference_count": trace.get("approved_reference_count"),
            "required_action": trace.get("required_action"),
        }
        observation.setdefault("production_fault_debug_by_type", {})[fault_type] = failure
        observation["production_fault_debug"] = failure
        raise
    delta_local = faulted_local - base_local
    if realism:
        realism_seed = seed_for(
            module.get("module_id"), observation.get("observation_id"), fault_type,
            int(observation.get("severity", 7)), "real-farm-fault-realism"
        )
        delta_local = real_farm_fault_realism_compositor(
            base_local, delta_local, fault_type, int(observation.get("severity", 7)), realism_seed
        )
    height_rgb, width_rgb = output.shape
    delta_rgb = cv2.warpPerspective(delta_local, to_rgb, (width_rgb, height_rgb), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support_local = np.ones((height, width), dtype=np.float32)
    support_rgb = cv2.warpPerspective(support_local, to_rgb, (width_rgb, height_rgb), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    module_mask = real_farm_module_mask((width_rgb, height_rgb), [module], active=True).astype(np.float32) / 255.0
    support = np.clip(support_rgb * module_mask, 0, 1)
    output += delta_rgb * support
    output = np.clip(output, 0, 255).astype(np.float32)
    production_support_local = np.asarray(trace.get("canonical_support", np.abs(delta_local) > 1e-5), dtype=np.float32)
    production_support_rgb = cv2.warpPerspective(
        production_support_local, to_rgb, (width_rgb, height_rgb),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    production_support_rgb *= module_mask
    reference_id = trace.get("reference_id")
    debug_metadata = {
        "fault_type": fault_type,
        "production_model": trace.get("production_model"),
        "reference_id": reference_id,
        "reference_bank_path": trace.get("reference_bank_path"),
        "reference_quality_status": trace.get("reference_quality_status"),
        "canonical_residual_shape": list(delta_local.shape),
        "canonical_support_shape": list(production_support_local.shape),
        "target_module_id": module.get("module_id"),
        "target_orientation": module.get("orientation_label") or module.get("orientation"),
        "module_polygon": module.get("active_polygon_rgb"),
        "homography": np.round(to_rgb, 8).tolist(),
        "warped_residual_shape": list(delta_rgb.shape),
        "warped_support_shape": list(production_support_rgb.shape),
        "scalar_peak_before_warp": round(float(np.max(np.abs(delta_local), initial=0.0)), 6),
        "scalar_peak_after_warp": round(float(np.max(np.abs(delta_rgb * module_mask), initial=0.0)), 6),
        "support_area_before_warp": int(np.count_nonzero(production_support_local > 0.05)),
        "support_area_after_warp": int(np.count_nonzero(production_support_rgb > 0.05)),
        "fallback_used": bool(trace.get("fallback_used", False)),
        "production_fault_render_failed": bool(trace.get("production_fault_render_failed", False)),
    }
    reference_ids[fault_type] = reference_id
    observation.setdefault("production_fault_debug_by_type", {})[fault_type] = debug_metadata
    observation["production_fault_debug"] = debug_metadata
    observation.setdefault("applied_faults", {})[fault_type] = {
        "fault_id": f"{observation.get('observation_id', 'RFH')}:{fault_type}",
        "fault_type": fault_type,
        "module_id": module.get("module_id"),
        "severity": int(observation.get("severity", 7)),
        "production_reference_id": reference_id,
        "production_model": debug_metadata["production_model"],
        "canonical_residual_provenance": {
            "reference_bank_path": debug_metadata["reference_bank_path"],
            "reference_quality_status": debug_metadata["reference_quality_status"],
            "canonical_residual_shape": debug_metadata["canonical_residual_shape"],
            "canonical_support_shape": debug_metadata["canonical_support_shape"],
        },
        "homography": debug_metadata["homography"],
        "support_geometry": {
            "active_polygon_rgb": debug_metadata["module_polygon"],
            "support_area_before_warp": debug_metadata["support_area_before_warp"],
            "support_area_after_warp": debug_metadata["support_area_after_warp"],
        },
        "scalar_strength": {
            "peak_before_warp": debug_metadata["scalar_peak_before_warp"],
            "peak_after_warp": debug_metadata["scalar_peak_after_warp"],
        },
        "seed": int(observation.get("seed", seed_for(observation.get("observation_id"), fault_type, observation.get("severity", 7)))),
        "created_at": observation.get("created_at") or real_farm_now(),
        "fallback_used": False,
        "production_fault_render_failed": False,
    }
    if return_debug:
        return output, {
            "metadata": debug_metadata,
            "source_field": trace.get("source_field"),
            "source_support": trace.get("source_support"),
            "canonical_residual": delta_local,
            "canonical_support": production_support_local,
            "warped_residual": delta_rgb * module_mask,
            "warped_support": production_support_rgb,
            "healthy_canonical": base_local,
            "faulted_canonical": faulted_local,
        }
    return output


def real_farm_faulted_master(master_layers, observations):
    output = np.asarray(master_layers["master_scalar"], dtype=np.float32).copy()
    module_lookup = {item.get("module_id"): item for item in master_layers.get("accepted_modules", [])}
    for observation in observations or []:
        # Export only faults rendered this pass; keep reference IDs pinned for reuse.
        observation["applied_faults"] = {}
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
    sensor = apply_sensor_model(faulted, "real_farm", "far")
    return thermal_display_rgb_from_scalar(sensor, "real_farm", "far")


def real_farm_close_thermal_image(faulted_master_scalar, module):
    """Rectify scalar data first so the close-view sensor retains production morphology."""
    points = np.asarray(module.get("active_polygon_rgb", []), dtype=np.float32).reshape(4, 2)
    stored_size = module.get("canonical_size") or real_farm_canonical_size(points)
    source_width = max(1, int(stored_size["width"]))
    source_height = max(1, int(stored_size["height"]))
    scale = min(1024.0 / max(source_width, source_height), max(1.0, 512.0 / min(source_width, source_height)))
    width = max(96, int(round(source_width * scale)))
    height = max(96, int(round(source_height * scale)))
    canonical = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    to_local = cv2.getPerspectiveTransform(points, canonical)
    scalar = cv2.warpPerspective(
        np.asarray(faulted_master_scalar, dtype=np.float32), to_local, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
    )
    sensor = apply_sensor_model(scalar, "real_farm", "close")
    return thermal_display_rgb_from_scalar(sensor, "real_farm", "close")


def real_farm_mask_preview(image_size, modules):
    mask = real_farm_module_mask(image_size, real_farm_recovered_modules(modules, image_size=image_size), active=True)
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


def real_farm_fault_metadata(observations):
    records = []
    for observation in observations or []:
        applied = observation.get("applied_faults") or {}
        for fault_type, record in applied.items():
            item = dict(record)
            item.setdefault("fault_type", fault_type)
            item["observation_id"] = observation.get("observation_id")
            records.append(item)
    return records


def real_farm_dataset_zip(state, image, faulted_thermal, healthy_thermal=None, annotated_thermal=None):
    healthy_thermal = healthy_thermal or faulted_thermal
    exported_modules = real_farm_recovered_modules(state.get("modules", []), image_size=image.size)
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("project.json", real_farm_project_json(state))
        package.writestr("module_geometry.json", json.dumps(exported_modules, indent=2))
        package.writestr("modules.csv", real_farm_modules_csv(state))
        package.writestr("hotspot_fault_report.csv", real_farm_observations_csv(state.get("observations", [])))
        package.writestr("hotspot_fault_report.json", json.dumps(state.get("observations", []), indent=2))
        package.writestr("fault_metadata.json", json.dumps(real_farm_fault_metadata(state.get("observations", [])), indent=2))
        package.writestr("thermal_mask.png", export_current_view_bytes(real_farm_mask_preview(image.size, state.get("modules", []))))
        package.writestr("healthy_thermal_clean.png", export_current_view_bytes(healthy_thermal))
        package.writestr("faulted_thermal_clean.png", export_current_view_bytes(faulted_thermal))
        package.writestr("source_rgb_clean.png", export_current_view_bytes(image))
        # Backward-compatible clean aliases used by existing dataset consumers.
        package.writestr("thermal_scene.png", export_current_view_bytes(faulted_thermal))
        package.writestr("source_rgb.png", export_current_view_bytes(image))
        if annotated_thermal is not None:
            package.writestr("faulted_thermal_annotated.png", export_current_view_bytes(annotated_thermal))
    return output.getvalue()


def real_farm_get_thermal_master(state, image, modules):
    geometry_signature = real_farm_geometry_signature(modules)
    signature = f"{state.get('source_rgb_sha256_prefix', '')}:{image.width}x{image.height}:{geometry_signature}:{state.get('thermal_source', REAL_FARM_SYNTHETIC_THERMAL)}"
    cached_signature = st.session_state.get("real_farm_thermal_master_signature")
    cached = st.session_state.get("real_farm_thermal_master_layers")
    if cached is None or cached_signature != signature:
        cached = real_farm_build_thermal_master(
            image, modules, final_pv_mask=st.session_state.get("real_farm_final_pv_mask")
        )
        st.session_state["real_farm_thermal_master_layers"] = cached
        st.session_state["real_farm_thermal_master_signature"] = signature
    state["thermal_display_range"] = list(cached["display_range"])
    state["thermal_master_shape"] = list(cached["master_scalar"].shape)
    state["thermal_accepted_module_ids"] = [item.get("module_id") for item in cached.get("accepted_modules", [])]
    state["thermal_rejected_module_ids"] = [item.get("module_id") for item in cached.get("rejected_modules", [])]
    return cached


def real_farm_review_panel(state, image, healthy_thermal, faulted_thermal, faulted_master_scalar, thermal_modules, selected_module_id):
    observations = state.get("observations", [])
    selected_observation_id = st.session_state.get("real_farm_selected_observation")
    selected_observation = next((item for item in observations if item.get("observation_id") == selected_observation_id), None)
    if selected_observation:
        selected_module_id = selected_observation.get("module_id")
    selected_module = next((item for item in thermal_modules if item.get("module_id") == selected_module_id), None)
    columns = st.columns(3)
    columns[0].image(image, caption="Clean Real RGB", use_container_width=True)
    columns[1].image(healthy_thermal, caption="Healthy Real V2 Thermal", use_container_width=True)
    columns[2].image(faulted_thermal, caption="Faulted Real Thermal", use_container_width=True)
    if selected_module is None:
        st.info("Select a recovered module in Workspace to inspect its Review record.")
        return
    detail_columns = st.columns([0.58, 0.42])
    detail_columns[0].image(
        real_farm_close_thermal_image(faulted_master_scalar, selected_module),
        caption=f"Selected-module Close View · {selected_module_id}", use_container_width=True,
    )
    diagnostic = selected_observation.get("diagnostic_fault_type") if selected_observation else None
    active_fault = diagnostic or (selected_observation.get("observation_type") if selected_observation else None)
    reference_ids = (selected_observation or {}).get("production_reference_ids", {})
    reference_id = reference_ids.get(active_fault) if active_fault else None
    detail_columns[1].markdown(
        f"**Module ID:** `{selected_module_id}`  \n"
        f"**Fault:** {fault_display_name(active_fault) if active_fault else 'No applied fault'}  \n"
        f"**Severity:** {(selected_observation or {}).get('severity', '—')}  \n"
        f"**Production reference:** `{reference_id or '—'}`  \n"
        f"**Orientation:** `{selected_module.get('orientation_label') or selected_module.get('orientation') or 'unknown'}`  \n"
        f"**Geometry confidence:** `{float(selected_module.get('geometry_confidence') or 0):.3f}`"
    )
    with st.expander("Developer fault provenance"):
        st.json((selected_observation or {}).get("production_fault_debug_by_type", {}))


def real_farm_export_panel(state, image, healthy_thermal, faulted_thermal, thermal_modules, selected_module_id):
    project_name = real_farm_safe_name(state.get("original_rgb_filename") or "real_farm")
    annotated = real_farm_annotated_preview(faulted_thermal, thermal_modules, selected_module_id, max_width=2400)
    st.image(faulted_thermal, caption="Clean faulted thermal export preview", use_container_width=True)
    row_a = st.columns(3)
    row_a[0].download_button("Download Clean RGB", export_current_view_bytes(image), file_name=f"{project_name}_rgb_clean.png", mime="image/png", use_container_width=True)
    row_a[1].download_button("Download Healthy Thermal", export_current_view_bytes(healthy_thermal), file_name=f"{project_name}_healthy_thermal.png", mime="image/png", use_container_width=True)
    row_a[2].download_button("Download Faulted Thermal", export_current_view_bytes(faulted_thermal), file_name=f"{project_name}_faulted_thermal.png", mime="image/png", use_container_width=True)
    row_b = st.columns(3)
    row_b[0].download_button("Download Annotated Thermal", export_current_view_bytes(annotated), file_name=f"{project_name}_faulted_annotated.png", mime="image/png", use_container_width=True)
    row_b[1].download_button("Download Module Geometry JSON", json.dumps(thermal_modules, indent=2), file_name=f"{project_name}_module_geometry.json", mime="application/json", use_container_width=True)
    row_b[2].download_button("Download Fault Metadata JSON", json.dumps(real_farm_fault_metadata(state.get("observations", [])), indent=2), file_name=f"{project_name}_fault_metadata.json", mime="application/json", use_container_width=True)
    st.download_button(
        "Export Complete Real Farm Dataset ZIP",
        real_farm_dataset_zip(state, image, faulted_thermal, healthy_thermal, annotated),
        file_name=f"{project_name}_real_farm_dataset.zip", mime="application/zip", use_container_width=True,
    )
    st.caption("Clean exports contain no debug masks, selection outlines, labels, or support overlays. The annotated image is provided separately.")


def real_farm_fault_workflow(state, image, modules):
    canonical_modules = st.session_state.get("real_farm_modules", modules)
    canonical_mask = st.session_state.get("real_farm_final_pv_mask")
    modules = canonical_modules
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
    geometry_summary = real_farm_geometry_qa_summary(image, modules)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Detected modules", geometry_summary["detected"])
    summary_columns[1].metric("Accepted modules", geometry_summary["accepted"])
    summary_columns[2].metric("Warnings", geometry_summary["warnings"])
    summary_columns[3].metric("Rejected geometry", geometry_summary["rejected"])
    recovered_geometry = real_farm_recovered_modules(valid_geometry, image_size=image.size)
    if recovered_geometry:
        st.success(f"{len(recovered_geometry)} accepted recovered module(s) available. The full-farm union mask can be generated.")
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
        layers = real_farm_build_thermal_master(image, modules, final_pv_mask=canonical_mask)
        if not layers.get("accepted_modules"):
            st.error("No mechanically valid module polygon is available. Correct the polygon geometry, then generate again.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        assessed = {item.get("module_id"): item for item in layers.get("accepted_parent_regions", []) + layers.get("rejected_modules", [])}
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
            "thermal_mask_geometry": "UNION_OF_REFINED_RECOVERED_MODULE_MASKS",
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
        st.info("Recover PV modules before generating the thermal mask.")
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
    if thermal_modules:
        st.caption(f"{len(thermal_modules)} recovered module instance(s) drive PV variation, hotspot placement, close view, and exports.")
    healthy_master_scalar = np.asarray(master_layers["master_scalar"], dtype=np.float32)
    healthy_sensor = apply_sensor_model(healthy_master_scalar, "real_farm", "far")
    healthy_thermal = thermal_display_rgb_from_scalar(healthy_sensor, "real_farm", "far")
    try:
        faulted_master_scalar = real_farm_faulted_master(master_layers, observations)
        state.pop("production_fault_error", None)
    except FaultGeometryError as exc:
        faulted_master_scalar = healthy_master_scalar.copy()
        state["production_fault_error"] = str(exc)
        st.error("Production fault reference unavailable.")
        with st.expander("Developer production failure"):
            st.code(str(exc))
    thermal_sensor = apply_sensor_model(faulted_master_scalar, "real_farm", "far")
    thermal = thermal_display_rgb_from_scalar(thermal_sensor, "real_farm", "far")
    module_ids = [item.get("module_id") for item in thermal_modules]
    selected_module_id = st.session_state.get("real_farm_target_module")
    if selected_module_id not in module_ids:
        selected_module_id = module_ids[0] if module_ids else None
        st.session_state["real_farm_target_module"] = selected_module_id

    stage = st.radio("Real Farm workflow", ["Workspace", "Review", "Export"], horizontal=True, key="real_farm_workflow_stage")
    if stage == "Review":
        real_farm_review_panel(state, image, healthy_thermal, thermal, faulted_master_scalar, thermal_modules, selected_module_id)
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if stage == "Export":
        real_farm_export_panel(state, image, healthy_thermal, thermal, thermal_modules, selected_module_id)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Select Target Module")
    st.caption("Click a recovered module marker in the scene. The selected stable module ID drives hotspot placement, Close View, Review, and Export.")
    module_event = st.plotly_chart(
        real_farm_module_selection_chart(image, thermal_modules, selected_module_id),
        key="real_farm_module_scene_picker", on_select="rerun", selection_mode="points",
        use_container_width=True, config={"displayModeBar": False, "scrollZoom": False},
    )
    selected_points = ((module_event or {}).get("selection") or {}).get("points", []) if isinstance(module_event, dict) else getattr(getattr(module_event, "selection", None), "points", [])
    if selected_points:
        custom = selected_points[0].get("customdata")
        clicked_module_id = custom[0] if isinstance(custom, (list, tuple)) else custom
        if clicked_module_id in module_ids and clicked_module_id != selected_module_id:
            st.session_state["real_farm_target_module"] = clicked_module_id
            st.session_state["real_farm_target_module_select"] = clicked_module_id
            st.rerun()
    if module_ids and st.session_state.get("real_farm_target_module_select") not in module_ids:
        st.session_state["real_farm_target_module_select"] = selected_module_id
    selected_module_id = st.selectbox(
        "Selected real module", module_ids, key="real_farm_target_module_select",
    ) if module_ids else None
    if selected_module_id and selected_module_id != st.session_state.get("real_farm_target_module"):
        st.session_state["real_farm_target_module"] = selected_module_id
        st.rerun()

    view = st.radio("Full-scene view", ["Faulted Thermal", "Healthy Thermal", "RGB + Accepted Modules"], horizontal=True, key="real_farm_thermal_view")
    if view == "RGB + Accepted Modules":
        st.image(real_farm_annotated_preview(image, thermal_modules, selected_module_id), use_container_width=True)
    elif view == "Healthy Thermal":
        st.image(healthy_thermal, use_container_width=True)
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

    st.markdown("#### Far View Observation")
    st.caption("Thermal Hotspot uses the frozen production bank. Partial Shading remains unavailable in Real Farm production mode.")
    target_col, severity_col, seed_col = st.columns([0.42, 0.28, 0.30])
    target_col.metric("Target module", selected_module_id or "—")
    selected_hotspot_severity = severity_col.slider("Hotspot severity", 1, 10, 7, key="real_farm_selected_hotspot_severity")
    selected_hotspot_seed = seed_col.number_input("Fault seed", min_value=0, max_value=999999, value=42, step=1, key="real_farm_selected_hotspot_seed")
    if st.button("Add Thermal Hotspot To Selected Module", type="primary", use_container_width=True, disabled=not selected_module_id):
        observation = real_farm_add_module_observation(
            thermal_modules, observations, selected_module_id, selected_hotspot_severity, selected_hotspot_seed,
        )
        observations.append(observation)
        state["updated_at"] = real_farm_now()
        st.session_state["real_farm_selected_observation"] = observation["observation_id"]
        st.session_state["real_farm_observation_picker"] = observation["observation_id"]
        st.session_state["real_farm_close_view"] = False
        st.rerun()

    st.caption("Batch generation remains available for multi-observation demonstrations.")
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
        selected_module = next(item for item in thermal_modules if item.get("module_id") == selected.get("module_id"))
        if st.session_state.get("real_farm_close_view"):
            st.markdown(f"#### Close View — {selected['observation_id']} on {selected['module_id']}")
            st.image(real_farm_close_thermal_image(faulted_master_scalar, selected_module), use_container_width=True)
            diagnosis_col, severity_col = st.columns([0.72, 0.28])
            fault_options = list(CLOSE_DIAGNOSTIC_FAULT_TYPES)
            supported_real_farm_diagnostics = {"SingleDiode", "SingleByPassed"}
            current_fault = selected.get("diagnostic_fault_type")
            default_index = fault_options.index(current_fault) if current_fault in fault_options else fault_options.index("SingleDiode")
            chosen_fault = diagnosis_col.selectbox(
                "Actual fault for this hotspot",
                fault_options,
                index=default_index,
                format_func=fault_display_name,
                key=f"real_farm_diagnosis_{selected['observation_id']}",
            )
            severity = severity_col.slider("Severity", 1, 10, int(selected.get("severity", 7)), key=f"real_farm_severity_{selected['observation_id']}")
            unsupported_diagnostic = chosen_fault not in supported_real_farm_diagnostics
            if unsupported_diagnostic:
                if chosen_fault == "StringOpenCircuit":
                    st.warning(
                        "Open-Circuit / Offline Module Pattern remains production-disabled. "
                        "Five semantically approved real references establish the fault signature, "
                        "but the qualified bank is too small for defensible production variation. "
                        "No synthetic or cross-fault fallback will be used."
                    )
                else:
                    st.warning(f"{fault_display_name(chosen_fault)} is visible for production-status transparency but is not supported in Real Farm mode.")
            if st.button("Apply Fault To This Hotspot", type="primary", use_container_width=True, disabled=unsupported_diagnostic):
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
        st.caption("Use the Review and Export stages above for clean state comparison and complete dataset packaging.")
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
            for key in (
                "real_farm_state", "real_farm_image", "real_farm_image_path", "real_farm_selected_module",
                "real_farm_selected_observation", "real_farm_observation_picker", "real_farm_close_view",
                "real_farm_target_module", "real_farm_target_module_select", "real_farm_workflow_stage",
                "real_farm_uploaded_name", "real_farm_modules", "real_farm_accepted_modules",
                "real_farm_final_pv_mask", "real_farm_nonzero_masks_unioned",
            ):
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
            for key in (
                "real_farm_modules", "real_farm_accepted_modules", "real_farm_final_pv_mask",
                "real_farm_nonzero_masks_unioned", "real_farm_thermal_master_layers",
                "real_farm_thermal_master_signature",
            ):
                st.session_state.pop(key, None)
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
        st.info("Upload a real RGB image, then recover the full PV module collection.")
        return

    modules = state.setdefault("modules", [])
    for module in modules:
        if "mask_source" not in module:
            module["coarse_roi_polygon_rgb"] = module.get("outer_polygon_rgb", [])
            module["mask_source"] = "SEARCH_ROI"
            module["refinement_status"] = "pending"
            module["refined_mask_contours_rgb"] = []
            module["refined_mask_holes_rgb"] = []
            module["module_instances"] = []
            module["refined_pv_polygon_rgb"] = None
    image_size = (int(state["image"]["width"]), int(state["image"]["height"]))
    recovered_for_ui = real_farm_recovered_modules(modules, include_unaccepted=True, image_size=image.size)
    recovery_columns = st.columns([0.72, 0.28])
    recovery_columns[0].caption(
        "Normal workflow: recover all modules, review the full-farm mask, then generate thermal. "
        "The search region is only a location hint and is never thermalized."
    )
    if recovery_columns[1].button("Detect & Refine Full PV Farm", type="primary", use_container_width=True):
        real_farm_push_history()
        real_farm_invalidate_thermal(state)
        # Always start the primary workflow from the complete source image.
        # Existing manual M001/ROI state must never divert automatic recovery.
        recovery = real_farm_recover_full_farm(state, image)
        if recovery.get("success"):
            recovered = recovery.get("module_instances", [])
            st.session_state["real_farm_selected_module"] = recovered[0].get("module_id") if recovered else None
            st.session_state.pop("real_farm_recovery_error", None)
            st.rerun()
        st.session_state["real_farm_recovery_error"] = recovery.get("message", "Full-farm recovery failed.")
    if st.session_state.get("real_farm_recovery_error"):
        st.error(st.session_state["real_farm_recovery_error"])

    modules = state.setdefault("modules", [])
    recovered_for_ui, accepted_for_ui, final_pv_mask, nonzero_masks = real_farm_sync_live_geometry(state, image)
    selected_id = st.session_state.get("real_farm_selected_module")
    if selected_id not in {item.get("module_id") for item in modules}:
        selected_id = modules[0]["module_id"] if modules else None
        st.session_state["real_farm_selected_module"] = selected_id

    geometry_summary = real_farm_geometry_qa_summary(image, modules)
    stats = st.columns(5)
    stats[0].metric("Source", "Real RGB")
    stats[1].metric("Dimensions", f"{image.width} x {image.height}")
    stats[2].metric("Modules", geometry_summary["detected"])
    stats[3].metric("Thermal", "Mask ready" if state.get("thermal_mask_ready") else "Not generated")
    stats[4].metric("Detection", "Recovered" if recovered_for_ui else "Pending")

    st.markdown('<div class="pv-card"><div class="pv-section-title">Full-Farm PV Support Review</div>', unsafe_allow_html=True)
    if recovered_for_ui:
        review_columns = st.columns(3)
        review_columns[0].image(image, caption="A · Original RGB", use_container_width=True)
        review_columns[1].image(
            real_farm_annotated_preview(image, recovered_for_ui, selected_id, max_width=2400),
            caption=f"B · {len(recovered_for_ui)} automatically recovered module outlines",
            use_container_width=True,
        )
        review_columns[2].image(
            Image.fromarray(final_pv_mask, "L"),
            caption=f"C · Final union of {len(accepted_for_ui)} accepted module masks",
            use_container_width=True, clamp=True,
        )
        mask_coverage = 100.0 * float(np.count_nonzero(final_pv_mask)) / max(1, final_pv_mask.size)
        st.caption(
            "Recovered module source: FULL_IMAGE_AUTOMATIC · "
            f"Recovered modules: {len(recovered_for_ui)} · "
            f"Accepted module masks: {len(accepted_for_ui)} · "
            f"Nonzero masks unioned: {nonzero_masks} · "
            f"Final mask coverage: {mask_coverage:.2f}%"
        )
    else:
        st.info("Click Detect & Refine Full PV Farm. No polygon or M001 starter is required.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Advanced / Manual Geometry Correction", expanded=False):
        editor_col, inspector_col = st.columns([0.68, 0.32])
        with editor_col:
            st.markdown('<div class="pv-card"><div class="pv-section-title">Recovered Module Review</div>', unsafe_allow_html=True)
            view_mode = st.radio("Preview", ["Annotated Geometry", "Clean Original", "Rectified Crops"], horizontal=True)
            zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.25)
            pan_cols = st.columns(2)
            pan_x = pan_cols[0].slider("Pan X", 0.0, 1.0, 0.5, 0.01)
            pan_y = pan_cols[1].slider("Pan Y", 0.0, 1.0, 0.5, 0.01)
            if view_mode == "Clean Original":
                st.image(real_farm_viewport_image(image, zoom, pan_x, pan_y), use_container_width=True)
            elif view_mode == "Rectified Crops":
                st.image(real_farm_rectified_contact_sheet(image, recovered_for_ui), use_container_width=True)
            else:
                preview_geometry = recovered_for_ui if recovered_for_ui else modules
                st.image(real_farm_viewport_image(real_farm_annotated_preview(image, preview_geometry, selected_id), zoom, pan_x, pan_y), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with inspector_col:
            st.markdown('<div class="pv-card"><div class="pv-section-title">Advanced / Manual Geometry Correction</div>', unsafe_allow_html=True)
            st.caption("Optional tools for missed modules, failed refinement, or explicit geometry overrides.")
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
        annotated = real_farm_annotated_preview(image, recovered_for_ui, selected_id, max_width=2400)
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
                    "warnings": sum(
                        notice.get("level") == "warning" for notice in item.get("warnings", [])
                    ),
                    "confidence": item.get("geometry_confidence"),
                }
                for item in recovered_for_ui
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Project provenance"):
            st.json(state.get("provenance", {}))
        st.markdown("</div>", unsafe_allow_html=True)
