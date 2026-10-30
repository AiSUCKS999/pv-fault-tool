"""Project existing RGB surface effects onto source-supported Real Farm modules."""
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

VISIBLE_RGB_EFFECTS = {"SurfaceObstruction", "SoilingDust", "CellCracking"}


def view_mode_changed(st):
    st.session_state["real_farm_camera_healthy"] = False
    st.session_state["real_farm_camera_grayscale"] = False


def view_bounds(ns, image, selected, candidates, view):
    """Same close-context geometry as the existing thermal viewer."""
    if view != "Close View":
        return None
    if selected:
        context = candidates
        provisional = selected.get("simulation_only")
        if provisional:
            area = abs(ns["real_farm_polygon_area"](selected["outer_polygon_rgb"]))
            context = [selected] + [m for m in candidates if abs(ns["real_farm_polygon_area"](m["outer_polygon_rgb"])) >= area * .5]
        bounds = ns["real_farm_close_context_bounds"](selected, context, (image.height, image.width), .75 if provisional else .25)
        if provisional:
            cx, cy = np.float32(selected["active_polygon_rgb"]).mean(0)
            radius = selected["simulation_context_radius"]
            bounds = (max(0, min(bounds[0], int(cx - radius))), max(0, min(bounds[1], int(cy - radius))),
                      min(image.width, max(bounds[2], int(cx + radius + 1))), min(image.height, max(bounds[3], int(cy + radius + 1))))
        return bounds
    return None


def rgb_fault_preview(ns, image, modules, observations):
    """No LUT, heat tint or changes to the original photo, masks or fault engines."""
    source = ns["real_farm_source_identity"](image)
    lookup = {m["module_id"]: m for m in modules}
    signature_data = {"source": source, "observations": [{k: o.get(k) for k in
        ("observation_id", "module_id", "severity", "seed", "local_uv", "diagnostic_fault_type", "observation_type")} for o in observations],
        "geometry": [{k: m.get(k) for k in ("module_id", "active_polygon_rgb", "refined_mask_contours_rgb", "refined_mask_holes_rgb", "refinement_status")}
                     for m in modules if any(o.get("module_id") == m["module_id"] for o in observations)],
        "version": "visible-surface-only-source-projection-v3",
        "engine_revision": (ns["ROOT"] / "pv_render_rgb.py").stat().st_mtime_ns}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True, default=lambda a: a.tolist()).encode()).hexdigest()
    cached = ns["st"].session_state.get("real_farm_rgb_preview_cache")
    if cached and cached[0] == signature:
        return cached[1], cached[2]
    output = np.array(image.convert("RGB"), copy=True)
    report = {"source_rgb_pixel_sha256": source, "rendered": [], "unavailable": [],
              "simulation_only": True, "physical_ground_truth_verified": False, "palette": "original natural RGB; existing RGB surface effects"}
    width, height = ns["BASE_SIZE"]
    for observation in observations:
        fault = observation.get("diagnostic_fault_type") or observation.get("observation_type", "SingleHotSpot")
        module = lookup.get(observation.get("module_id"))
        reason = None
        if observation.get("source_rgb_pixel_sha256", source) != source:
            reason = "Observation source mismatch"
        elif fault not in VISIBLE_RGB_EFFECTS:
            reason = "No established daylight RGB surface effect for this electrical/thermal diagnosis"
        elif not module or module.get("physical_module_unresolved") or module.get("placement_kind") == "pv_point":
            reason = "A resolved physical module is required for an RGB surface preview"
        elif not ns["real_farm_mechanical_validation"](np.float32(module["active_polygon_rgb"]), image.size)["mechanically_valid"]:
            reason = "Invalid target polygon"
        if reason:
            report["unavailable"].append({"observation_id": observation.get("observation_id"), "fault_type": fault, "reason": reason})
            continue
        quad = np.float32(module["active_polygon_rgb"])
        canonical = ns["real_farm_canonical_size"](quad)
        scale = min((width - 48) / canonical["width"], (height - 48) / canonical["height"])
        cw, ch = canonical["width"] * scale, canonical["height"] * scale
        ox, oy = (width - cw) / 2, (height - ch) / 2
        rect = np.float32([[ox, oy], [ox + cw, oy], [ox + cw, oy + ch], [ox, oy + ch]])
        to_canvas = cv2.getPerspectiveTransform(quad, rect)
        original = cv2.warpPerspective(output, to_canvas, (width, height), flags=cv2.INTER_LINEAR)
        u, v = observation.get("local_uv", [.5, .5])
        # The thermal adapter's obstruction occupies .32 x .38 of the same
        # target plane. Heat-only symptoms must never invent RGB scorching.
        bw, bh = (cw * .32, ch * .38) if fault == "SurfaceObstruction" else (cw * .8, ch * .8)
        cx, cy = ox + cw * float(u), oy + ch * float(v)
        x1 = max(ox, min(cx - bw / 2, ox + cw - bw))
        y1 = max(oy, min(cy - bh / 2, oy + ch - bh))
        target = {"id": f"{module['module_id']}:{observation['observation_id']}",
                  "x1": int(x1), "y1": int(y1), "x2": int(x1 + bw), "y2": int(y1 + bh)}
        painted = ns["draw_realistic_fault"](Image.fromarray(original), target, fault, int(observation.get("severity", 7)), "rgb")
        delta = np.asarray(painted, dtype=np.float32) - original.astype(np.float32)
        low = np.maximum(0, np.floor(quad.min(0)).astype(int) - 2)
        high = np.minimum(image.size, np.ceil(quad.max(0)).astype(int) + 3)
        bounds = tuple(map(int, (*low, *high)))
        inverse = np.linalg.inv(to_canvas)
        inverse[:2] -= low[:, None] * inverse[2]
        rw, rh = int(high[0] - low[0]), int(high[1] - low[1])
        residual = cv2.warpPerspective(delta, inverse, (rw, rh), flags=cv2.INTER_LINEAR)
        support = ns["real_farm_module_mask"](image.size, [module], active=True, bounds=bounds) > 0
        region = output[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        before = region.copy()
        region[support] = np.uint8(np.clip(region[support].astype(np.float32) + residual[support], 0, 255))
        pixels = int(np.count_nonzero(np.any(before != region, axis=2)))
        if not pixels:
            report["unavailable"].append({"observation_id": observation["observation_id"], "fault_type": fault,
                "reason": "The existing RGB effect produced no visible change at this source resolution"})
            continue
        report["rendered"].append({"observation_id": observation["observation_id"], "module_id": module["module_id"],
            "fault_type": fault, "severity": int(observation.get("severity", 7)), "changed_pixels": pixels,
            "canonical_changed_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
            "engine": "draw_realistic_fault(view='rgb')", "geometry_verified": False})
    result = Image.fromarray(output, "RGB")
    ns["st"].session_state["real_farm_rgb_preview_cache"] = (signature, result, report)
    return result, report
