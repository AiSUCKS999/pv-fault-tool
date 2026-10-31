"""Visible Far conditions, using existing geometry and approved fault functions."""
from copy import deepcopy
import uuid

import numpy as np
from PIL import Image

from pv_real_farm_rgb_view import rgb_fault_preview

FAR_SURFACES = {"SurfaceObstruction": "Visible surface obstruction", "SoilingDust": "Visible dust / soiling"}


def validate_condition(fault, severity):
    if fault not in FAR_SURFACES or isinstance(severity, bool) or not 1 <= int(severity) <= 10:
        raise ValueError("Choose a supported visible Far condition and severity 1–10")


def generated_rgb_scene(ns, scene, location, row, cell, records):
    """Project only surface evidence through the exact cell polygons used by thermal."""
    visible = [r for r in ns["thermal_records_for_distance"](records, "far")
               if r["fault_type"] in {"SurfaceObstruction", "SoilingDust", "CellCracking"}]
    if not visible:
        return scene, {"rendered": [], "unavailable": []}
    # Resolve source geometry at native resolution, then scale every target
    # polygon to this RGB acquisition. Never infer targets from altered pixels.
    master = ns["cached_native_thermal_master"](location, ns["solar_render_cache_signature"](location))
    if master is None:
        return scene, {"rendered": [], "unavailable": [{"reason": "Exact source geometry unavailable"}]}
    scalar, geometry = master
    scale = np.float32([scene.width / scalar.shape[1], scene.height / scalar.shape[0]])
    modules, observations = [], []
    for record in visible:
        target = ns["rendered_fault_cell_from_record"](location, record, row, cell, geometry)
        if target is None:
            raise ValueError("Visible RGB target is absent from the actual layout")
        mid = f"{location}:{record['row_id']}:{record['target_cell_id']}"
        # A single selected condition belongs to a physical module. The old
        # adapter passed a cell quad and then applied module-local UV again.
        # Legacy explicit multi-cell surface records retain their cell support.
        broad = bool(record.get("far_surface_group_id"))
        logical = ns["cell_by_id"](location, record["row_id"], record["target_cell_id"])
        module = ns["array_module_polygons"](geometry[(ns["row_number_from_id"](location, record["row_id"]), logical["panel_number"])])[logical["module_number_in_panel"]]
        polygon = (np.float32(target["active_module_polygon"] if broad else ns["module_active_polygon"](module)) * scale).tolist()
        modules.append({"module_id": mid, "active_polygon_rgb": polygon, "outer_polygon_rgb": polygon,
                        "refinement_status": "accepted", "mask_source": "MANUAL_FINAL_POLYGON"})
        observation = {"observation_id": record.get("experiment_fault_id") or record["target_cell_id"], "module_id": mid,
            "severity": record["fault_scale"], "diagnostic_fault_type": record["fault_type"],
            "local_uv": [(int(target["module_cell_col"]) - .5) / ns["CUSTOM_MODULE_CELL_COLS"],
                         (int(target["module_cell_row"]) - .5) / ns["CUSTOM_MODULE_CELL_ROWS"]]}
        if record["fault_type"] == "CellCracking" and not broad:
            extent = .45 + .045 * int(record["fault_scale"])
            observation["patch_extent_uv"] = [extent / ns["CUSTOM_MODULE_CELL_COLS"], extent / ns["CUSTOM_MODULE_CELL_ROWS"]]
        observations.append(observation)
    return rgb_fault_preview(ns, scene, modules, observations)


def generated_surface_records(ns, location, fault, severity, selected_row, selected_cell):
    """One broad condition comprises honest existing cell-level engine records."""
    validate_condition(fault, severity)
    if (location["id"] == "rooftop_warehouse" and fault == "SoilingDust"
            and ns["custom_row_count"](location["id"]) == 6 and ns["custom_panels_per_row"](location["id"]) == 6):
        raise ValueError("Known approved-main rooftop 6×6 SoilingDust failure: this condition remains blocked")
    target = ns["cell_by_id"](location["id"], selected_row, selected_cell)
    cells = [c for c in ns["cells_for_row"](location["id"], selected_row)
             if c["panel_number"] == target["panel_number"] and c["module_number_in_panel"] == target["module_number_in_panel"]]
    if len(cells) != ns["CUSTOM_MODULE_CELL_COLS"] * ns["CUSTOM_MODULE_CELL_ROWS"]:
        raise ValueError("Exact physical module cells are unavailable; no substitute scope was applied")
    if fault == "SurfaceObstruction":
        cells = [c for c in cells if 2 <= c["module_cell_col"] <= 4 and 3 <= c["module_cell_row"] <= 6]
    group = "surface-" + uuid.uuid4().hex
    records = [ns["fault_record"](location, "thermal", ns["row_by_id"](location["id"], selected_row), c, fault, int(severity)) for c in cells]
    for record in records:
        record.update(far_surface_group_id=group, far_surface_scope="visible module surface; explicit cell-level engine contributions")
    return records


def add_real_surface(ns, state, image, module, fault, severity):
    validate_condition(fault, severity)
    credible = ns["real_farm_camera_candidates"](state, image)[1]
    if module is None or module["module_id"] not in {m["module_id"] for m in credible}:
        raise ValueError("Select a resolved physical module; provisional points cannot host a visible surface condition")
    observation = ns["real_farm_camera_new_observation"](state, image, module, int(severity))
    observation.update(diagnostic_fault_type=fault, far_surface_condition=True)
    state["updated_at"] = ns["real_farm_now"]()
    return observation


def visibility(healthy, faulted, display_width=1024):
    """Pixel evidence at a bounded Far display size, not a physical-detection claim."""
    ratio = min(1., display_width / healthy.width)
    size = (max(1, round(healthy.width * ratio)), max(1, round(healthy.height * ratio)))
    a = np.asarray(healthy.convert("RGB").resize(size, Image.Resampling.BOX), dtype=np.int16)
    b = np.asarray(faulted.convert("RGB").resize(size, Image.Resampling.BOX), dtype=np.int16)
    changed = np.max(abs(a - b), axis=2)
    pixels = int(np.count_nonzero(changed >= 8))
    return {"display_size": list(size), "contrast_pixels": pixels, "visible_at_tested_scale": pixels >= 64,
            "criterion": "at least 64 display pixels differ by 8/255; simulator visibility only"}
