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
        polygon = (np.float32(target["active_module_polygon"]) * scale).tolist()
        modules.append({"module_id": mid, "active_polygon_rgb": polygon, "outer_polygon_rgb": polygon,
                        "refinement_status": "accepted", "mask_source": "MANUAL_FINAL_POLYGON"})
        observations.append({"observation_id": record["target_cell_id"], "module_id": mid,
            "severity": record["fault_scale"], "diagnostic_fault_type": record["fault_type"],
            "local_uv": [(int(target["module_cell_col"]) - .5) / ns["CUSTOM_MODULE_CELL_COLS"],
                         (int(target["module_cell_row"]) - .5) / ns["CUSTOM_MODULE_CELL_ROWS"]]})
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


def generated_controls(ns, location, row, cell):
    st = ns["st"]
    st.subheader("Visible Far conditions · RGB + thermal")
    modules = {}
    for actual_row, actual_cell in ns["fault_target_cells"](location["id"]):
        key = (actual_row["id"], actual_cell["panel_number"], actual_cell["module_number_in_panel"])
        modules.setdefault(key, actual_cell)
    current = ns["cell_by_id"](location["id"], row, cell)
    current_key = (row, current["panel_number"], current["module_number_in_panel"])
    target = st.selectbox("Visible condition module", list(modules), index=list(modules).index(current_key),
        format_func=lambda k: f"Row {ns['row_number_from_id'](location['id'], k[0])} · Array {k[1]} · Module {k[2]}", key="pv_far_generated_module")
    row, cell = target[0], modules[target]["id"]
    fault = st.selectbox("Visible Far condition", list(FAR_SURFACES), format_func=FAR_SURFACES.get, key="pv_far_generated_type")
    severity = st.slider("Visible condition severity", 1, 10, 7, key="pv_far_generated_severity")
    if st.button("GENERATE VISIBLE FAR CONDITION", type="primary", use_container_width=True):
        try:
            additions = generated_surface_records(ns, location, fault, severity, row, cell)
            occupied = {(r["row_id"], r["target_cell_id"]) for r in ns["records_for_location"](location["id"], "thermal")}
            if any((r["row_id"], r["target_cell_id"]) in occupied for r in additions):
                raise ValueError("This module already has a fault on the intended surface. Clear it or choose another module first.")
            # Render both modalities before accepting any new state.
            proposed = ns["records_for_location"](location["id"], "thermal") + additions
            base_rgb = ns["module_field_image"](location["id"], "rgb", row, cell, [], False)
            rgb, _ = generated_rgb_scene(ns, base_rgb, location["id"], row, cell, proposed)
            thermal = ns["module_field_image"](location["id"], "thermal", row, cell, proposed, False)
            baseline = ns["module_field_image"](location["id"], "thermal", row, cell, proposed[:-len(additions)], False)
            previous_rgb, _ = generated_rgb_scene(ns, base_rgb, location["id"], row, cell, proposed[:-len(additions)])
            checks = {"RGB": visibility(previous_rgb, rgb), "Thermal": visibility(baseline, thermal)}
            ns["push_fault_history"]()
            st.session_state["random_fault_records"] = st.session_state.get("random_fault_records", []) + additions
            st.session_state["pv_far_surface_visibility"] = checks
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if checks := st.session_state.get("pv_far_surface_visibility"):
        with st.expander("Visibility diagnostics",expanded=False):st.json(checks)


def real_controls(ns, state, image, selected, layers):
    st = ns["st"]
    st.subheader("Visible Far conditions · RGB + thermal")
    fault = st.selectbox("Visible Far condition", list(FAR_SURFACES), format_func=FAR_SURFACES.get, key="pv_far_real_type")
    severity = st.slider("Visible condition severity", 1, 10, 7, key="pv_far_real_severity")
    if st.button("GENERATE VISIBLE FAR CONDITION", type="primary", use_container_width=True, disabled=selected is None):
        try:
            trial = deepcopy(state)
            if any(o["module_id"] == selected["module_id"] for o in trial.get("camera_observations", [])):
                raise ValueError("This module already has an observation. Clear it or select another module first.")
            before = deepcopy(trial.get("camera_observations", []))
            observation = add_real_surface(ns, trial, image, selected, fault, severity)
            active_layers = layers if "master_scalar" in layers else ns["real_farm_camera_build_layers"](trial, image)
            old_rgb, _ = rgb_fault_preview(ns, image, active_layers["accepted_modules"], before)
            rgb, report = rgb_fault_preview(ns, image, active_layers["accepted_modules"], trial["camera_observations"])
            if observation["observation_id"] not in {r["observation_id"] for r in report["rendered"]}:
                raise ValueError("The selected module cannot render this RGB surface condition")
            healthy = ns["real_farm_camera_acquisition"](active_layers, before, "far")
            thermal = ns["real_farm_camera_acquisition"](active_layers, trial["camera_observations"], "far")
            checks = {"RGB": visibility(old_rgb, rgb), "Thermal": visibility(healthy, thermal)}
            state["camera_observations"] = trial["camera_observations"]
            state["camera_select_observation_queue"] = observation["observation_id"]
            state["updated_at"] = ns["real_farm_now"]()
            state["far_surface_visibility"] = {"observation_id": observation["observation_id"], **checks}
            if "master_scalar" not in layers:
                frame = state.get("camera_preparation_response", {}).get("physical_frame_width_fraction")
                revision = "real-native-source-surfaces-v24-edges-v1-gap-v1-table-v2" + (f"-frame-width-{float(frame):.6f}" if frame is not None else "")
                st.session_state["real_farm_camera_layers"] = ((ns["real_farm_source_identity"](image),
                    ns["real_farm_geometry_signature"](state["modules"]), revision), active_layers)
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
