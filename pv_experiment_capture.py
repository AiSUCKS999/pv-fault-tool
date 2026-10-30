"""Read actual state and call approved image/export functions. No substitute rendering."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from pv_experiment_log import Journal, fingerprint


@lru_cache(maxsize=128)
def _file_hash(path, size, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "sha256": _file_hash(str(path), stat.st_size, stat.st_mtime_ns)}


def renderer_identity(ns):
    return {name: file_identity(ns["ROOT"] / name)["sha256"] for name in
            ("pv_render_thermal.py", "pv_thermal_realism.py", "pv_real_farm_camera.py", "pv_render_rgb.py")}


def _cell(ns, location, row, cid):
    result = next((c for c in ns["cells_for_row"](location, row) if c["id"] == cid), None)
    if result is None:
        raise ValueError("The actual selected cell is not present in this layout")
    return result


def _target(ns, location, row, cell):
    return {"row_id": row, "row": int(ns["row_number_from_id"](location, row)),
            "array": int(cell["panel_number"]), "module": int(cell["module_number_in_panel"]),
            "cell": int(cell["module_cell_number"]), "cell_id": cell["id"],
            "bbox_px": [float(cell[k]) for k in ("x1", "y1", "x2", "y2")],
            "center_px": [float(cell["x1"] + cell["x2"]) / 2, float(cell["y1"] + cell["y2"]) / 2]}


def actual_state(ns, mode):
    """Never populate actual fields from a scenario plan or preparation receipt."""
    st = ns["st"]
    if st.session_state.get("thermal_renderer_mode", "Current Thermal") != "Current Thermal":
        raise ValueError("Use Current Thermal before saving experiment references")
    if mode == "Generated":
        loc = ns["selected_location"]()
        source = ns["solar_render_asset_path"](loc["id"])
        if source is None:
            raise ValueError("The actual Generated render asset is unavailable")
        row = ns["qp_get"]("selected_pv", ns["panels"](loc["id"])[0]["id"])
        cid = ns["qp_get"]("selected_cell", ns["cells_for_row"](loc["id"], row)[0]["id"])
        target = _target(ns, loc["id"], row, _cell(ns, loc["id"], row, cid))
        faults = []
        effective = ns["thermal_records_for_distance"](ns["records_for_location"](loc["id"], "thermal"), "far")
        for record in effective:
            cell = _cell(ns, loc["id"], record["row_id"], record["target_cell_id"])
            item = {"fault_type": ns["canonical_fault_type"](record["fault_type"]), "severity": int(record["fault_scale"]),
                    "target": _target(ns, loc["id"], record["row_id"], cell), "engine_observation_id": record.get("observation_id"),
                    "applied_seed": None, "seed_behavior": "Fixed-target engine derives morphology internally; independent case seed is unsupported"}
            item["record_identity"] = "generated-" + fingerprint(item)[:20]
            item["identity_kind"] = "collection-derived identity of the actual engine record; not a native observation ID"
            faults.append(item)
        return {"mode": mode, "scene_identity": loc["id"], "scene_name": loc["name"], "source_image": file_identity(source),
                "rows": ns["custom_row_count"](loc["id"]), "columns": ns["custom_panels_per_row"](loc["id"]),
                "selection": target, "faults": faults, "renderer": "Current Thermal", "renderer_sources": renderer_identity(ns)}
    state, image = st.session_state.get("real_farm_state"), st.session_state.get("real_farm_image")
    if not state or image is None:
        raise ValueError("Load the actual Real Farm photograph first")
    source_hash = ns["real_farm_source_identity"](image)
    if source_hash != state.get("source_rgb_pixel_sha256"):
        raise ValueError("Real Farm state belongs to a different source image")
    modules = {m["module_id"]: m for m in ns["real_farm_recovered_modules"](
        state.get("modules", []), include_unaccepted=True, image_size=image.size)}
    faults = []
    for observation in state.get("camera_observations", []):
        if observation.get("source_rgb_pixel_sha256") != source_hash:
            raise ValueError("An observation belongs to a different photograph")
        module = modules.get(observation["module_id"]) or observation.get("target_geometry")
        if not module:
            raise ValueError("An observation has no matching physical target geometry")
        faults.append({"engine_observation_id": observation["observation_id"], "record_identity": observation["observation_id"],
            "identity_kind": "native simulator observation ID", "module_id": observation["module_id"],
            "fault_type": observation.get("diagnostic_fault_type") or observation.get("observation_type", "SingleHotSpot"),
            "severity": int(observation["severity"]), "applied_seed": observation.get("seed"),
            "seed_behavior": "Existing seeded observation placement; existing morphology engine unchanged",
            "center_rgb_px": deepcopy(observation.get("center_rgb_px")), "local_uv": deepcopy(observation.get("local_uv")),
            "polygon_rgb": deepcopy(module["active_polygon_rgb"]),
            "target_provisional": bool(module.get("simulation_only") or module.get("physical_module_unresolved")),
            "engine_footprint_native_pixels": observation.get("footprint_rgb", {}).get("native_pixels"),
            "reference_ids": deepcopy(observation.get("production_reference_ids", {}))})
    selected = next((o for o in faults if o["record_identity"] == state.get("camera_selected_observation")), faults[0] if faults else None)
    return {"mode": mode, "scene_identity": state.get("prepared_catalog_id", "uploaded") + "-" + source_hash[:12],
            "scene_name": state.get("original_rgb_filename", "Real Farm photograph"),
            "source_image": {"name": Path(st.session_state.get("real_farm_image_path", "source")).name,
                             "pixel_sha256": source_hash, "size": list(image.size)},
            "geometry_signature": ns["real_farm_geometry_signature"](state["modules"]),
            "selection": {"module_id": selected["module_id"]} if selected else
                ({"module_id": st.session_state["real_farm_camera_target"]} if st.session_state.get("real_farm_camera_target") in modules else {}), "faults": faults,
            "renderer": "Current Thermal", "renderer_sources": renderer_identity(ns)}


def assessment(case, actual, view):
    expected = case["CLOSE diagnosis code"] if view == "CLOSE" and case["Scenario stage"] == "Close diagnosis" else case["FAR observable code"]
    faults = actual["faults"]
    errors = []
    if expected == "Healthy":
        if faults:
            errors.append("The actual scene contains faults; the plan is healthy")
    else:
        if expected not in {"Healthy", "Multiple SingleHotSpot observations"} and len(faults) != 1:
            errors.append("A single target observation is required")
        code = "SingleHotSpot" if expected == "Multiple SingleHotSpot observations" else expected
        if not faults or any(f["fault_type"] != code for f in faults):
            errors.append(f"Expected {expected}; recorded actual engine fault types differ")
        if expected == "Multiple SingleHotSpot observations" and len({f.get("module_id") for f in faults}) < 2:
            errors.append("Multiple separate hotspot modules are required")
        if any(f["severity"] != int(case["Severity (1–10)"]) for f in faults):
            errors.append("Actual severity differs from the plan")
    if actual["mode"] == "Generated":
        if (actual["rows"], actual["columns"]) != (int(case["Rows"]), int(case["Columns / row"])):
            errors.append("Actual layout differs from the plan")
        expected_target = tuple(int(case[k]) for k in ("Target row", "Target array col", "Module slot (1–4)", "Cell index (1–40)"))
        targets = [actual["selection"]] + [f["target"] for f in faults]
        if any(tuple(t[k] for k in ("row", "array", "module", "cell")) != expected_target for t in targets):
            errors.append("Actual target differs from the plan")
    else:
        if any(f["target_provisional"] for f in faults):
            errors.append("Target geometry remains provisional")
        if any(f["applied_seed"] != int(case["Seed"]) for f in faults):
            errors.append("Actual observation seed differs from the plan")
    return {"configuration_matches_plan": not errors, "differences": errors,
            "synthetic_only": True, "physical_ground_truth_verified": False, "yolo_detection_verified": False}


def _real_layers(ns):
    st = ns["st"]
    state, image = st.session_state["real_farm_state"], st.session_state["real_farm_image"]
    source = ns["real_farm_source_identity"](image)
    geometry = ns["real_farm_geometry_signature"](state["modules"])
    cached = st.session_state.get("real_farm_camera_layers")
    if cached and cached[0][:2] == (source, geometry):
        return cached[1]
    # Explicit user preparation/reference action only. No detector/recovery, new mask or renderer.
    layers = ns["real_farm_camera_build_layers"](state, image)
    if ns["real_farm_geometry_signature"](state["modules"]) != geometry:
        raise ValueError("Existing thermal generation unexpectedly changed geometry")
    frame = state.get("camera_preparation_response", {}).get("physical_frame_width_fraction")
    revision = "real-native-source-surfaces-v24-edges-v1-gap-v1-table-v2" + (f"-frame-width-{float(frame):.6f}" if frame is not None else "")
    st.session_state["real_farm_camera_layers"] = ((source, geometry, revision), layers)
    return layers


def render_reference(ns, mode, view):
    actual = actual_state(ns, mode)
    close = view == "CLOSE"
    bounds = None
    if mode == "Generated":
        loc, target = actual["scene_identity"], actual["selection"]
        row, cid = target["row_id"], target["cell_id"]
        records = ns["records_for_location"](loc, "thermal")
        full = ns["module_field_image"](loc, "thermal", row, cid, records, False, scene_distance="close" if close else "far").convert("RGB")
        healthy = ns["module_field_image"](loc, "thermal", row, cid, [], False, scene_distance="close" if close else "far").convert("RGB")
        if close:
            rgb = ns["module_field_image"](loc, "rgb", row, cid, [], False).convert("RGB")
            region = next((r for r in ns["array_regions"](loc, rgb) if int(r["row"]) == target["row"] and int(r["col"]) == target["array"]), None)
            if region is None:
                raise ValueError("Actual selected array cannot be resolved for CLOSE export")
            native = ns["close_source_scene"](loc, "thermal", row, cid, records, full)
            healthy_native = ns["close_source_scene"](loc, "thermal", row, cid, [], healthy)
            full, bounds, _, _ = ns["crop_natural_module_image"](native, rgb.size, region, target["module"], .62)
            healthy, _, _, _ = ns["crop_natural_module_image"](healthy_native, rgb.size, region, target["module"], .62)
    else:
        st = ns["st"]
        layers = _real_layers(ns)
        state, source = st.session_state["real_farm_state"], st.session_state["real_farm_image"]
        observations = state.get("camera_observations", [])
        target_ids = {m["module_id"] for m in layers["accepted_modules"]}
        if any(o["module_id"] not in target_ids for o in observations):
            raise ValueError("An actual observation has no supported target in this thermal acquisition")
        if close:
            mid = actual["selection"].get("module_id")
            selected = next((m for m in layers["accepted_modules"] if m["module_id"] == mid), None)
            if selected is None:
                if not observations:
                    raise ValueError("Select a real physical module for the optional healthy CLOSE view")
                raise ValueError("Actual target is absent from the thermal acquisition geometry")
            bounds = ns["real_farm_close_context_bounds"](selected, layers.get("camera_candidates", layers["accepted_modules"]),
                    (source.height, source.width), .25)
        healthy = ns["real_farm_camera_acquisition"](layers, [], "close" if close else "far", bounds)
        full = ns["real_farm_camera_acquisition"](layers, observations, "close" if close else "far", bounds)
        if observations and any(o.get("footprint_rgb", {}).get("native_pixels", 0) <= 0 for o in observations):
            raise ValueError("An actual observation has no rendered native footprint")
    if full.size != healthy.size:
        raise ValueError("Reference/baseline image size mismatch")
    delta = int(np.count_nonzero(np.any(np.asarray(full.convert("RGB")) != np.asarray(healthy.convert("RGB")), axis=2)))
    if actual["faults"] and not delta:
        raise ValueError("The existing renderer produced no visible fault change; export is blocked")
    png = ns["export_current_view_bytes"](full)
    # Production reference IDs can be pinned while the existing renderer runs.
    actual = actual_state(ns, mode)
    return png, actual, {"view": view, "distance": "close" if close else "far", "crop_bounds": list(bounds) if bounds else None,
                          "overlays": False, "rendered_changed_pixels": delta,
                          "existing_export_function": "export_current_view_bytes",
                          "existing_renderer": "real_farm_camera_acquisition" if mode == "Real Farm" else "module_field_image / close_source_scene"}


def stay_collection(ns, mode):
    if mode == "Generated":
        ns["st"].query_params["section"] = "DATA COLLECTION"
    else:
        ns["st"].session_state["real_farm_section"] = "DATA COLLECTION"


def ensure_reference(ns, journal, exp, view, force=False):
    if ns["st"].session_state.get("pv_dc_live_experiment") != exp["experiment_id"]:
        raise ValueError("Prepare this test's stage before saving a reference; the simulator is on another context")
    actual = actual_state(ns, exp["mode"])
    far = exp["stages"]["FAR"]["preparations"]
    if far and actual["scene_identity"] != far[0]["actual"]["scene_identity"]:
        raise ValueError("The simulator source changed. Start a new preparation for that scene before saving references.")
    ref = journal.current_reference(exp, view)
    if not force and ref and ref["state_fingerprint"] == fingerprint(actual) and ref["reference_id"] in journal.blobs:
        return ref
    png, actual, parameters = render_reference(ns, exp["mode"], view)
    checked = assessment(exp["plan"], actual, view)
    if view == "CLOSE" and far:
        before = far[-1]["actual"]
        if actual.get("geometry_signature") != before.get("geometry_signature"):
            checked["differences"].append("Geometry changed between FAR and CLOSE")
        if actual["mode"] == "Real Farm" and {f["module_id"] for f in actual["faults"]} != {f["module_id"] for f in before["faults"]}:
            checked["differences"].append("Target modules changed between FAR and CLOSE")
        checked["configuration_matches_plan"] = not checked["differences"]
    checked["rendered_synthetic_effect_verified"] = bool(not actual["faults"] or parameters["rendered_changed_pixels"] > 0)
    journal.prepared(exp, view, actual, checked)
    return journal.reference(exp, view, actual, png, parameters)


def prepare(ns, journal, case, options, view="FAR", exp=None):
    from pv_data_collection_prepare import prepare_generated, prepare_real, real_targets
    key = "scene" if case["Source mode"] == "Generated" else "source"
    changed_source = exp and view == "FAR" and exp.get("setup_options") and exp["setup_options"].get(key) != options.get(key)
    if exp is None or exp.get("finished_utc") or changed_source:
        exp = journal.start(case)
    journal.data["active_by_mode"][case["Source mode"]] = exp["experiment_id"]
    settings = deepcopy(options)
    settings["phase"] = "Close diagnosis" if view == "CLOSE" and case["Scenario stage"] == "Close diagnosis" else "Far observation"
    try:
        if view == "CLOSE":
            far = exp["stages"]["FAR"]["preparations"]
            if not far:
                raise ValueError("Prepare FAR first")
            bound = far[-1]["actual"]
            if case["Source mode"] == "Generated":
                settings["scene"] = bound["scene_identity"]
            else:
                settings["source"] = exp["setup_options"]["source"]
                settings["modules"] = [f["module_id"] for f in bound["faults"]]
        if case["Source mode"] == "Real Farm" and not settings.get("modules") and case["Scenario stage"] != "Healthy":
            targets = real_targets(ns, settings.get("source")) if settings.get("source") else []
            count = 3 if case["FAR observable code"] == "Multiple SingleHotSpot observations" else 1
            if len(targets) < count:
                raise ValueError("Not enough credible physical modules. Select/confirm targets in the existing simulator.")
            import random
            settings["modules"] = random.Random(int(case["Seed"])).sample(sorted(targets), count)
        cached = ns["st"].session_state.get("real_farm_camera_layers")
        result = (prepare_generated if case["Source mode"] == "Generated" else prepare_real)(ns, case, settings)
        if cached and case["Source mode"] == "Real Farm":
            state = ns["st"].session_state["real_farm_state"]
            if cached[0][:2] == (state["source_rgb_pixel_sha256"], ns["real_farm_geometry_signature"](state["modules"])):
                ns["st"].session_state["real_farm_camera_layers"] = cached
        exp["setup_options"] = settings if view == "FAR" else exp["setup_options"]
        exp["preparation_result"] = {"status": result["status"], "identity": result["identity"]}
        ns["st"].session_state["pv_dc_live_experiment"] = exp["experiment_id"]
        actual = actual_state(ns, case["Source mode"])
        checked = assessment(case, actual, view)
        checked["preparation_engine_evidence"] = deepcopy(result["details"])
        journal.prepared(exp, view, actual, checked)
    except Exception as exc:
        journal.fail(exp, "preparation/reference", str(exc))
    finally:
        stay_collection(ns, case["Source mode"])
    return exp


def collect_automatically(ns, journal, case, options):
    """One foreground action collects simulator evidence; no external recording assertion."""
    exp = prepare(ns, journal, case, options, "FAR", journal.active(case["Source mode"], case["Case ID"]))
    if exp["blocked_reason"]:
        return exp
    try:
        ensure_reference(ns, journal, exp, "FAR", force=True)
        if case["CLOSE required?"] == "Yes":
            prepare(ns, journal, case, options, "CLOSE", exp)
            if exp["blocked_reason"]:
                return exp
            ensure_reference(ns, journal, exp, "CLOSE", force=True)
        journal.finish_simulator(exp)
    except Exception as exc:
        journal.fail(exp, "automatic simulator collection", exc)
    finally:
        stay_collection(ns, case["Source mode"])
    return exp
