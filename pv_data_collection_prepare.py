"""Collection adapters over existing simulator APIs; no renderer or geometry edits."""
from copy import deepcopy

from pv_data_collection import generated_preparation_issue


def snapshot(ns):
    st = ns["st"]
    state = {k: v for k, v in st.session_state.items() if not k.startswith("pv_dc_")}
    for key in ("real_farm_state", "random_fault_records", "panel_cell_fault_records", "fault_undo_stack", "fault_redo_stack"):
        if key in state:
            state[key] = deepcopy(state[key])
    return {"state": state, "query": dict(st.query_params)}


def restore(ns, saved):
    st = ns["st"]
    for key in list(st.session_state):
        if not key.startswith("pv_dc_"):
            del st.session_state[key]
    st.session_state.update(saved["state"])
    ns["qp_set"](**saved["query"])


def restore_previous(ns, original=False):
    saved = ns["st"].session_state.pop("pv_dc_original_simulator" if original else "pv_dc_before_preparation", None)
    if saved:
        restore(ns, saved)
        ns["st"].session_state.pop("pv_dc_active_case", None)
        if original:
            ns["st"].session_state.pop("pv_dc_before_preparation", None)


def transaction(ns, action):
    saved = snapshot(ns)
    try:
        result = action()
    except Exception:
        restore(ns, saved)
        raise
    ns["st"].session_state.setdefault("pv_dc_original_simulator", saved)
    ns["st"].session_state["pv_dc_before_preparation"] = saved
    return result


def generated_block(case, scene):
    issue = generated_preparation_issue(case)
    if issue:
        return issue
    # The clean-main regression fails within this layout/fault combination.
    # Conservatively hold this combination; do not patch the legacy engine.
    if (scene == "rooftop_warehouse" and case["Rows"] == "6" and case["Columns / row"] == "6"
            and case["CLOSE diagnosis code"] == "SoilingDust"):
        return "Known approved-main rooftop 6×6 SoilingDust regression. This combination is blocked pending a separate engine review."
    return None


def prepare_generated(ns, case, options):
    def action():
        st = ns["st"]
        scene = options["scene"]
        location = next(loc for loc in ns["LOCATIONS"] if loc["id"] == scene)
        if issue := generated_block(case, scene):
            raise ValueError(issue)
        rows, cols = int(case["Rows"]), int(case["Columns / row"])
        asset = ns["ASSET_DIR"] / "solar-renders" / scene / f"{rows}x{cols}.webp"
        legacy = ns["ASSET_DIR"] / "solar-renders" / f"{rows}x{cols}.webp"
        if not asset.is_file() and not (scene == "grass_open" and legacy.is_file()):
            raise ValueError("Exact approved render unavailable. No substitute layout was opened.")
        st.session_state["thermal_renderer_mode"] = "Current Thermal"
        ns["qp_set"](location=scene, custom_layout=1, row_count=rows, panels_per_row=cols, section="DATA COLLECTION")
        row, col = int(case["Target row"]), int(case["Target array col"])
        module, index = int(case["Module slot (1–4)"]), int(case["Cell index (1–40)"])
        row_id, cell = ns["default_cell_for_array"](scene, row, col, module, index)
        if (ns["row_number_from_id"](scene, row_id), cell["panel_number"], cell["module_number_in_panel"], cell["module_cell_number"]) != (row, col, module, index):
            raise ValueError("Existing target lookup did not resolve the exact planned cell.")
        phase = options.get("phase", case["Scenario stage"])
        fault = case["CLOSE diagnosis code"] if phase == "Close diagnosis" else case["FAR observable code"]
        records = []
        severity, seed = int(case["Severity (1–10)"] or 1), int(case["Seed"])
        if case["Scenario stage"] != "Healthy":
            if fault not in ns["FAULT_TYPES"]:
                raise ValueError("This fault code has no supported Generated engine mapping.")
            record = ns["fault_record"](location, "thermal", ns["row_by_id"](scene, row_id), cell, fault, severity)
            records = [record]
        # Test actual production rendering before committing simulator state.
        healthy = ns["module_field_image"](scene, "thermal", row_id, cell["id"], [], False, scene_distance="close" if phase == "Close diagnosis" else "far")
        if records:
            rendered = ns["module_field_image"](scene, "thermal", row_id, cell["id"], records, False, scene_distance="close" if phase == "Close diagnosis" else "far")
            import numpy as np
            changed = int(np.count_nonzero(np.any(np.asarray(healthy) != np.asarray(rendered), axis=2)))
            if not changed:
                raise ValueError("The existing engine produced no visible fault delta. Preparation blocked.")
        else:
            changed = 0
        ns["push_fault_history"]()
        st.session_state.update(random_fault_records=records if phase == "Far observation" else [],
                                panel_cell_fault_records=records if phase == "Close diagnosis" else [],
                                thermal_renderer_mode="Current Thermal", generated_fault_severity=severity,
                                modern_random_fault_seed=seed)
        if phase == "Close diagnosis":
            st.session_state["new_fault_type"] = fault
        st.session_state["pv_dc_active_case"] = case
        ns["qp_set"](location=scene, view="thermal", thermal_renderer="current", section="Faults", custom_layout=1,
                     row_count=rows, panels_per_row=cols, selected_pv=row_id, selected_cell=cell["id"],
                     selected_array=f"R{row}-A{col:02d}" if phase == "Close diagnosis" else "",
                     selected_module=module if phase == "Close diagnosis" else "", cell_active=int(phase == "Close diagnosis"))
        detail = {"scene": scene, "rows": rows, "columns": cols, "row": row, "array": col, "module": module,
                  "cell": index, "target_cell_id": cell["id"], "fault": fault if records else "Healthy",
                  "severity": severity if records else None, "requested_seed": seed, "changed_pixels": changed,
                  "seed_behavior": "Existing fixed-target renderer derives morphology from target/type/severity; independent case seed is not supported. Recorded only."}
        return {"identity": f"Generated:{scene}:{rows}x{cols}", "status": "Guided" if records else "Prepared",
                "phase": phase, "details": detail}
    return transaction(ns, action)


def real_source(ns, source_id):
    entry = next(e for e in ns["real_farm_catalog_entries"]() if e["id"] == source_id)
    path = ns["real_farm_catalog_asset"](entry["rgb"], entry["rgb_file_sha256"])
    image = ns["ImageOps"].exif_transpose(ns["Image"].open(path)).convert("RGB")
    if ns["real_farm_source_identity"](image) != entry["rgb_pixel_sha256"]:
        raise ValueError("Prepared source identity mismatch")
    live = ns["st"].session_state.get("real_farm_state", {})
    if live.get("source_rgb_pixel_sha256") == entry["rgb_pixel_sha256"]:
        state = deepcopy(live)  # Retain current edited geometry; never reload over it.
    else:
        state = ns["real_farm_default_state"](path, image, path.name, entry["rgb_file_sha256"][:16])
        if not ns["real_farm_camera_saved_geometry"](state, image):
            raise ValueError("Existing prepared geometry is unavailable")
    state["prepared_catalog_id"] = source_id
    return state, image, path


def real_targets(ns, source_id):
    state, image, _ = real_source(ns, source_id)
    return [m["module_id"] for m in ns["real_farm_camera_candidates"](state, image)[1]]


def prepare_real(ns, case, options):
    def action():
        source = options.get("source")
        if not source:
            raise ValueError("Choose the actual prepared photograph for this workbook slot.")
        state, image, path = real_source(ns, source)
        signature = ns["real_farm_geometry_signature"](state["modules"])
        phase = options.get("phase", case["Scenario stage"])
        fault = case["CLOSE diagnosis code"] if phase == "Close diagnosis" else case["FAR observable code"]
        ids = options.get("modules", [])
        targets = ns["real_farm_camera_candidates"](state, image)[1] if ids else []
        lookup = {m["module_id"]: m for m in targets}
        if any(mid not in lookup for mid in ids):
            raise ValueError("Selected module lacks currently credible physical PV support.")
        if fault == "Multiple SingleHotSpot observations" and len(ids) < 2:
            ids = []  # Ambiguous count/positions must be chosen explicitly.
        elif fault != "Multiple SingleHotSpot observations" and len(ids) > 1:
            raise ValueError("Select exactly one module for this scenario.")
        severity, seed = int(case["Severity (1–10)"] or 1), int(case["Seed"])
        state["camera_observations"] = []
        observations = state["camera_observations"]
        guided = case["Scenario stage"] != "Healthy" and not ids
        if not guided and case["Scenario stage"] != "Healthy":
            for mid in ids:
                observation = ns["real_farm_add_module_observation"](targets, observations, mid, severity, seed)
                observation.update(diagnostic_fault_type=fault if phase == "Close diagnosis" else None,
                    source_rgb_pixel_sha256=state["source_rgb_pixel_sha256"], target_geometry=deepcopy(lookup[mid]),
                    simulation_only=True, training_verified=False, geometry_status="UNVERIFIED")
                observations.append(observation)
            state["camera_select_observation_queue"] = observations[0]["observation_id"]
        # Validate the existing fault engine on native coordinates, without
        # launching expensive full-scene thermal generation or any recovery.
        changed = 0
        if observations:
            import numpy as np
            field = np.full((image.height, image.width), 150., np.float32)
            for observation in observations:
                before = field.copy()
                ns["real_farm_apply_production_fault"](field, lookup[observation["module_id"]], observation,
                    observation.get("diagnostic_fault_type") or "SingleHotSpot", in_place=True)
                delta = int(np.count_nonzero(field != before))
                if not delta:
                    raise ValueError("The existing fault engine produced no scalar delta for this target.")
                changed += delta
        if ns["real_farm_geometry_signature"](state["modules"]) != signature:
            raise ValueError("Preparation unexpectedly changed geometry; all changes rolled back.")
        state["observations"] = deepcopy(observations)
        st = ns["st"]
        # Replace only the working Real Farm context. The full prior context is recoverable.
        for key in list(st.session_state):
            if key.startswith("real_farm_"):
                del st.session_state[key]
        st.session_state.update(real_farm_image=image, real_farm_image_path=str(path), real_farm_state=state,
            real_farm_prepared_picker=source, real_farm_section="Faults", thermal_renderer_mode="Current Thermal",
            real_farm_camera_ui_source=state["source_rgb_pixel_sha256"],
            real_farm_camera_distance="Close View" if phase == "Close diagnosis" else "Far View")
        st.session_state["pv_dc_active_case"] = case
        detail = {"source": source, "rgb_sha256": state["source_rgb_pixel_sha256"], "geometry": signature,
                  "modules": ids, "fault": "No fault applied — choose valid module(s) manually" if guided else fault,
                  "severity": severity if observations else None, "seed": seed, "observation_ids": [o["observation_id"] for o in observations],
                  "scalar_test_changed_pixels": changed,
                  "next_step": ("Use existing Manual Select on visible PV support and choose/confirm target modules, then return to prepare. No fault was applied. " if guided else "")
                    + "Open Thermal and press Generate Thermal, then inspect FAR/CLOSE. Download camera view and unverified analysis JSON in Diagnostics. Full-scene rendering is explicitly manual; scalar validation is not capture verification.",
                  "geometry_export": "Verified-label exports require independent geometry review and user confirmation.",
                  "seed_behavior": "Existing seeded observation placement; morphology remains the existing engine's behavior."}
        return {"identity": f"Real Farm:{source}:{state['source_rgb_pixel_sha256']}",
                "status": "Guided", "phase": phase, "details": detail}
    return transaction(ns, action)
