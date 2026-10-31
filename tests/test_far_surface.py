"""Far-scale paired surface evidence, truthful heat-only RGB, and UI acceptance."""
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pv_data_collection import _load_cases
from pv_data_collection_prepare import prepare_generated, prepare_real, real_targets
from pv_far_surface import generated_surface_records, generated_rgb_scene, add_real_surface, visibility
from pv_real_farm_rgb_view import rgb_fault_preview


@pytest.fixture(scope="module")
def application():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.fixture
def ns(application):
    application["st"].session_state.clear()
    application["st"].query_params.clear()
    return application


def case(cid):
    return next(c for c in _load_cases() if c["Case ID"] == cid)


@pytest.mark.parametrize("cid,scene,fault,severity", [
    ("TC0001", "grass_open", "SurfaceObstruction", 7),
    ("TC0001", "grass_open", "SoilingDust", 5),
    ("TC0386", "grass_open", "SurfaceObstruction", 9),
    ("TC0386", "desert_track", "SoilingDust", 8),
])
def test_generated_broad_surface_is_visible_in_both_modalities_at_far_scale(ns, cid, scene, fault, severity):
    prepare_generated(ns, case(cid), {"scene": scene})
    row, cell = ns["qp_get"]("selected_pv"), ns["qp_get"]("selected_cell")
    records = generated_surface_records(ns, ns["location_by_id"](scene), fault, severity, row, cell)
    assert len(records) == (40 if fault == "SoilingDust" else 12)
    assert {r["fault_type"] for r in records} == {fault}
    assert {r["fault_scale"] for r in records} == {severity}
    assert len({r["far_surface_group_id"] for r in records}) == 1
    original = ns["module_field_image"](scene, "rgb", row, cell, [], False)
    original_bytes = ns["export_current_view_bytes"](original)
    geometry_before = deepcopy(ns["array_regions"](scene, original))
    rgb, report = generated_rgb_scene(ns, original, scene, row, cell, records)
    assert len(report["rendered"]) == len(records), report["unavailable"]
    thermal = ns["module_field_image"](scene, "thermal", row, cell, records, False)
    baseline = ns["module_field_image"](scene, "thermal", row, cell, [], False)
    assert visibility(original, rgb)["visible_at_tested_scale"]
    assert visibility(baseline, thermal)["visible_at_tested_scale"]
    assert ns["array_regions"](scene, original) == geometry_before
    assert ns["export_current_view_bytes"](original) == original_bytes
    png = ns["export_current_view_bytes"](rgb)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(png))), np.asarray(rgb))
    # Source mask remains unchanged; all new RGB pixels stay on the exact
    # approved engine target support rather than on roofs/ground/gutters.
    master, geometries = ns["cached_native_thermal_master"](scene, ns["solar_render_cache_signature"](scene))
    support = np.zeros((original.height, original.width), np.uint8)
    import cv2
    for record in records:
        target = ns["rendered_fault_cell_from_record"](scene, record, row, cell, geometries)
        p = np.float32(target["active_module_polygon"]) * [original.width/master.shape[1], original.height/master.shape[0]]
        cv2.fillConvexPoly(support, np.rint(p).astype(np.int32), 255)
    changed = np.any(np.array(rgb) != np.array(original), axis=2)
    assert not np.any(changed & (support == 0))
    if cid == "TC0001" and fault == "SoilingDust":
        projected, modules, mask = ns["generated_export_geometry"](scene, rgb, geometry_before, original.size, records)
        package = ns["generated_geometry_dataset"](ns["location_by_id"](scene), "rgb", ns["row_by_id"](scene, row),
            ns["cell_by_id"](scene, row, cell), projected, rgb, original, rgb, (0, 0, rgb.width, rgb.height), modules, mask)
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(archive.read("pv_scene.png")))), np.asarray(rgb))
            assert archive.read("source_rgb.png") == original_bytes
            assert len(json.loads(archive.read("faults.json"))) == 40
        ns["st"].session_state["random_fault_records"] = records
        from pv_experiment_capture import actual_state
        actual = actual_state(ns, "Generated")
        assert {f["far_surface_group_id"] for f in actual["faults"]} == {records[0]["far_surface_group_id"]}


@pytest.mark.parametrize("fault", ["SurfaceObstruction", "SoilingDust"])
def test_real_condition_has_paired_far_evidence_and_preserves_photo_geometry(ns, fault):
    mid = real_targets(ns, "image_05")[0]
    prepare_real(ns, case("TC0496"), {"source": "image_05"})
    state, image = ns["st"].session_state["real_farm_state"], ns["st"].session_state["real_farm_image"]
    modules = ns["real_farm_camera_candidates"](state, image)[1]
    module = next(m for m in modules if m["module_id"] == mid)
    geometry = deepcopy(state["modules"])
    source = np.array(image).copy()
    observation = add_real_surface(ns, state, image, module, fault, 8)
    layers = ns["real_farm_camera_build_layers"](state, image)
    rgb, report = rgb_fault_preview(ns, image, modules, state["camera_observations"])
    thermal = ns["real_farm_camera_acquisition"](layers, state["camera_observations"], "far")
    baseline = ns["real_farm_camera_acquisition"](layers, [], "far")
    assert report["rendered"][0]["observation_id"] == observation["observation_id"]
    assert visibility(image, rgb)["visible_at_tested_scale"]
    assert visibility(baseline, thermal)["visible_at_tested_scale"]
    assert observation["footprint_rgb"]["native_pixels"] > 0
    assert observation["diagnostic_fault_type"] == fault
    assert state["modules"] == geometry and np.array_equal(np.asarray(image), source)
    close = ns["real_farm_close_context_bounds"](module, modules, (image.height, image.width), .25)
    cropped = ns["real_farm_camera_acquisition"](layers, state["camera_observations"], "close", close)
    assert cropped.width < image.width and observation["diagnostic_fault_type"] == fault


def test_visible_far_safety_rejects_unsupported_and_rooftop_baseline(ns):
    prepare_generated(ns, case("TC0386"), {"scene": "rooftop_warehouse"})
    row, cell = ns["qp_get"]("selected_pv"), ns["qp_get"]("selected_cell")
    location = ns["location_by_id"]("rooftop_warehouse")
    before = deepcopy(ns["st"].session_state)
    with pytest.raises(ValueError, match="remains blocked"):
        generated_surface_records(ns, location, "SoilingDust", 7, row, cell)
    assert ns["st"].session_state == before
    with pytest.raises(ValueError, match="supported visible Far"):
        generated_surface_records(ns, location, "SingleHotSpot", 7, row, cell)
    image = Image.new("RGB", (20, 20), (50, 50, 50))
    assert not visibility(image, image)["visible_at_tested_scale"]


@pytest.mark.parametrize("visible,initial_view", [(True, "RGB faults"), (True, "Thermal"), (False, "RGB faults")])
def test_real_far_ui_generates_visible_pair_without_close_diagnosis_or_typing(ns, monkeypatch, visible, initial_view):
    mid = real_targets(ns, "image_05")[0]
    prepare_real(ns, case("TC0496"), {"source": "image_05"})
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    # Exercise the preserved individual controls explicitly; scene workflow is the new default.
    app.session_state["scene_individual_real"] = True
    app.session_state["scene_individual_generated"] = True
    app.session_state["collection_mode_Generated"] = "929 controlled cases"
    app.session_state["collection_mode_Real Farm"] = "929 controlled cases"
    app.query_params["base_source"] = "real_farm"
    for key in ("real_farm_state", "real_farm_image", "real_farm_image_path"):
        app.session_state[key] = deepcopy(ns["st"].session_state[key])
    app.session_state["real_farm_camera_ui_source"] = ns["real_farm_source_identity"](ns["st"].session_state["real_farm_image"])
    app.session_state["real_farm_camera_display"] = initial_view
    app.session_state["real_farm_camera_target"] = mid
    app.session_state["real_farm_section"] = "Faults"
    app.run()
    assert not app.exception
    if not visible:
        import pv_far_surface
        monkeypatch.setattr(pv_far_surface, "visibility", lambda *args: {"visible_at_tested_scale": False})
    next(b for b in app.button if b.label == "GENERATE VISIBLE FAR CONDITION").click().run()
    assert not app.exception
    state = app.session_state["real_farm_state"]
    assert not app.error
    assert len(state["camera_observations"]) == 1
    observation = state["camera_observations"][0]
    assert observation["diagnostic_fault_type"] == "SurfaceObstruction"
    assert state["far_surface_visibility"]["RGB"]["visible_at_tested_scale"] == visible
    assert state["far_surface_visibility"]["Thermal"]["visible_at_tested_scale"] == visible
    app.radio(key="real_farm_camera_display").set_value("Thermal").run()
    assert not app.exception
    assert app.session_state["real_farm_state"]["camera_observations"][0]["observation_id"] == observation["observation_id"]


def test_generated_far_ui_creates_actual_paired_records_without_typing(ns):
    prepare_generated(ns, case("TC0001"), {"scene": "grass_open"})
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    # Exercise the preserved individual controls explicitly; scene workflow is the new default.
    app.session_state["scene_individual_real"] = True
    app.session_state["scene_individual_generated"] = True
    app.session_state["collection_mode_Generated"] = "929 controlled cases"
    app.session_state["collection_mode_Real Farm"] = "929 controlled cases"
    app.query_params.update(dict(ns["st"].query_params))
    app.query_params["section"] = "Faults"
    app.query_params["view"] = "rgb"
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label == "GENERATE VISIBLE FAR CONDITION").click().run()
    assert not app.exception and not app.error
    records = app.session_state["random_fault_records"]
    assert len(records) == 12 and {r["fault_type"] for r in records} == {"SurfaceObstruction"}
    assert all(c["visible_at_tested_scale"] for c in app.session_state["pv_far_surface_visibility"].values())
    before = deepcopy(records)
    app.query_params["view"] = "thermal"
    app.run()
    assert not app.exception and app.session_state["random_fault_records"] == before
