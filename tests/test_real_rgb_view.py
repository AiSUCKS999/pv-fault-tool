"""Natural RGB view tests: existing effects, unchanged source/geometry/thermal engine."""
from copy import deepcopy
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pv_data_collection import _load_cases
from pv_data_collection_prepare import prepare_real, real_targets
from pv_real_farm_rgb_view import rgb_fault_preview, view_bounds


@pytest.fixture(scope="module")
def app_namespace():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.fixture
def scene(app_namespace):
    ns = app_namespace
    ns["st"].session_state.clear()
    ns["st"].query_params.clear()
    targets = real_targets(ns, "image_05")
    return ns, targets[0]


def setup(scene, cid):
    ns, mid = scene
    case = next(c for c in _load_cases() if c["Case ID"] == cid)
    prepare_real(ns, case, {"source": "image_05", "modules": [mid],
        "phase": "Close diagnosis" if case["Scenario stage"] == "Close diagnosis" else "Far observation"})
    session = ns["st"].session_state
    state, image = session["real_farm_state"], session["real_farm_image"]
    modules = ns["real_farm_camera_candidates"](state, image)[1]
    return ns, state, image, modules


@pytest.mark.parametrize("cid", ["TC0499", "TC0500", "TC0501"])
def test_surface_rgb_effect_is_visible_confined_and_does_not_recolor_the_farm(scene, cid):
    ns, state, image, modules = setup(scene, cid)
    original = np.array(image).copy()
    geometry = deepcopy(state["modules"])
    metadata = deepcopy(state)
    result, report = rgb_fault_preview(ns, image, modules, state["camera_observations"])
    changed = np.any(np.array(result) != original, axis=2)
    target = next(m for m in modules if m["module_id"] == state["camera_observations"][0]["module_id"])
    allowed = ns["real_farm_module_mask"](image.size, [target], True) > 0
    assert np.count_nonzero(changed) > 0
    assert np.count_nonzero(changed & ~allowed) == 0
    assert np.count_nonzero(changed) < image.width * image.height * .1
    assert np.array_equal(np.asarray(image), original)
    assert state["modules"] == geometry
    assert state == metadata
    assert report["rendered"][0]["changed_pixels"] > 0 and not report["physical_ground_truth_verified"]
    assert report["rendered"][0]["engine"] == "draw_realistic_fault(view='rgb')"
    again, _ = rgb_fault_preview(ns, image, modules, state["camera_observations"])
    assert np.array_equal(np.asarray(again), np.asarray(result))


@pytest.mark.parametrize("cid", ["TC0497", "TC0502", "TC0503", "TC0504", "TC0505"])
def test_electrical_diagnoses_do_not_fabricate_rgb_damage(scene, cid):
    ns, state, image, modules = setup(scene, cid)
    result, report = rgb_fault_preview(ns, image, modules, state["camera_observations"])
    assert np.array_equal(np.asarray(result), np.asarray(image))
    assert report["rendered"] == []
    assert "No established daylight RGB" in report["unavailable"][0]["reason"]


def test_rgb_close_view_uses_the_selected_physical_module_bounds(scene):
    ns, state, image, modules = setup(scene, "TC0501")
    selected = next(m for m in modules if m["module_id"] == state["camera_observations"][0]["module_id"])
    assert view_bounds(ns, image, selected, modules, "Far View") is None
    close = view_bounds(ns, image, selected, modules, "Close View")
    assert close == ns["real_farm_close_context_bounds"](selected, modules, (image.height, image.width), .25)
    assert image.crop(close).width < image.width


def test_cold_rgb_ui_has_natural_fault_view_without_generating_thermal(scene):
    ns, state, image, modules = setup(scene, "TC0501")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=35)
    # Exercise the preserved individual controls explicitly; scene workflow is the new default.
    app.session_state["scene_individual_real"] = True
    app.session_state["scene_individual_generated"] = True
    app.session_state["collection_mode_Generated"] = "929 controlled cases"
    app.session_state["collection_mode_Real Farm"] = "929 controlled cases"
    app.query_params["base_source"] = "real_farm"
    for key in ("real_farm_state", "real_farm_image", "real_farm_image_path"):
        app.session_state[key] = deepcopy(ns["st"].session_state[key])
    app.session_state["real_farm_section"] = "Faults"
    app.session_state["real_farm_camera_ui_source"] = state["source_rgb_pixel_sha256"]
    app.session_state["real_farm_camera_display"] = "RGB faults"
    app.run()
    assert not app.exception
    assert app.radio(key="real_farm_camera_display").value == "RGB faults"
    assert "real_farm_camera_layers" not in app.session_state
    assert app.session_state["real_farm_rgb_preview_cache"][2]["rendered"][0]["changed_pixels"] > 0
    app.radio(key="real_farm_camera_distance").set_value("Close View").run()
    assert not app.exception
    assert any("Close View · simulated RGB surface appearance" in image.caption for group in app.get("image") for image in group.proto.imgs)
    app.radio(key="real_farm_camera_display").set_value("Original RGB").run()
    assert not app.exception
    assert any("Close View · original RGB photograph" in image.caption for group in app.get("image") for image in group.proto.imgs)


def test_thermal_acquisition_pixels_match_approved_main_after_rgb_ui_changes(scene):
    current, mid = scene
    loader = runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]
    baseline = loader("baf1f8f")
    images = []
    case = next(c for c in _load_cases() if c["Case ID"] == "TC0502")
    for ns in (baseline, current):
        prepare_real(ns, case, {"source": "image_05", "modules": [mid], "phase": "Close diagnosis"})
        state, photo = ns["st"].session_state["real_farm_state"], ns["st"].session_state["real_farm_image"]
        layers = ns["real_farm_camera_build_layers"](state, photo)
        images.append(np.array(ns["real_farm_camera_acquisition"](layers, state["camera_observations"], "far")))
    assert np.array_equal(images[0], images[1])
