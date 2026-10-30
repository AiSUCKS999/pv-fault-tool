"""Exercise the actual random-hotspot buttons after Far/Close widgets exist."""
from copy import deepcopy
from pathlib import Path
import runpy
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pv_data_collection import _load_cases
from pv_data_collection_prepare import prepare_real, real_targets


@pytest.fixture(scope="module")
def application():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.mark.parametrize("initial_view", ["Far View", "Close View"])
def test_random_generation_switches_to_far_and_preserves_manual_observation(application, initial_view):
    ns = application
    ns["st"].session_state.clear()
    ns["st"].query_params.clear()
    mid = real_targets(ns, "image_05")[0]
    case = next(c for c in _load_cases() if c["Case ID"] == "TC0497")
    prepare_real(ns, case, {"source": "image_05", "modules": [mid], "phase": "Far observation"})
    saved = ns["st"].session_state
    manual = deepcopy(saved["real_farm_state"]["camera_observations"][0])
    geometry = deepcopy(saved["real_farm_state"]["modules"])
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=35)
    app.query_params["base_source"] = "real_farm"
    for key in ("real_farm_state", "real_farm_image", "real_farm_image_path"):
        app.session_state[key] = deepcopy(saved[key])
    app.session_state["real_farm_section"] = "Faults"
    app.session_state["real_farm_camera_ui_source"] = saved["real_farm_state"]["source_rgb_pixel_sha256"]
    app.session_state["real_farm_camera_display"] = "RGB faults"
    app.session_state["real_farm_camera_distance"] = initial_view
    app.session_state["real_farm_camera_placement"] = "Random Hotspots"
    app.run()
    assert not app.exception

    def click(label):
        next(b for b in app.button if b.label == label).click().run()
        assert not app.exception

    def assert_batch(count):
        state = app.session_state["real_farm_state"]
        observations = state["camera_observations"]
        assert len(observations) == count + 1
        assert sum(o.get("placement_origin") == "random" for o in observations) == count
        kept = next(o for o in observations if o["observation_id"] == manual["observation_id"])
        for key in ("module_id", "severity", "seed", "local_uv", "diagnostic_fault_type"):
            assert kept.get(key) == manual.get(key)
        assert state["modules"] == geometry
        assert state["source_rgb_pixel_sha256"] == saved["real_farm_state"]["source_rgb_pixel_sha256"]

    click("Generate Random Hotspots")
    assert app.radio(key="real_farm_camera_distance").value == "Far View"
    assert_batch(3)
    app.radio(key="real_farm_camera_distance").set_value("Close View").run()
    app.number_input(key="real_farm_camera_random_count").set_value(2).run()
    click("Generate Random Hotspots")
    assert app.radio(key="real_farm_camera_distance").value == "Far View"
    assert_batch(2)
    # Existing thermal frame, so this checks the real acquisition without recovery.
    app.session_state["real_farm_camera_layers"] = ((saved["real_farm_state"]["source_rgb_pixel_sha256"],
        ns["real_farm_geometry_signature"](geometry), "real-native-source-surfaces-v24-edges-v1-gap-v1-table-v2"),
        ns["real_farm_camera_build_layers"](saved["real_farm_state"], saved["real_farm_image"]))
    app.radio(key="real_farm_camera_display").set_value("Thermal").run()
    assert not app.exception
    assert_batch(2)
    click("Remove Random Hotspots")
    assert_batch(0)
