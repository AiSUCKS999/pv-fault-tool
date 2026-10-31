"""Random hotspot buttons must update camera widgets before they are drawn."""
from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def application():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.mark.parametrize("initial_distance,initial_display,healthy", [
    ("Far View", "RGB faults", False),
    ("Close View", "Original RGB", True),
])
def test_random_generation_regeneration_and_removal_preserve_manual_hotspots(application, initial_distance, initial_display, healthy):
    ns = application
    ns["st"].session_state.clear()
    ns["st"].query_params.clear()
    case = next(c for c in ns["real_farm_catalog_entries"]() if c["id"] == "image_05")
    ns["st"].session_state["real_farm_prepared_picker"] = case["id"]
    ns["real_farm_catalog_selected"]()
    state = ns["st"].session_state["real_farm_state"]
    image = ns["st"].session_state["real_farm_image"]
    modules = ns["real_farm_camera_candidates"](state, image)[1]
    ns["real_farm_camera_new_observation"](state, image, modules[0])
    manual_id = state["camera_observations"][0]["observation_id"]

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    # Exercise the preserved individual controls explicitly; scene workflow is the new default.
    app.session_state["scene_individual_real"] = True
    app.session_state["scene_individual_generated"] = True
    app.session_state["collection_mode_Generated"] = "929 controlled cases"
    app.session_state["collection_mode_Real Farm"] = "929 controlled cases"
    app.query_params["base_source"] = "real_farm"
    for key in ("real_farm_state", "real_farm_image", "real_farm_image_path"):
        app.session_state[key] = deepcopy(ns["st"].session_state[key])
    app.session_state["real_farm_camera_ui_source"] = ns["real_farm_source_identity"](image)
    app.session_state["real_farm_section"] = "Faults"
    app.session_state["real_farm_camera_display"] = initial_display
    app.session_state["real_farm_camera_healthy"] = healthy
    app.session_state["real_farm_camera_grayscale"] = healthy
    app.session_state["real_farm_camera_distance"] = initial_distance
    app.session_state["real_farm_camera_placement"] = "Random Hotspots"
    app.session_state["real_farm_camera_random_fault_types"] = ["SingleHotSpot"]
    app.session_state["real_farm_camera_random_count"] = 2
    app.session_state["real_farm_camera_random_seed"] = 137
    app.run()
    assert not app.exception
    assert app.radio(key="real_farm_camera_placement").value == "Random Faults"

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception
        return app.session_state["real_farm_state"]["camera_observations"]

    observations = click("Generate Random Faults")
    assert app.radio(key="real_farm_camera_distance").value == "Far View"
    assert app.radio(key="real_farm_camera_display").value == "RGB + Thermal"
    assert not app.checkbox(key="real_farm_camera_healthy").value
    assert not app.checkbox(key="real_farm_camera_grayscale").value
    layers = app.session_state["real_farm_camera_layers"][1]
    assert "master_scalar" in layers
    assert layers["acquisition_audit"]["post_sensor_peak"] > 0
    assert any("With synthetic fault" in caption for group in app.get("image") for caption in group.captions)
    assert any("RGB surface effects" in caption for group in app.get("image") for caption in group.captions)
    assert {button.label for button in app.get("download_button")} >= {"Download RGB faults", "Download thermal faults"}
    acquired = next(value for key, value in app.session_state["real_farm_camera_acquisitions"].items() if key[3] == "far")
    healthy_image = ns["real_farm_camera_acquisition"](layers, [], "far")
    assert np.any(np.asarray(acquired) != np.asarray(healthy_image))
    random = [o for o in observations if o.get("placement_origin") == "random"]
    assert len(random) == 2 and {o["seed"] for o in random} == {137}
    assert manual_id in {o["observation_id"] for o in observations}
    previous_batch = random[0]["random_batch_id"]

    app.number_input(key="real_farm_camera_random_count").set_value(3).run()
    app.number_input(key="real_farm_camera_random_seed").set_value(271).run()
    observations = click("Generate Random Faults")
    random = [o for o in observations if o.get("placement_origin") == "random"]
    assert len(random) == 3 and {o["seed"] for o in random} == {271}
    assert all(o["random_batch_id"] != previous_batch for o in random)
    assert manual_id in {o["observation_id"] for o in observations}

    # Existing hotspots also have a one-click route back from RGB to their
    # thermal result, without regenerating or changing the observation batch.
    before = [(o["observation_id"], o["module_id"], o["seed"]) for o in observations]
    app.radio(key="real_farm_camera_display").set_value("RGB faults").run()
    assert not any("could not be shown in RGB" in info.value for info in app.info)
    assert any("Thermal only" in caption.value for caption in app.caption)
    observations = click("Show RGB + thermal")
    assert app.radio(key="real_farm_camera_display").value == "RGB + Thermal"
    assert [(o["observation_id"], o["module_id"], o["seed"]) for o in observations] == before

    observations = click("Remove Random Faults")
    assert [o["observation_id"] for o in observations] == [manual_id]
