"""Random fault batches cover selected diagnoses without moving their targets."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def application():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.fixture
def scene(application):
    ns = application
    ns["st"].session_state.clear()
    ns["st"].query_params.clear()
    ns["st"].session_state["real_farm_prepared_picker"] = "image_05"
    ns["real_farm_catalog_selected"]()
    state = ns["st"].session_state["real_farm_state"]
    image = ns["st"].session_state["real_farm_image"]
    credible = ns["real_farm_camera_candidates"](state, image)[1]
    return SimpleNamespace(ns=ns, state=state, image=image, credible=credible,
                           geometry=deepcopy(state["modules"]), pixels=image.tobytes())


def random_observations(state):
    return [o for o in state["camera_observations"] if o.get("placement_origin") == "random"]


def classes(observations):
    return [o["diagnostic_fault_type"] or "SingleHotSpot" for o in observations]


def placement(observations):
    return [(o["module_id"], o["center_rgb_px"], o["local_uv"], o["severity"])
            for o in observations]


def generate(scene, count, seed=137, fault_types=None, policy="Balanced"):
    return scene.ns["real_farm_camera_generate_random"](
        scene.state, scene.image, scene.credible, count, seed, fault_types=fault_types, policy=policy)


def test_explicit_balanced_batches_cover_every_diagnostic_once_per_cycle(scene):
    expected = set(scene.ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"])
    assert len(expected) == 7 and len(scene.credible) >= 14
    assert generate(scene, 14) == 14
    observations = random_observations(scene.state)
    assigned = classes(observations)
    assert len(observations) == 14
    assert set(assigned[:7]) == expected
    assert set(assigned[7:14]) == expected
    assert Counter(assigned) == Counter({fault: 2 for fault in expected})
    assert all(o["diagnostic_fault_type"] is not None for o in observations)
    batch = scene.state["camera_random_batch"]
    assert set(batch["fault_types"]) == expected
    assert batch["fault_counts"] == dict(Counter(assigned))
    layers = scene.ns["real_farm_camera_build_layers"](scene.state, scene.image)
    scene.ns["real_farm_camera_acquisition"](layers, observations, "far")
    # Offline/open-circuit patterns can cool rather than heat. The engine's
    # per-observation footprint measures absolute change and covers both.
    footprints = [{"id": o["observation_id"], "assigned": o["diagnostic_fault_type"],
                   "rendered": o.get("footprint_rgb", {}).get("diagnosis"),
                   "native_pixels": o.get("footprint_rgb", {}).get("native_pixels", 0)}
                  for o in observations]
    assert all(f["assigned"] == f["rendered"] and f["native_pixels"] > 0 for f in footprints), footprints
    assert scene.state["modules"] == scene.geometry
    assert scene.image.tobytes() == scene.pixels


def test_small_batches_are_unique_repeatable_and_class_rng_does_not_move_targets(scene):
    generate(scene, 3, seed=271)
    first = deepcopy(random_observations(scene.state))
    assert len(set(classes(first))) == 3
    generate(scene, 3, seed=271)
    repeated = random_observations(scene.state)
    assert classes(repeated) == classes(first)
    assert placement(repeated) == placement(first)

    # Choosing a different diagnosis subset must not consume the placement RNG.
    generate(scene, 3, seed=271, fault_types=["CellCracking"])
    cracks = random_observations(scene.state)
    assert classes(cracks) == ["CellCracking"] * 3
    assert placement(cracks) == placement(first)
    assert scene.state["modules"] == scene.geometry
    assert scene.image.tobytes() == scene.pixels


def test_selected_subset_replaces_only_random_faults_and_keeps_manual_observation(scene):
    manual = scene.ns["real_farm_camera_new_observation"](
        scene.state, scene.image, scene.credible[0])
    manual["diagnostic_fault_type"] = "SingleDiode"
    original_manual = deepcopy(manual)
    generate(scene, 5, fault_types=["CellCracking"])
    assert classes(random_observations(scene.state)) == ["CellCracking"] * 5
    assert all(o["module_id"] != manual["module_id"] for o in random_observations(scene.state))
    first_batch = scene.state["camera_random_batch"]["id"]

    generate(scene, 2, seed=271, fault_types=["SoilingDust"])
    random = random_observations(scene.state)
    assert classes(random) == ["SoilingDust"] * 2
    assert len(scene.state["camera_observations"]) == 3
    assert next(o for o in scene.state["camera_observations"]
                if o["observation_id"] == manual["observation_id"]) == original_manual
    batch = scene.state["camera_random_batch"]
    assert batch["id"] != first_batch
    assert batch["fault_types"] == ["SoilingDust"]
    assert batch["fault_counts"] == {"SoilingDust": 2}
    assert scene.state["modules"] == scene.geometry


@pytest.mark.parametrize("invalid", [[], ["NotAFault"], ["CellCracking", "NotAFault"]])
def test_invalid_fault_selection_is_rejected_before_changing_observations(scene, invalid):
    manual = scene.ns["real_farm_camera_new_observation"](
        scene.state, scene.image, scene.credible[0])
    generate(scene, 2, fault_types=["SoilingDust"])
    # Even ordinary source/label normalization must wait until validation.
    manual["diagnostic_fault_type"] = "SingleHotSpot"
    before = deepcopy(scene.state)
    with pytest.raises(ValueError):
        generate(scene, 3, fault_types=invalid)
    assert scene.state == before
    assert scene.image.tobytes() == scene.pixels


def test_hotspot_symptom_is_available_only_when_explicitly_selected(scene):
    generate(scene, 2, fault_types=["CellCracking", "SingleHotSpot"])
    observations = random_observations(scene.state)
    assert set(classes(observations)) == {"CellCracking", "SingleHotSpot"}
    assert sum(o["diagnostic_fault_type"] is None for o in observations) == 1
    assert scene.state["camera_random_batch"]["fault_counts"] == {
        "CellCracking": 1, "SingleHotSpot": 1}


def test_cold_rgb_random_cracking_generates_physical_rgb_and_paired_thermal(scene):
    ns = scene.ns
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    # Exercise the preserved individual controls explicitly; scene workflow is the new default.
    app.session_state["scene_individual_real"] = True
    app.session_state["scene_individual_generated"] = True
    app.session_state["collection_mode_Generated"] = "929 controlled cases"
    app.session_state["collection_mode_Real Farm"] = "929 controlled cases"
    app.query_params["base_source"] = "real_farm"
    for key in ("real_farm_state", "real_farm_image", "real_farm_image_path"):
        app.session_state[key] = deepcopy(ns["st"].session_state[key])
    app.session_state["real_farm_camera_ui_source"] = ns["real_farm_source_identity"](scene.image)
    app.session_state["real_farm_section"] = "Faults"
    app.session_state["real_farm_camera_display"] = "RGB faults"
    app.session_state["real_farm_camera_placement"] = "Random Faults"
    app.session_state["real_farm_camera_random_count"] = 1
    app.session_state["real_farm_camera_random_seed"] = 137
    app.run()
    assert not app.exception
    assert "real_farm_camera_layers" not in app.session_state
    selector = app.multiselect(key="real_farm_camera_random_fault_types")
    assert set(selector.value) == set(ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"])
    selector.set_value(["CellCracking"]).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Generate Random Faults").click().run()
    assert not app.exception and not app.error

    state = app.session_state["real_farm_state"]
    observations = random_observations(state)
    assert len(observations) == 1
    assert observations[0]["diagnostic_fault_type"] == "CellCracking"
    assert state["camera_random_batch"]["fault_counts"] == {"CellCracking": 1}
    assert app.radio(key="real_farm_camera_display").value == "RGB + Thermal"
    assert {button.label for button in app.get("download_button")} >= {
        "Download RGB faults", "Download thermal faults"}
    captions = [caption for group in app.get("imgs") for caption in [im.caption for im in group.proto.imgs]]
    assert any("RGB surface effects" in caption for caption in captions)
    assert any("With synthetic fault" in caption for caption in captions)
    layers = app.session_state["real_farm_camera_layers"][1]
    assert layers["acquisition_audit"]["post_sensor_peak"] > 0

    # Inspect the unannotated surface cache: a location marker alone is not a
    # successful physical RGB cracking simulation.
    _, physical_rgb, report = app.session_state["real_farm_rgb_preview_cache"]
    rendered = [r for r in report["rendered"]
                if r["observation_id"] == observations[0]["observation_id"]]
    assert rendered, report
    assert rendered[0]["fault_type"] == "CellCracking" and rendered[0]["changed_pixels"] > 0
    assert np.any(np.asarray(physical_rgb) != np.asarray(scene.image))
    assert app.session_state["real_farm_image"].tobytes() == scene.pixels
    assert state["modules"] == scene.geometry
