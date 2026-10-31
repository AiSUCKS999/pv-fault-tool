"""Bounded real-engine and Streamlit tests for hands-free recording."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pv_data_collection import _load_cases
from pv_data_collection_prepare import restore_previous
from pv_experiment_log import Journal, KEY, fingerprint
from pv_experiment_capture import actual_state, assessment, collect_automatically, ensure_reference, prepare, render_reference


@pytest.fixture(scope="module")
def application_namespace():
    return runpy.run_path(str(ROOT / "tests/test_collection_acceptance.py"))["namespace"]()


@pytest.fixture
def ns(application_namespace):
    application_namespace["st"].session_state.clear()
    application_namespace["st"].query_params.clear()
    return application_namespace


def case(cid):
    return next(c for c in _load_cases() if c["Case ID"] == cid)


@pytest.mark.parametrize("cid,scene", [("TC0001", "grass_open"), ("TC0002", "agri_field_new"), ("TC0007", "grass_open"), ("TC0393", "desert_track")])
def test_actual_generated_state_and_exact_clean_reference(ns, cid, scene):
    j = Journal(ns["st"].session_state)
    c = case(cid)
    exp = prepare(ns, j, c, {"scene": scene})
    assert not exp["blocked_reason"]
    target = exp["stages"]["FAR"]["preparations"][-1]["actual"]["selection"]
    assert target["cell_id"] == ns["qp_get"]("selected_cell")
    assert target["row"] == int(c["Target row"])
    assert target["array"] == int(c["Target array col"])
    far = ensure_reference(ns, j, exp, "FAR", True)
    raw = ns["module_field_image"](scene, "thermal", target["row_id"], target["cell_id"],
        ns["records_for_location"](scene, "thermal"), False, scene_distance="far")
    assert ns["export_current_view_bytes"](raw) == j.blobs[far["reference_id"]]
    assert far["sha256"] == hashlib.sha256(j.blobs[far["reference_id"]]).hexdigest()
    assert far["parameters"]["overlays"] is False
    if c["Scenario stage"] == "Healthy":
        assert far["actual"]["faults"] == [] and far["parameters"]["rendered_changed_pixels"] == 0
    else:
        assert far["actual"]["faults"][0]["fault_type"] == c["FAR observable code"]
        assert far["actual"]["faults"][0]["severity"] == int(c["Severity (1–10)"])
        assert far["actual"]["faults"][0]["applied_seed"] is None
        assert far["actual"]["faults"][0]["engine_observation_id"] is None
        assert "collection-derived" in far["actual"]["faults"][0]["identity_kind"]
        assert far["parameters"]["rendered_changed_pixels"] > 0
    j.confirm_capture(exp, "FAR", actual_state(ns, "Generated"))
    if c["CLOSE required?"] == "Yes":
        eid = exp["experiment_id"]
        assert prepare(ns, j, c, {"scene": scene}, "CLOSE", exp)["experiment_id"] == eid
        close = ensure_reference(ns, j, exp, "CLOSE", True)
        assert close["actual"]["faults"][0]["fault_type"] == c["CLOSE diagnosis code"]
        assert close["parameters"]["crop_bounds"] and close["parameters"]["rendered_changed_pixels"] > 0
        assert exp["stages"]["CLOSE"]["preparations"][-1]["changes_from_far"]
        j.confirm_capture(exp, "CLOSE", actual_state(ns, "Generated"))
    j.finish(exp)
    assert exp["finished_utc"] and not exp["physical_ground_truth_verified"]


def test_existing_manual_diagnosis_is_read_from_effective_engine_state(ns):
    c = case("TC0007")
    j = Journal(ns["st"].session_state)
    exp = prepare(ns, j, c, {"scene": "grass_open"})
    far = ensure_reference(ns, j, exp, "FAR", True)
    j.confirm_capture(exp, "FAR", actual_state(ns, "Generated"))
    target = far["actual"]["selection"]
    location = ns["location_by_id"]("grass_open")
    cell = ns["cell_by_id"]("grass_open", target["row_id"], target["cell_id"])
    record = ns["fault_record"](location, "thermal", ns["row_by_id"]("grass_open", target["row_id"]), cell, "CellCracking", 7)
    ns["st"].session_state["panel_cell_fault_records"] = [record]
    assert len(ns["records_for_location"]("grass_open", "thermal")) == 2
    close = ensure_reference(ns, j, exp, "CLOSE", True)
    assert len(close["actual"]["faults"]) == 1  # Existing diagnosis replacement semantics.
    assert close["actual"]["faults"][0]["severity"] == 7  # Not blindly copied from plan severity 3.
    assert far["actual"]["faults"][0]["fault_type"] == "SingleHotSpot"
    assert not exp["stages"]["CLOSE"]["preparations"][-1]["assessment"]["configuration_matches_plan"]
    assert far["experiment_id"] == close["experiment_id"]
    j.confirm_capture(exp, "CLOSE", actual_state(ns, "Generated"))
    with pytest.raises(ValueError, match="Advanced"):
        j.finish(exp)


@pytest.mark.parametrize("cid", ["TC0496", "TC0502"])
def test_real_photo_native_reference_actual_ids_and_geometry(ns, cid):
    c = case(cid)
    j = Journal(ns["st"].session_state)
    exp = prepare(ns, j, c, {"source": "image_05"})
    assert not exp["blocked_reason"]
    state = ns["st"].session_state["real_farm_state"]
    original_geometry = deepcopy(state["modules"])
    far = ensure_reference(ns, j, exp, "FAR", True)
    assert far["image_size"] == [741, 762]
    assert far["actual"]["source_image"]["pixel_sha256"] == state["source_rgb_pixel_sha256"]
    assert far["parameters"]["existing_renderer"] == "real_farm_camera_acquisition"
    if c["Scenario stage"] != "Healthy":
        faults = far["actual"]["faults"]
        assert [f["engine_observation_id"] for f in faults] == [o["observation_id"] for o in state["camera_observations"]]
        assert faults[0]["applied_seed"] == int(c["Seed"])
        assert faults[0]["engine_footprint_native_pixels"] > 0
    j.confirm_capture(exp, "FAR", actual_state(ns, "Real Farm"))
    if c["CLOSE required?"] == "Yes":
        prepare(ns, j, c, {"source": "image_05"}, "CLOSE", exp)
        close = ensure_reference(ns, j, exp, "CLOSE", True)
        state = ns["st"].session_state["real_farm_state"]
        assert state["modules"] == original_geometry
        assert close["actual"]["faults"][0]["fault_type"] == c["CLOSE diagnosis code"]
        bounds = tuple(close["parameters"]["crop_bounds"])
        layers = ns["st"].session_state["real_farm_camera_layers"][1]
        exact = ns["real_farm_camera_acquisition"](layers, state["camera_observations"], "close", bounds)
        assert ns["export_current_view_bytes"](exact) == j.blobs[close["reference_id"]]
        j.confirm_capture(exp, "CLOSE", actual_state(ns, "Real Farm"))
    j.finish(exp)
    assert ns["st"].session_state["real_farm_state"]["modules"] == original_geometry
    assert not state.get("geometry_confirmation")


def test_failed_preparation_and_export_do_not_claim_success(ns, monkeypatch):
    import pv_data_collection_prepare as adapter
    import pv_experiment_ui as ui
    j = Journal(ns["st"].session_state)
    ns["st"].session_state["random_fault_records"] = [{"keep": "prior user state"}]
    exp = prepare(ns, j, case("TC0391"), {"scene": "rooftop_warehouse"})
    assert exp["blocked_reason"] and not exp["stages"]["FAR"]["references"]
    assert ns["st"].session_state["random_fault_records"] == [{"keep": "prior user state"}]
    exp = prepare(ns, j, case("TC0002"), {"scene": "grass_open"})
    monkeypatch.setattr(ui, "st", ns["st"])
    def failed_export(*args):
        raise RuntimeError("Exporter failed")
    monkeypatch.setitem(ns, "export_current_view_bytes", failed_export)
    ui.action(ns, "Generated", exp["experiment_id"], "save", "FAR")
    assert "Exporter failed" in exp["blocked_reason"]
    assert not exp["stages"]["FAR"]["references"] and not exp["stages"]["FAR"]["capture_confirmations"]
    assert not exp.get("finished_utc")


def test_source_change_is_not_silently_paired_and_original_state_is_recoverable(ns):
    ns["st"].session_state["random_fault_records"] = [{"original": True}]
    j = Journal(ns["st"].session_state)
    exp = prepare(ns, j, case("TC0002"), {"scene": "grass_open"})
    far = ensure_reference(ns, j, exp, "FAR", True)
    j.confirm_capture(exp, "FAR", actual_state(ns, "Generated"))
    ns["st"].query_params["location"] = "agri_field_new"
    with pytest.raises(ValueError, match="source changed"):
        ensure_reference(ns, j, exp, "CLOSE", True)
    assert not exp["stages"]["CLOSE"]["references"]
    other = prepare(ns, j, case("TC0002"), {"scene": "agri_field_new"}, exp=exp)
    assert other["experiment_id"] != exp["experiment_id"]
    restore_previous(ns, original=True)
    assert ns["st"].session_state["random_fault_records"] == [{"original": True}]
    assert j.get(exp["experiment_id"])["stages"]["FAR"]["references"][0] == far


def application(mode="Generated"):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=35)
    if mode == "Generated": app.query_params["section"] = "DATA COLLECTION"
    else:
        app.query_params["base_source"] = "real_farm"
        app.session_state["real_farm_section"] = "DATA COLLECTION"
    app.run()
    assert not app.exception
    return app


@pytest.mark.parametrize("mode,cid", [("Generated", "TC0007"), ("Real Farm", "TC0502")])
def test_normal_ui_completion_requires_no_manual_recording_or_typing(mode, cid):
    app = application(mode)
    app.selectbox(key=f"pv_dc_case_{mode}").select(cid).run()
    if mode == "Real Farm": app.selectbox(key="pv_dc_photo_RF01").select("image_05").run()
    app.button(key=f"pv_dc_collect_{mode}").click().run()
    assert not app.exception and not app.error
    data = app.session_state[KEY]
    exp = data["experiments"][data["active_by_mode"][mode]]
    assert exp["finished_utc"] and exp["completion_scope"] == "simulator_only"
    assert exp["status"] == "Simulator collection complete"
    assert not exp["physical_ground_truth_verified"] and not exp["yolo_detection_verified"]
    assert not app.text_input and not app.get("form") and not app.checkbox
    assert not any(x.label in {"Test status", "Fault type", "Severity", "Seed"} for x in app.selectbox)
    advanced = next(x for x in app.expander if x.label.startswith("Advanced"))
    assert advanced.proto.expanded is False
    assert not any(x.label.startswith("CONFIRM FAR") or x.label.startswith("CONFIRM CLOSE") for x in app.button)
    for view in ("FAR", "CLOSE"):
        stage = exp["stages"][view]
        assert len(stage["references"]) == 1
        assert stage["capture_confirmations"] == [] and stage["video"] is None
        ref = stage["references"][0]
        assert ref["experiment_id"] == exp["experiment_id"]
        assert not ref["file_saved_confirmed_utc"] and not ref["download_requested_utc"]
    app.button(key=f"pv_dc_next_{mode}").click().run()
    assert app.selectbox(key=f"pv_dc_case_{mode}").value != cid
    assert exp["experiment_id"] in app.session_state[KEY]["experiments"]


def test_original_priorities_and_restrictions_remain_visible_in_ui():
    app = application()
    assert len(app.selectbox(key="pv_dc_case_Generated").options) == 789
    app.selectbox(key="pv_dc_priority_Generated").select("P0 – Essential").run()
    assert len(app.selectbox(key="pv_dc_case_Generated").options) == 22
    app.selectbox(key="pv_dc_priority_Generated").select("All priorities").run()
    app.selectbox(key="pv_dc_case_Generated").select("TC0391").run()
    app.selectbox(key="pv_dc_generated_scene").select("rooftop_warehouse").run()
    assert app.button(key="pv_dc_collect_Generated").disabled
    exp = Journal({KEY: app.session_state[KEY]}).active("Generated", "TC0391")
    assert "Known approved-main" in exp["blocked_reason"]
    assert not exp["stages"]["FAR"]["references"]
    real = application("Real Farm")
    assert len(real.selectbox(key="pv_dc_case_Real Farm").options) == 140
    assert len(real.selectbox(key="pv_dc_photo_RF01").options) == 14


def test_switching_cases_resumes_the_same_experiment_and_rebinds_the_simulator(ns):
    j = Journal(ns["st"].session_state)
    a = prepare(ns, j, case("TC0007"), {"scene": "grass_open"})
    far = ensure_reference(ns, j, a, "FAR", True)
    j.confirm_capture(a, "FAR", actual_state(ns, "Generated"))
    b = prepare(ns, j, case("TC0002"), {"scene": "agri_field_new"})
    assert j.active("Generated", "TC0007")["experiment_id"] == a["experiment_id"]
    with pytest.raises(ValueError, match="another context"):
        ensure_reference(ns, j, a, "CLOSE", True)
    prepare(ns, j, case("TC0007"), {"scene": "agri_field_new"}, "CLOSE", j.active("Generated", "TC0007"))
    close = ensure_reference(ns, j, a, "CLOSE", True)
    assert close["experiment_id"] == far["experiment_id"] == a["experiment_id"]
    assert close["actual"]["scene_identity"] == "grass_open"
    assert j.get(b["experiment_id"])["stages"]["FAR"]["preparations"]
    j.confirm_capture(a, "CLOSE", actual_state(ns, "Generated"))
    j.finish(a)


def test_automatic_export_failure_does_not_finish_or_confirm_any_recording(ns, monkeypatch):
    def failed_export(*args):
        raise RuntimeError("Reference exporter unavailable")
    monkeypatch.setitem(ns, "export_current_view_bytes", failed_export)
    j = Journal(ns["st"].session_state)
    exp = collect_automatically(ns, j, case("TC0007"), {"scene": "grass_open"})
    assert "Reference exporter unavailable" in exp["blocked_reason"]
    assert not exp.get("finished_utc")
    assert all(not s["references"] and not s["capture_confirmations"] for s in exp["stages"].values())
