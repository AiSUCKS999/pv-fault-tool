"""Actual engine preparation and clean-approved-main baseline checks."""
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pv_data_collection as dc
from pv_data_collection_prepare import prepare_generated, prepare_real, real_targets, restore_previous

BASE = "3db093bfeb48e3636c29375d46a64cd48bd5fe9f"


def namespace(ref=None):
    if ref:
        def source(name):
            return subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT).decode("utf-8-sig")
        tree = ast.parse(source("app.py"))
        files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_SECTION_FILES" for t in n.targets))
        ns = {"__file__": str(ROOT / "app.py"), "__name__": "clean_baseline"}
        for name in files:
            exec(compile(source(name), str(ROOT / name), "exec"), ns)
    else:
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="collection_acceptance")
    fake = SimpleNamespace(session_state={}, query_params={})
    ns["st"] = fake
    ns["qp_set"].__globals__["st"] = fake
    return ns


@pytest.fixture(scope="module")
def ns():
    return namespace()


def case(cid):
    return next(c for c in dc._load_cases() if c["Case ID"] == cid)


@pytest.mark.parametrize("cid,scene,phase", [
    ("TC0001", "grass_open", "Healthy"),
    ("TC0002", "grass_open", "Far observation"),
    ("TC0007", "agri_field_new", "Close diagnosis"),
    ("TC0094", "grass_open", "Close diagnosis"),
    ("TC0173", "desert_track", "Close diagnosis"),
    ("TC0387", "floating_water", "Far observation"),
    ("TC0394", "grass_open", "Close diagnosis"),
    ("TC0092", "rooftop_warehouse", "Far observation"),
    ("TC0003", "grass_open", "Far observation"),
    ("TC0388", "grass_open", "Far observation"),
    ("TC0010", "grass_open", "Close diagnosis"),
    ("TC0011", "grass_open", "Close diagnosis"),
    ("TC0005", "grass_open", "Close diagnosis"),
])
def test_actual_generated_target_fault_severity_and_views(ns, cid, scene, phase):
    c = case(cid)
    result = prepare_generated(ns, c, {"scene": scene, "phase": phase})
    detail = result["details"]
    assert (detail["rows"], detail["columns"], detail["row"], detail["array"], detail["module"], detail["cell"]) == tuple(
        int(c[k]) for k in ("Rows", "Columns / row", "Target row", "Target array col", "Module slot (1–4)", "Cell index (1–40)"))
    assert ns["custom_row_count"](scene) == int(c["Rows"])
    assert ns["custom_panels_per_row"](scene) == int(c["Columns / row"])
    assert detail["requested_seed"] == int(c["Seed"])
    records = ns["records_for_location"](scene, "thermal")
    if phase == "Healthy":
        assert records == [] and detail["changed_pixels"] == 0
    else:
        assert len(records) == 1
        assert records[0]["target_cell_id"] == detail["target_cell_id"]
        assert records[0]["fault_scale"] == int(c["Severity (1–10)"])
        assert records[0]["fault_type"] == (c["CLOSE diagnosis code"] if phase == "Close diagnosis" else c["FAR observable code"])
        assert detail["changed_pixels"] > 0
        assert result["status"] == "Guided" and "not supported" in detail["seed_behavior"]
    # Actual existing image and metadata exporters, not substitute exports.
    image = ns["module_field_image"](scene, "thermal", records[0]["row_id"] if records else ns["qp_get"]("selected_pv"),
                detail["target_cell_id"], records, False, scene_distance="far")
    png = ns["export_current_view_bytes"](image)
    assert png.startswith(b"\x89PNG")
    assert isinstance(ns["fault_records_csv"](records), str)


def test_far_then_close_phase_and_recovery_of_existing_faults(ns):
    ns["st"].session_state["random_fault_records"] = [{"prior": "keep"}]
    old_query = dict(ns["st"].query_params)
    c = case("TC0173")
    far = prepare_generated(ns, c, {"scene": "desert_track", "phase": "Far observation"})
    assert far["details"]["fault"] == "SingleHotSpot"
    restore_previous(ns)
    assert ns["st"].session_state["random_fault_records"] == [{"prior": "keep"}]
    assert ns["st"].query_params == old_query
    close = prepare_generated(ns, c, {"scene": "desert_track", "phase": "Close diagnosis"})
    assert close["details"]["fault"] == "SingleDiode"
    assert ns["qp_get"]("selected_module") == c["Module slot (1–4)"]


def test_original_simulator_survives_multiple_preparations(ns):
    ns["st"].session_state.pop("pv_dc_original_simulator", None)
    ns["st"].session_state["random_fault_records"] = [{"original": "retain across cases"}]
    before = dict(ns["st"].query_params)
    prepare_generated(ns, case("TC0001"), {"scene": "grass_open"})
    prepare_generated(ns, case("TC0002"), {"scene": "grass_open"})
    restore_previous(ns, original=True)
    assert ns["st"].session_state["random_fault_records"] == [{"original": "retain across cases"}]
    assert ns["st"].query_params == before


@pytest.mark.parametrize("source,offset,stage", [("image_01", 0, "Far observation"), ("image_02", 1, "Close diagnosis")])
def test_real_seeded_module_and_actual_engine_preserves_geometry(ns, source, offset, stage):
    c = next(c for c in dc._load_cases() if c["Source mode"] == "Real Farm" and c["Scenario stage"] == stage
             and (stage != "Close diagnosis" or c["CLOSE diagnosis code"] == "SingleDiode"))
    ids = real_targets(ns, source)
    options = {"source": source, "modules": [ids[offset]], "phase": stage}
    result = prepare_real(ns, c, options)
    state = ns["st"].session_state["real_farm_state"]
    modules = deepcopy(state["modules"])
    observation = state["camera_observations"][0]
    assert observation["module_id"] == ids[offset]
    assert observation["seed"] == int(c["Seed"]) and observation["severity"] == int(c["Severity (1–10)"])
    assert observation["diagnostic_fault_type"] == ("SingleDiode" if stage == "Close diagnosis" else None)
    assert result["details"]["scalar_test_changed_pixels"] > 0
    uv = observation["local_uv"]
    repeated = prepare_real(ns, c, options)
    assert ns["st"].session_state["real_farm_state"]["camera_observations"][0]["local_uv"] == uv
    assert ns["st"].session_state["real_farm_state"]["modules"] == modules
    assert repeated["status"] == "Guided"  # Full-scene capture and GT remain explicit manual steps.
    assert not state.get("geometry_confirmation") and not observation["training_verified"]
    exported = json.loads(ns["real_farm_project_json"](state))
    assert exported["camera_observations"][0]["module_id"] == ids[offset]
    assert exported["observations"][0]["diagnostic_fault_type"] == observation["diagnostic_fault_type"]
    assert ns["export_current_view_bytes"](ns["st"].session_state["real_farm_image"]).startswith(b"\x89PNG")
    expected = ns["real_farm_add_module_observation"](modules, [], ids[offset], int(c["Severity (1–10)"]), int(c["Seed"]))
    assert expected["local_uv"] == uv
    with pytest.raises(ValueError, match="Verified-label export blocked"):
        ns["real_farm_dataset_zip"](state, ns["st"].session_state["real_farm_image"], ns["st"].session_state["real_farm_image"])


def test_real_healthy_manual_multiple_and_invalid_targets(ns):
    healthy = case("TC0496")
    prepare_real(ns, healthy, {"source": "image_01"})
    assert ns["st"].session_state["real_farm_state"]["camera_observations"] == []
    c = case("TC0498")
    manual = prepare_real(ns, c, {"source": "image_01", "modules": []})
    assert "No fault applied" in manual["details"]["fault"]
    assert ns["st"].session_state["real_farm_state"]["camera_observations"] == []
    ids = real_targets(ns, "image_01")[:2]
    multiple = prepare_real(ns, c, {"source": "image_01", "modules": ids})
    assert len(multiple["details"]["observation_ids"]) == 2
    geometry = deepcopy(ns["st"].session_state["real_farm_state"]["modules"])
    with pytest.raises(ValueError, match="credible physical"):
        prepare_real(ns, c, {"source": "image_01", "modules": ["invented"]})
    assert ns["st"].session_state["real_farm_state"]["modules"] == geometry
    assert len(ns["st"].session_state["real_farm_state"]["camera_observations"]) == 2


def test_blocked_combination_rolls_back_without_pretending_fault(ns):
    before = dict(ns["st"].query_params)
    with pytest.raises(ValueError, match="Known approved-main"):
        prepare_generated(ns, case("TC0391"), {"scene": "rooftop_warehouse", "phase": "Close diagnosis"})
    assert ns["st"].query_params == before
    with pytest.raises(ValueError, match="planned topology"):
        prepare_generated(ns, next(c for c in dc._load_cases() if c["Topology"] == "Ragged edge"), {"scene": "grass_open"})


@pytest.fixture(scope="module")
def baseline():
    return namespace(BASE)


def test_clean_approved_main_real_regression_failure(baseline, monkeypatch):
    regression = runpy.run_path(str(ROOT / "scripts/regression_real_farm.py"), run_name="baseline_regression")
    monkeypatch.setattr(runpy, "run_path", lambda *a, **kw: baseline)
    with pytest.raises(ValueError, match="Verified-label export blocked"):
        regression["main"]()


def test_clean_approved_main_rooftop_matrix_failure(baseline):
    ns = baseline
    ns["qp_set"](location="rooftop_warehouse", custom_layout=1, row_count=6, panels_per_row=6)
    scene = "rooftop_warehouse"
    row_id, cell = ns["default_cell_for_array"](scene, 4, 1, 3, 36)
    rgb = ns["module_field_image"](scene, "rgb", row_id, cell["id"], [], False)
    regions = ns["array_regions"](scene, rgb)
    geometry = {(int(r["row"]), int(r["col"])): {"box": r["box"], "polygon": r.get("polygon") or ns["box_polygon"](r["box"])} for r in regions}
    record = ns["fault_record"]({"id": scene, "name": scene}, "thermal", ns["row_by_id"](scene, row_id), cell, "SoilingDust", 8)
    rendered = ns["rendered_fault_cell_from_record"](scene, record, row_id, cell["id"], geometry)
    with pytest.raises(ValueError, match="truth value of an array"):
        ns["apply_thermal_fault"](np.full((720, 1080), 156., np.float32), rendered, "SoilingDust", 8)
