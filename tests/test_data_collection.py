"""Bounded integration tests: no server, training, detector, or background job."""
import csv
import hashlib
import io
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageOps
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pv_data_collection as dc


@pytest.fixture
def records(monkeypatch):
    session = {"unrelated_fault_state": {"keep": True}}
    monkeypatch.setattr(dc, "st", SimpleNamespace(session_state=session))
    return session


def app(mode="Generated"):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45)
    if mode == "Generated":
        at.query_params["section"] = "DATA COLLECTION"
    else:
        at.query_params["base_source"] = "real_farm"
        at.session_state["real_farm_section"] = "DATA COLLECTION"
    at.run()
    assert not at.exception
    return at


def by_label(elements, label):
    return next(element for element in elements if element.label == label)


def test_catalog_exact_cases_priorities_and_preparation_limits():
    assert hashlib.sha256(dc.DATA_FILE.read_bytes()).hexdigest() == "03d82a4f958dcbfd9c91864d776fcb72677797caf87fc465689a87c10bcee0a4"
    cases = dc._load_cases()
    assert len(cases) == len({c["Case ID"] for c in cases}) == 929
    assert {c["Case ID"] for c in cases} == {f"TC{i:04}" for i in range(1, 930)}
    assert Counter(c["Source mode"] for c in cases) == {"Generated": 789, "Real Farm": 140}
    assert Counter((c["Source mode"], c["Priority"][:2]) for c in cases) == {
        ("Generated", "P0"): 22, ("Generated", "P1"): 86, ("Generated", "P2"): 681,
        ("Real Farm", "P0"): 20, ("Real Farm", "P1"): 50, ("Real Farm", "P2"): 70,
    }
    generated = [c for c in cases if c["Source mode"] == "Generated"]
    assert sum(dc.generated_preparation_issue(c) is not None for c in generated) == 99
    assert all(dc.generated_preparation_issue(c) is None for c in generated if c["Priority"].startswith("P0"))
    for c in generated:
        if dc.generated_preparation_issue(c) is None:
            for scene in ("grass_open", "agri_field_new", "desert_track", "rooftop_warehouse", "floating_water"):
                asset = ROOT / "assets/solar-renders" / scene / f"{c['Rows']}x{c['Columns / row']}.webp"
                legacy = ROOT / "assets/solar-renders" / f"{c['Rows']}x{c['Columns / row']}.webp"
                assert asset.is_file() or (scene == "grass_open" and legacy.is_file())


@pytest.mark.parametrize("mode,total,p0", [("Generated", 789, 22), ("Real Farm", 140, 20)])
def test_ui_filtering(mode, total, p0):
    at = app(mode)
    assert by_label(at.metric, "Planned cases").value == str(total)
    assert by_label(at.metric, "Essential complete").value == f"0/{p0}"
    cases = [c for c in dc._load_cases() if c["Source mode"] == mode]
    for priority in ("P0 – Essential", "P1 – Recommended", "P2 – Extended"):
        at.selectbox(key=f"pv_dc_priority_{mode}").select(priority).run()
        actual = at.selectbox(key=f"pv_dc_case_{mode}")
        assert len(actual.options) == sum(c["Priority"].startswith(priority[:2]) for c in cases)
        assert len(at.dataframe[0].value) == min(150, len(actual.options))
    at.selectbox(key=f"pv_dc_priority_{mode}").select("All priorities").run()
    scene = cases[0]["Scene / layout ID"]
    at.selectbox(key=f"pv_dc_scene_{mode}").select(scene).run()
    assert len(at.selectbox(key=f"pv_dc_case_{mode}").options) == sum(c["Scene / layout ID"] == scene for c in cases)
    at.selectbox(key=f"pv_dc_type_{mode}").select(cases[0]["Planned scenario"]).run()
    assert at.selectbox(key=f"pv_dc_case_{mode}").value == cases[0]["Case ID"]
    at.text_input(key=f"pv_dc_search_{mode}").input(cases[0]["Case ID"].lower()).run()
    assert len(at.selectbox(key=f"pv_dc_case_{mode}").options) == 1
    at.text_input(key=f"pv_dc_search_{mode}").input("no-such-case").run()
    assert any("No cases match" in x.value for x in at.info)
    assert not at.exception


def test_generated_preparation_return_and_unsupported_layouts():
    at = app()
    at.selectbox(key="pv_dc_priority_Generated").select("P0 – Essential").run()
    cid = at.selectbox(key="pv_dc_case_Generated").value
    renderer = at.selectbox(key="thermal_renderer_mode")
    renderer.select(renderer.options[1]).run()
    at.button(key="pv_dc_prepare_Generated").click().run()
    assert not at.exception
    assert at.query_params["section"] == ["Faults"]
    assert at.query_params["row_count"] == ["4"]
    assert at.query_params["panels_per_row"] == ["6"]
    assert at.session_state["thermal_renderer_mode"] == "Current Thermal"
    record = at.session_state["pv_dc_records_v1"][cid]
    assert record["Status"] == "Prepared"
    assert record["GT verified?"] == "No"
    assert "grass_open:4x6" in record["Prepared scene identity"]
    assert not at.session_state.filtered_state.get("random_fault_records")
    assert not at.session_state.filtered_state.get("panel_cell_fault_records")
    by_label(at.radio, "Workflow").set_value("DATA COLLECTION").run()
    assert by_label(at.metric, "Started").value == "1"
    at.selectbox(key="pv_dc_status_Generated").select("Prepared").run()
    assert at.selectbox(key="pv_dc_case_Generated").value == cid
    at.selectbox(key="pv_dc_status_Generated").select("All statuses").run()
    at.selectbox(key="pv_dc_priority_Generated").select("All priorities").run()
    for topology in ("Ragged edge", "Stress larger"):
        case = next(c for c in dc._load_cases() if c["Topology"] == topology)
        at.selectbox(key="pv_dc_case_Generated").select(case["Case ID"]).run()
        assert at.button(key="pv_dc_prepare_Generated").disabled
        assert at.warning
    assert not at.exception


def test_baseline_affected_cases_are_visibly_blocked_for_rooftop():
    at = app()
    at.selectbox(key="pv_dc_scene_Generated").select("G06x06").run()
    label = next(c["Planned scenario"] for c in dc._load_cases() if c["CLOSE diagnosis code"] == "SoilingDust")
    at.selectbox(key="pv_dc_type_Generated").select(label).run()
    at.selectbox(key="pv_dc_generated_scene").select("rooftop_warehouse").run()
    assert len(at.selectbox(key="pv_dc_case_Generated").options) == 40
    assert set(at.dataframe[0].value["Preparation support"]) == {"Blocked: baseline failure"}
    assert at.button(key="pv_dc_prepare_Generated").disabled
    assert not at.exception


def test_real_preparation_uses_explicit_photo_and_keeps_logs():
    at = app("Real Farm")
    at.selectbox(key="pv_dc_priority_Real Farm").select("P0 – Essential").run()
    cid = at.selectbox(key="pv_dc_case_Real Farm").value
    assert len(at.selectbox(key="pv_dc_real_source_Real Farm").options) == 14
    at.selectbox(key="pv_dc_real_source_Real Farm").select("image_14").run()
    at.button(key="pv_dc_prepare_Real Farm").click().run()
    assert not at.exception
    assert at.session_state["real_farm_section"] == "Faults"
    state = at.session_state["real_farm_state"]
    assert state["prepared_catalog_id"] == "image_14"
    assert not state["camera_observations"]
    record = at.session_state["pv_dc_records_v1"][cid]
    assert state["source_rgb_pixel_sha256"] in record["Prepared scene identity"]
    assert record["GT verified?"] == "No"
    by_label(at.radio, "Real Farm workflow").set_value("DATA COLLECTION").run()
    assert at.session_state["pv_dc_records_v1"][cid] == record
    assert not at.exception


@pytest.mark.parametrize("mode,cid", [("Generated", "TC0007"), ("Real Farm", "TC0497")])
def test_ui_prepares_actual_fault_and_preserves_case_selection_state(mode, cid):
    at = app(mode)
    at.selectbox(key=f"pv_dc_case_{mode}").select(cid).run()
    if mode == "Generated":
        at.radio(key="pv_dc_phase_Generated").set_value("Close diagnosis").run()
    else:
        at.selectbox(key="pv_dc_real_source_Real Farm").select("image_01").run()
        modules = at.multiselect(key="pv_dc_real_modules_image_01")
        modules.select(modules.options[0]).run()
    at.button(key=f"pv_dc_prepare_{mode}").click().run()
    assert not at.exception
    if mode == "Generated":
        actual = at.session_state["panel_cell_fault_records"][0]
        assert actual["fault_type"] == "CellCracking" and actual["fault_scale"] == 3
        assert actual["module_number_in_panel"] == 3 and actual["module_cell_number"] == 33
    else:
        actual = at.session_state["real_farm_state"]["camera_observations"][0]
        assert actual["severity"] == 6 and actual["seed"] == 62496
        assert actual["diagnostic_fault_type"] is None
    navigation = "Workflow" if mode == "Generated" else "Real Farm workflow"
    by_label(at.radio, navigation).set_value("DATA COLLECTION").run()
    at.selectbox(key=f"pv_dc_case_{mode}").select("TC0001" if mode == "Generated" else "TC0496").run()
    if mode == "Generated":
        assert at.session_state["panel_cell_fault_records"][0] == actual
    else:
        assert at.session_state["real_farm_state"]["camera_observations"][0] == actual
    assert not at.exception


@pytest.mark.parametrize("mode", ["Generated", "Real Farm"])
def test_existing_workflow_sections_still_open(mode):
    at = app(mode)
    if mode == "Real Farm":
        at.selectbox(key="pv_dc_real_source_Real Farm").select("image_14").run()
    at.button(key=f"pv_dc_prepare_{mode}").click().run()
    assert not at.exception
    label = "Workflow" if mode == "Generated" else "Real Farm workflow"
    sections = (["Scene Setup", "Layout", "Faults", "Review", "Export", "Settings", "Help"]
                if mode == "Generated" else ["Farm Setup", "Thermal", "Faults", "Review", "Export"])
    for section in sections:
        by_label(at.radio, label).set_value(section).run()
        assert not at.exception, section
    if mode == "Real Farm":
        assert not at.session_state["real_farm_state"].get("geometry_confirmation")


@pytest.mark.parametrize("mode,p0", [("Generated", 22), ("Real Farm", 20)])
def test_log_completion_guard_and_csv_downloads(mode, p0):
    at = app(mode)
    if mode == "Real Farm":
        at.selectbox(key="pv_dc_real_source_Real Farm").select("image_01").run()
    case = next(c for c in dc._load_cases() if c["Source mode"] == mode and c["Priority"].startswith("P0") and c["CLOSE required?"] == "Yes")
    cid = case["Case ID"]
    at.selectbox(key=f"pv_dc_case_{mode}").select(cid).run()
    by_label(at.selectbox, "Test status").select("Complete")
    by_label(at.button, "Save case record").click().run()
    assert any("Complete requires" in x.value for x in at.error)
    assert cid not in at.session_state["pv_dc_records_v1"]
    for checkbox in at.checkbox:
        if checkbox.label in {"FAR recorded", "CLOSE recorded", "Clean exports saved", "Ground truth verified"}:
            checkbox.check()
    for label, value in {"Recorded observation IDs": "OBS-test", "Prepared scene identity": "operator checked scene",
                         "FAR video filename": "far.mp4", "CLOSE video filename": "close.mp4",
                         "FAR clean reference": "far.png", "CLOSE clean reference": "close.png",
                         "Dataset / metadata ZIP": "metadata.zip"}.items():
        by_label(at.text_input, label).input(value)
    by_label(at.text_area, "Notes / exceptions").input('Test log, with "quotes"\nand newline')
    by_label(at.button, "Save case record").click().run()
    assert not at.exception
    record = at.session_state["pv_dc_records_v1"][cid]
    assert record["Status"] == "Complete"
    assert by_label(at.metric, "Completed").value == "1"
    assert by_label(at.metric, "Essential complete").value == f"1/{p0}"
    downloads = at.get("download_button")
    assert {x.proto.label for x in downloads} == {"Download progress JSON", "Export entire plan + tracking CSV (this mode)"}
    assert all(x.proto.url for x in downloads)


def test_backup_restore_csv_and_atomic_rejection(records):
    cases = dc._load_cases()
    generated = next(c for c in cases if c["Source mode"] == "Generated")
    real = next(c for c in cases if c["Source mode"] == "Real Farm")
    record = {**dc._blank_record(), "Status": "Complete", "FAR captured?": "Yes", "CLOSE captured?": "Yes",
              "Exports saved?": "Yes", "GT verified?": "Yes", "Notes / exceptions": 'comma, quote "\n' + "x" * 2500}
    records["pv_dc_records_v1"] = {generated["Case ID"]: record, real["Case ID"]: {**record, "Status": "Needs review"}}
    original = json.loads(dc._backup_json())
    records["pv_dc_records_v1"] = {}
    upload = SimpleNamespace(getvalue=lambda: json.dumps(original).encode())
    assert dc._restore_json(upload, {c["Case ID"] for c in cases}) == 2
    assert records["pv_dc_records_v1"] == original["records"]
    assert records["unrelated_fault_state"] == {"keep": True}
    for mode, total in (("Generated", 789), ("Real Farm", 140)):
        rows = [{**c, **dc._current_record(c["Case ID"])} for c in cases if c["Source mode"] == mode]
        exported = list(csv.DictReader(io.StringIO(dc._as_csv(rows, dc.CASE_FIELDS + dc.TRACK_FIELDS).decode("utf-8-sig"))))
        assert len(exported) == total
        cid = generated["Case ID"] if mode == "Generated" else real["Case ID"]
        assert next(r for r in exported if r["Case ID"] == cid)["Notes / exceptions"] == record["Notes / exceptions"]
    snapshot = dict(records["pv_dc_records_v1"])
    for invalid in (b"not JSON", b"{}", b'{"schema":"pv_dc_records_v1","records":[]}',
                    json.dumps({"schema": "pv_dc_records_v1", "records": {generated["Case ID"]: {"Status": "Complete"}}}).encode(),
                    json.dumps({"schema": "pv_dc_records_v1", "records": {generated["Case ID"]: {"Status": "invented"}}}).encode(),
                    json.dumps({"schema": "pv_dc_records_v1", "records": {generated["Case ID"]: {"GT verified?": True}}}).encode()):
        with pytest.raises(ValueError):
            dc._restore_json(SimpleNamespace(getvalue=lambda: invalid), set(snapshot))
        assert records["pv_dc_records_v1"] == snapshot


def test_preparation_failure_does_not_claim_prepared_and_source_change_requires_review(records):
    case = dc._load_cases()[0]
    def fail(*args):
        raise ValueError("Unavailable")
    dc._prepare_case(case, None, fail)
    assert dc._current_record(case["Case ID"])["Status"] == "Blocked"
    assert records["pv_dc_prepare_error"] == "Unavailable"
    records["pv_dc_records_v1"] = {case["Case ID"]: {**dc._blank_record(), "Status": "Complete", "GT verified?": "Yes",
                                                    "Prepared scene identity": "old", "Notes / exceptions": "keep"}}
    dc._prepare_case(case, None, lambda *args: "new")
    rec = dc._current_record(case["Case ID"])
    assert rec["Status"] == "Needs review" and rec["GT verified?"] == "No"
    assert rec["Notes / exceptions"] == "keep"


def test_all_14_prepared_sources_identity_assets_geometry_and_loader():
    ns = runpy.run_path(str(ROOT / "app.py"), run_name="pv_dc_catalog_test")
    entries = ns["real_farm_catalog_entries"]()
    assert len(entries) == 14
    session = {"pv_dc_records_v1": {"keep": "log"}, "real_farm_section": "DATA COLLECTION"}
    ns["real_farm_catalog_selected"].__globals__["st"] = SimpleNamespace(session_state=session)
    for entry in entries:
        for field, digest in (("rgb", "rgb_file_sha256"), ("project", "project_sha256"), ("pv_mask", "pv_mask_sha256"),
                              ("thermal", "thermal_sha256"), ("ignore_mask", "ignore_mask_sha256")):
            path = ns["real_farm_catalog_asset"](entry[field], entry[digest])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[digest]
        image = ImageOps.exif_transpose(Image.open(ns["real_farm_catalog_asset"](entry["rgb"]))).convert("RGB")
        assert ns["real_farm_source_identity"](image) == entry["rgb_pixel_sha256"]
        session["real_farm_prepared_picker"] = entry["id"]
        ns["real_farm_catalog_selected"]()
        state = session["real_farm_state"]
        assert state["prepared_catalog_id"] == entry["id"]
        assert len(state["modules"]) == entry["module_count"]
        assert not state["camera_observations"]
        assert session["pv_dc_records_v1"] == {"keep": "log"}
