"""Evidence semantics, transactional restore and metadata-only video handling."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pv_data_collection import _load_cases
from pv_experiment_log import Journal, KEY, BLOBS, fingerprint, video_metadata


def case(cid="TC0007"):
    return next(c for c in _load_cases() if c["Case ID"] == cid)


def actual(view="FAR", severity=3):
    return {"mode": "Generated", "scene_identity": "grass_open", "source_image": {"name": "1x1.webp", "sha256": "a" * 64},
            "rows": 1, "columns": 1, "selection": {"row": 1, "array": 1, "module": 3, "cell": 33},
            "faults": [{"fault_type": "SingleHotSpot" if view == "FAR" else "CellCracking", "severity": severity}]}


def png():
    out = io.BytesIO()
    Image.new("RGB", (13, 11), (28, 91, 142)).save(out, "PNG")
    return out.getvalue()


def stage(journal, exp, view="FAR", checked=True, state=None):
    state = state or actual(view)
    journal.prepared(exp, view, state, {"configuration_matches_plan": checked, "rendered_synthetic_effect_verified": True})
    return journal.reference(exp, view, state, png(), {"view": view, "overlays": False})


def test_button_events_save_without_a_form_and_do_not_claim_external_facts():
    session = {}
    j = Journal(session)
    exp = j.start(case())
    eid = exp["experiment_id"]
    for view in ("FAR", "CLOSE"):
        ref = stage(j, exp, view)
        assert ref["server_export_verified"] and ref["export_ready_utc"]
        assert not ref["download_requested_utc"] and not ref["file_saved_confirmed_utc"]
        assert not j.stage_complete(exp, view)
        j.confirm_capture(exp, view, actual(view))
        assert j.stage_complete(exp, view)
        assert exp["stages"][view]["capture_confirmations"][-1]["automatically_verified"] is False
        assert session[KEY]["experiments"][eid] is exp
    j.finish(exp)
    assert exp["experiment_id"] == eid and exp["finished_utc"]
    assert not exp["physical_ground_truth_verified"] and not exp["yolo_detection_verified"]
    assert len(exp["events"]) >= 8
    j.request_download(exp, "FAR")
    assert j.current_reference(exp, "FAR")["download_requested_utc"]
    assert not j.current_reference(exp, "FAR")["file_saved_confirmed_utc"]
    j.confirm_file_saved(exp, "FAR")
    assert j.current_reference(exp, "FAR")["file_saved_confirmed_utc"]


def test_unique_filenames_and_revisions_invalidate_old_capture_confirmation():
    j = Journal({})
    exp = j.start(case())
    first = stage(j, exp)
    j.confirm_capture(exp, "FAR", actual())
    second = stage(j, exp)
    assert first["filename"] != second["filename"]
    assert first["reference_id"] != second["reference_id"]
    assert (first["revision"], second["revision"]) == (1, 2)
    assert all(part in second["filename"] for part in ("TC0007", "FAR", "grass_open", "r0002"))
    assert first["sha256"] == hashlib.sha256(png()).hexdigest()
    assert not j.stage_complete(exp, "FAR")
    other = j.start(case())
    assert other["experiment_id"] != exp["experiment_id"]
    assert stage(j, other)["filename"] != first["filename"]


def test_wrong_state_cannot_be_confirmed_and_reference_is_immutable():
    j = Journal({})
    exp = j.start(case())
    ref = stage(j, exp)
    before = deepcopy(ref)
    with pytest.raises(ValueError, match="simulator changed"):
        j.confirm_capture(exp, "FAR", actual(severity=9))
    assert not exp["stages"]["FAR"]["capture_confirmations"] and ref == before
    with pytest.raises(ValueError, match="Simulator state changed"):
        j.request_download(exp, "FAR", actual(severity=9))
    assert ref["download_requested_utc"] is None


def test_failed_png_export_has_no_reference_success_or_capture_confirmation():
    j = Journal({})
    exp = j.start(case())
    j.prepared(exp, "FAR", actual(), {"configuration_matches_plan": True})
    for data in (b"not PNG", b"\x89PNG\r\n\x1a\nbroken"):
        with pytest.raises((ValueError, OSError)):
            j.reference(exp, "FAR", actual(), data, {})
        assert not exp["stages"]["FAR"]["references"] and not j.blobs
    with pytest.raises(ValueError, match="reference first"):
        j.confirm_capture(exp, "FAR", actual())
    j.fail(exp, "export", "Exporter failed")
    with pytest.raises(ValueError, match="Exporter failed"):
        j.finish(exp)


def test_finish_requires_both_stages_and_exception_is_bound_to_actual_state():
    j = Journal({})
    exp = j.start(case())
    stage(j, exp)
    j.confirm_capture(exp, "FAR", actual())
    with pytest.raises(ValueError, match="CLOSE"):
        j.finish(exp)
    ref = stage(j, exp, "CLOSE", checked=False, state=actual("CLOSE", 8))
    j.confirm_capture(exp, "CLOSE", actual("CLOSE", 8))
    exp["stages"]["CLOSE"]["exception_accepted_fingerprint"] = fingerprint(actual("CLOSE", 3))
    with pytest.raises(ValueError, match="Advanced"):
        j.finish(exp)
    exp["stages"]["CLOSE"]["exception_accepted_fingerprint"] = ref["state_fingerprint"]
    j.finish(exp)
    assert exp["finished_utc"] and not exp["physical_ground_truth_verified"]


def test_stage_changes_are_recorded_under_the_same_experiment_id():
    j = Journal({})
    exp = j.start(case())
    stage(j, exp)
    stage(j, exp, "CLOSE")
    receipt = exp["stages"]["CLOSE"]["preparations"][-1]
    assert receipt["changes_from_far"]["faults"]["before"][0]["fault_type"] == "SingleHotSpot"
    assert receipt["changes_from_far"]["faults"]["after"][0]["fault_type"] == "CellCracking"
    assert {r["experiment_id"] for s in exp["stages"].values() for r in s["references"]} == {exp["experiment_id"]}


def test_json_and_zip_restore_survive_session_reset_without_altering_simulator():
    j = Journal({})
    exp = j.start(case("TC0002"))
    stage(j, exp)
    j.confirm_capture(exp, "FAR", actual())
    exp["notes"] = 'optional, "quoted"\nnotes'
    data, bundle = j.backup(), j.bundle()
    session = {"random_fault_records": [{"keep": True}], "real_farm_state": {"geometry": "untouched"}}
    restored = Journal(session)
    assert restored.restore(data, _load_cases()) == 1
    assert restored.get(exp["experiment_id"]) == exp
    assert not restored.blobs
    with pytest.raises(ValueError, match="reference ZIP"):
        restored.finish(restored.get(exp["experiment_id"]))
    assert restored.restore(bundle, _load_cases(), is_zip=True) == 1
    assert restored.blobs == j.blobs
    restored.finish(restored.get(exp["experiment_id"]))
    assert session["random_fault_records"] == [{"keep": True}]
    assert session["real_farm_state"] == {"geometry": "untouched"}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read("reference_manifest.json"))
        assert all(item["included"] for item in manifest)
        for item in manifest:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
        assert b"optional" in archive.read("experiment_log.csv")


@pytest.mark.parametrize("corruption", ["plan", "stage", "reference_id", "fingerprint", "capture", "finished", "schema", "type"])
def test_malformed_restore_is_atomic(corruption):
    j = Journal({})
    exp = j.start(case())
    ref = stage(j, exp)
    original = deepcopy(j.data)
    data = json.loads(j.backup())
    e = data["experiments"][exp["experiment_id"]]
    if corruption == "plan": e["plan"]["Priority"] = "P0 fake"
    if corruption == "stage": e["stages"]["CLOSE"] = []
    if corruption == "reference_id": e["stages"]["FAR"]["current_reference_id"] = "foreign"
    if corruption == "fingerprint": e["stages"]["FAR"]["references"][0]["actual"]["rows"] = 99
    if corruption == "capture": e["stages"]["FAR"]["capture_confirmations"] = [{"reference_id": ref["reference_id"], "operator_confirmed_recording": True, "automatically_verified": True}]
    if corruption == "finished": e["finished_utc"] = "invented"
    if corruption == "schema": data["schema"] = "wrong"
    if corruption == "type": data = []
    with pytest.raises(ValueError):
        j.restore(json.dumps(data).encode(), _load_cases())
    assert j.data == original and j.blobs


def test_corrupt_zip_reference_cannot_be_imported():
    j = Journal({})
    exp = j.start(case())
    stage(j, exp)
    good = j.bundle()
    bad = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(good)) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"corrupt" if name.startswith("references/") else source.read(name))
    before = deepcopy(j.data)
    with pytest.raises(ValueError, match="hash mismatch"):
        j.restore(bad.getvalue(), _load_cases(), True)
    assert j.data == before


def test_legacy_progress_import_retains_notes_without_inventing_actual_evidence():
    j = Journal({})
    raw = json.dumps({"schema": "pv_dc_records_v1", "records": {"TC0007": {"Status": "Complete", "GT verified?": "Yes", "Notes / exceptions": "old notes"}}}).encode()
    assert j.restore(raw, _load_cases()) == 1
    exp = j.active("Generated")
    assert exp["legacy_record"]["Notes / exceptions"] == "old notes"
    assert not exp["stages"]["FAR"]["preparations"] and not exp["physical_ground_truth_verified"]
    assert not exp.get("finished_utc")


def test_video_metadata_does_not_upload_content_or_confirm_a_recording():
    value = {"name": "flight_0009.mp4", "selection_id": "browser-selection", "mime": "video/mp4",
             "size_bytes": 20 * 1024**3, "duration_seconds": 132.5, "width": 3840, "height": 2160,
             "last_modified_ms": 1793200000000, "contents": object()}
    metadata = video_metadata(value)
    assert metadata["size_bytes"] == 20 * 1024**3 and "contents" not in metadata
    assert metadata["automatically_verified_recording"] is False
    j = Journal({})
    exp = j.start(case())
    j.attach_video(exp, "FAR", value)
    assert not j.stage_complete(exp, "FAR")
    assert exp["stages"]["FAR"]["video"]["name"] == "flight_0009.mp4"
    events = len(exp["events"])
    j.attach_video(exp, "FAR", value)
    assert len(exp["events"]) == events
    with pytest.raises(ValueError): video_metadata({"name": "../wrong.mp4"})
    with pytest.raises(ValueError): video_metadata({"name": "flight.mp4", "size_bytes": float("nan")})


def test_restore_callback_reselects_progress_without_replacing_the_simulator(monkeypatch):
    import pv_experiment_ui as ui
    original = Journal({})
    exp = original.start(case())
    stage(original, exp)
    original.data["selected_case_by_mode"] = {"Generated": "TC0007"}
    raw = original.bundle()
    session = {"random_fault_records": [{"do not replace": True}],
               "pv_dc_restore_Generated": SimpleNamespace(name="progress.zip", getvalue=lambda: raw)}
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state=session))
    ui.restore_progress("Generated")
    assert session["random_fault_records"] == [{"do not replace": True}]
    assert session["pv_dc_case_Generated"] == "TC0007"
    assert Journal(session).active("Generated")["experiment_id"] == exp["experiment_id"]


def test_simulator_collection_never_requires_or_claims_drone_recording():
    j = Journal({})
    exp = j.start(case())
    for view in ("FAR", "CLOSE"):
        stage(j, exp, view)
    j.finish_simulator(exp)
    assert exp["status"] == "Simulator collection complete"
    assert exp["completion_scope"] == "simulator_only"
    assert all(s["capture_confirmations"] == [] and s["video"] is None for s in exp["stages"].values())
    assert not exp["physical_ground_truth_verified"] and not exp["yolo_detection_verified"]
    recovered = Journal({})
    assert recovered.restore(j.bundle(), _load_cases(), True) == 1
    restored = recovered.get(exp["experiment_id"])
    assert restored == exp
    assert recovered.reference_complete(restored, "FAR") and recovered.reference_complete(restored, "CLOSE")
    assert not recovered.stage_complete(restored, "FAR")  # No operator/drone confirmation was invented.
