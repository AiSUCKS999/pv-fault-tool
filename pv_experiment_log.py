"""Session experiment journal. Assertions by the operator are not physical verification."""
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import re
import uuid
import zipfile

from PIL import Image

SCHEMA = "pv_experiments_v2"
KEY = "pv_dc_experiments_v2"
BLOBS = "pv_dc_reference_bytes"
VIEWS = ("FAR", "CLOSE")


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def tag(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:70] or "scene"


def changes(before, after):
    return {key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}


def video_metadata(value):
    """Accept only small browser-provided metadata, never file bytes or paths."""
    if not isinstance(value, dict):
        raise ValueError("Invalid video metadata")
    name = value.get("name")
    if not isinstance(name, str) or not name or len(name) > 512 or any(x in name for x in ("/", "\\", "\x00")):
        raise ValueError("Invalid video filename")
    result = {"name": name, "selection_id": str(value.get("selection_id", ""))[:100],
              "mime": str(value.get("mime", ""))[:100], "selected_utc": utc(),
              "metadata_source": "operator-selected browser File; contents not uploaded",
              "automatically_verified_recording": False}
    for key in ("size_bytes", "last_modified_ms", "duration_seconds", "width", "height"):
        item = value.get(key)
        if item is not None:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
                raise ValueError(f"Invalid video {key}")
            result[key] = item
    result["metadata_error"] = str(value.get("metadata_error", ""))[:300]
    return result


class Journal:
    def __init__(self, session):
        self.session = session
        self.data = session.setdefault(KEY, {"schema": SCHEMA, "experiments": {}, "active_by_mode": {}})
        self.blobs = session.setdefault(BLOBS, {})

    def event(self, exp, action, **fields):
        exp["events"].append({"event_id": uuid.uuid4().hex, "utc": utc(), "action": action, **deepcopy(fields)})
        exp["updated_utc"] = utc()

    def start(self, case):
        eid = "exp-" + uuid.uuid4().hex
        exp = {"experiment_id": eid, "case_id": case["Case ID"], "mode": case["Source mode"],
               "plan": deepcopy(case), "created_utc": utc(), "updated_utc": utc(), "status": "Not prepared",
               "active_stage": "FAR", "events": [], "blocked_reason": "", "notes": "",
               "physical_ground_truth_verified": False, "yolo_detection_verified": False,
               "stages": {view: {"preparations": [], "references": [], "capture_confirmations": [], "video": None}
                          for view in VIEWS}}
        self.data["experiments"][eid] = exp
        self.data["active_by_mode"][exp["mode"]] = eid
        self.event(exp, "experiment_started")
        return exp

    def get(self, eid):
        return self.data["experiments"][eid]

    def active(self, mode, case_id=None):
        exp = self.data["experiments"].get(self.data["active_by_mode"].get(mode))
        if exp and (case_id is None or exp["case_id"] == case_id):
            return exp
        if case_id is not None:
            return next((e for e in reversed(list(self.data["experiments"].values()))
                         if e["mode"] == mode and e["case_id"] == case_id), None)
        return None

    def prepared(self, exp, view, actual, assessment):
        if view not in VIEWS:
            raise ValueError("Invalid acquisition stage")
        if actual.get("mode") != exp["mode"]:
            raise ValueError("Simulator mode does not match this experiment")
        if exp.get("finished_utc"):
            raise ValueError("Start a new experiment before preparing a finished test")
        receipt = {"preparation_id": uuid.uuid4().hex, "utc": utc(), "actual": deepcopy(actual),
                   "state_fingerprint": fingerprint(actual), "assessment": deepcopy(assessment)}
        if view == "CLOSE" and exp["stages"]["FAR"]["preparations"]:
            receipt["changes_from_far"] = changes(exp["stages"]["FAR"]["preparations"][-1]["actual"], actual)
        exp["stages"][view]["preparations"].append(receipt)
        exp["active_stage"] = view
        exp["blocked_reason"] = ""
        exp["status"] = f"{view} prepared" if assessment["configuration_matches_plan"] else f"{view}: review actual setup"
        self.event(exp, "simulator_state_recorded", view=view, preparation_id=receipt["preparation_id"])
        return receipt

    def fail(self, exp, action, error):
        exp["blocked_reason"] = str(error)
        exp["status"] = f"Blocked: {action}"
        self.event(exp, "action_failed", operation=action, reason=str(error))

    def reference(self, exp, view, actual, png, parameters):
        if exp.get("finished_utc"):
            raise ValueError("Start a new experiment before exporting a new reference")
        if view not in VIEWS or not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Reference export did not return a PNG")
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            size = list(image.size)
        stage = exp["stages"][view]
        revision = len(stage["references"]) + 1
        rid = "ref-" + uuid.uuid4().hex
        state_hash = fingerprint(actual)
        if not stage["preparations"] or stage["preparations"][-1]["state_fingerprint"] != state_hash:
            raise ValueError("Reference state does not match the latest preparation")
        scene = actual["scene_identity"]
        filename = f"{exp['case_id']}_{view}_{tag(scene)}_r{revision:04}_{exp['experiment_id'][4:]}_{rid[4:]}.png"
        ref = {"reference_id": rid, "experiment_id": exp["experiment_id"], "view": view,
               "revision": revision, "filename": filename, "sha256": hashlib.sha256(png).hexdigest(),
               "size_bytes": len(png), "image_size": size, "created_utc": utc(),
               "state_fingerprint": state_hash, "actual": deepcopy(actual), "parameters": deepcopy(parameters),
               "server_export_verified": True, "export_ready_utc": utc(),
               "download_requested_utc": None, "file_saved_confirmed_utc": None}
        self.blobs[rid] = png
        stage["references"].append(ref)
        stage["current_reference_id"] = rid
        self.event(exp, "reference_generated", view=view, reference_id=rid, sha256=ref["sha256"])
        return ref

    def current_reference(self, exp, view):
        stage = exp["stages"][view]
        return next((r for r in stage["references"] if r["reference_id"] == stage.get("current_reference_id")), None)

    def request_download(self, exp, view, actual=None):
        ref = self.current_reference(exp, view)
        if not ref or ref["reference_id"] not in self.blobs:
            raise ValueError("Reference bytes are unavailable; restore the reference bundle or regenerate from the simulator")
        if actual is not None and fingerprint(actual) != ref["state_fingerprint"]:
            raise ValueError("Simulator state changed. Refresh the reference before recording.")
        if hashlib.sha256(self.blobs[ref["reference_id"]]).hexdigest() != ref["sha256"]:
            raise ValueError("Reference integrity check failed")
        ref["download_requested_utc"] = utc()
        exp["blocked_reason"] = ""
        self.event(exp, "reference_download_requested", view=view, reference_id=ref["reference_id"])

    def confirm_capture(self, exp, view, actual):
        if exp.get("finished_utc"):
            raise ValueError("This experiment is already finished")
        ref = self.current_reference(exp, view)
        if not ref or not ref["export_ready_utc"]:
            raise ValueError("Save this stage's reference first")
        if fingerprint(actual) != ref["state_fingerprint"]:
            raise ValueError("The simulator changed after this reference. Save a matching reference first.")
        confirmation = {"reference_id": ref["reference_id"], "utc": utc(), "operator_confirmed_recording": True,
                        "automatically_verified": False, "associated_video": deepcopy(exp["stages"][view]["video"])}
        exp["stages"][view]["capture_confirmations"].append(confirmation)
        exp["status"] = f"{view} recorded (operator confirmed)"
        exp["blocked_reason"] = ""
        self.event(exp, "external_recording_confirmed_by_operator", view=view, reference_id=ref["reference_id"])

    def confirm_file_saved(self, exp, view):
        ref = self.current_reference(exp, view)
        if not ref or not ref["export_ready_utc"]:
            raise ValueError("Save the reference first")
        ref["file_saved_confirmed_utc"] = utc()
        self.event(exp, "file_saved_confirmed_by_operator", view=view, reference_id=ref["reference_id"])

    def attach_video(self, exp, view, value):
        metadata = video_metadata(value)
        old = exp["stages"][view]["video"]
        if old and old["selection_id"] == metadata["selection_id"]:
            return
        exp["stages"][view]["video"] = metadata
        self.event(exp, "video_metadata_selected", view=view, metadata=metadata)

    def stage_complete(self, exp, view):
        stage = exp["stages"][view]
        ref = self.current_reference(exp, view)
        return bool(ref and ref["server_export_verified"] and ref["export_ready_utc"]
                    and any(c["reference_id"] == ref["reference_id"] and c["operator_confirmed_recording"]
                            for c in stage["capture_confirmations"]))

    def reference_complete(self, exp, view):
        stage = exp["stages"][view]
        ref = self.current_reference(exp, view)
        return bool(ref and ref["server_export_verified"] and ref["export_ready_utc"]
                    and ref["reference_id"] in self.blobs and stage["preparations"]
                    and stage["preparations"][-1]["state_fingerprint"] == ref["state_fingerprint"])

    def confirm_archived_capture(self, exp, view):
        """Optional explicit operator statement about the immutable saved reference."""
        ref = self.current_reference(exp, view)
        if not ref or not ref["export_ready_utc"]:
            raise ValueError("Collect this stage's reference first")
        exp["stages"][view]["capture_confirmations"].append({"reference_id": ref["reference_id"], "utc": utc(),
            "operator_confirmed_recording": True, "automatically_verified": False,
            "scope": "operator associates an external recording with this archived reference",
            "associated_video": deepcopy(exp["stages"][view]["video"])})
        self.event(exp, "archived_reference_recording_confirmed_by_operator", view=view, reference_id=ref["reference_id"])

    def finish(self, exp):
        return self._finish(exp, require_recording=True)

    def finish_simulator(self, exp):
        return self._finish(exp, require_recording=False)

    def _finish(self, exp, require_recording):
        if exp["blocked_reason"]:
            raise ValueError(exp["blocked_reason"])
        required = ["FAR"] + (["CLOSE"] if exp["plan"]["CLOSE required?"] == "Yes" else [])
        for view in required:
            stage = exp["stages"][view]
            if require_recording and not self.stage_complete(exp, view):
                raise ValueError(f"Save the {view} reference and confirm the external recording first")
            ref = self.current_reference(exp, view)
            if not ref or not ref["server_export_verified"] or not ref["export_ready_utc"]:
                raise ValueError(f"Collect the {view} reference first")
            if ref["reference_id"] not in self.blobs:
                raise ValueError(f"Restore the reference ZIP to recover the {view} image before finishing")
            if hashlib.sha256(self.blobs[ref["reference_id"]]).hexdigest() != ref["sha256"]:
                raise ValueError(f"{view} reference integrity check failed")
            if not stage["preparations"] or stage["preparations"][-1]["state_fingerprint"] != ref["state_fingerprint"]:
                raise ValueError(f"{view} reference does not match its preparation")
            if (not stage["preparations"][-1]["assessment"]["configuration_matches_plan"]
                    and stage.get("exception_accepted_fingerprint") != ref["state_fingerprint"]):
                raise ValueError(f"Review the {view} setup difference in Advanced before finishing")
        exp["finished_utc"] = utc()
        exp["completion_scope"] = "operator_capture" if require_recording else "simulator_only"
        exp["status"] = "Finished (operator capture confirmation)" if require_recording else "Simulator collection complete"
        self.event(exp, "experiment_finished", completion_scope=exp["completion_scope"],
                   physical_ground_truth_verified=False, yolo_detection_verified=False)

    def backup(self):
        return json.dumps({**self.data, "created_utc": utc(), "reference_bytes_included": False},
                          ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")

    def csv(self):
        out = io.StringIO(newline="")
        fields = ["experiment_id", "case_id", "priority", "mode", "status", "created_utc", "finished_utc", "notes", "view",
                  "actual_json", "preparation_json", "references_json", "capture_confirmations_json", "video_json", "blocked_reason"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for exp in self.data["experiments"].values():
            for view in VIEWS:
                stage = exp["stages"][view]
                writer.writerow({"experiment_id": exp["experiment_id"], "case_id": exp["case_id"], "priority": exp["plan"]["Priority"],
                    "mode": exp["mode"], "status": exp["status"], "created_utc": exp["created_utc"], "finished_utc": exp.get("finished_utc", ""),
                    "notes": exp["notes"], "view": view, "actual_json": canonical(stage["preparations"][-1]["actual"] if stage["preparations"] else {}),
                    "preparation_json": canonical(stage["preparations"]), "references_json": canonical(stage["references"]),
                    "capture_confirmations_json": canonical(stage["capture_confirmations"]), "video_json": canonical(stage["video"]),
                    "blocked_reason": exp["blocked_reason"]})
        return out.getvalue().encode("utf-8-sig")

    def bundle(self):
        out = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("experiment_log.json", self.backup())
            archive.writestr("experiment_log.csv", self.csv())
            for exp in self.data["experiments"].values():
                for view in VIEWS:
                    for ref in exp["stages"][view]["references"]:
                        data = self.blobs.get(ref["reference_id"])
                        available = data is not None and hashlib.sha256(data).hexdigest() == ref["sha256"]
                        path = "references/" + ref["filename"]
                        manifest.append({"reference_id": ref["reference_id"], "path": path, "sha256": ref["sha256"], "included": available})
                        if available:
                            archive.writestr(path, data)
            archive.writestr("reference_manifest.json", canonical(manifest))
        return out.getvalue()

    def restore(self, raw, cases, is_zip=False):
        try:
            return self._restore(raw, cases, is_zip)
        except (KeyError, TypeError, AttributeError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid experiment backup: {exc}") from exc

    def _restore(self, raw, cases, is_zip=False):
        """Validate everything before replacing collection state; never touch simulator state."""
        blobs = {}
        if is_zip:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if len(archive.infolist()) > 10000 or sum(i.file_size for i in archive.infolist()) > 256 * 1024 * 1024:
                    raise ValueError("Reference bundle exceeds the restore limit")
                if len(set(archive.namelist())) != len(archive.namelist()):
                    raise ValueError("Duplicate reference bundle members")
                data = json.loads(archive.read("experiment_log.json"))
                manifest = json.loads(archive.read("reference_manifest.json"))
                for item in manifest:
                    if item["included"]:
                        path = item["path"]
                        if not re.fullmatch(r"references/[A-Za-z0-9_-]+\.png", path):
                            raise ValueError("Invalid reference path")
                        value = archive.read(path)
                        if hashlib.sha256(value).hexdigest() != item["sha256"]:
                            raise ValueError("Reference bundle hash mismatch")
                        with Image.open(io.BytesIO(value)) as image:
                            image.verify()
                        blobs[item["reference_id"]] = value
        else:
            data = json.loads(raw.decode("utf-8-sig"))
        catalog = {c["Case ID"]: c for c in cases}
        if not isinstance(data, dict):
            raise ValueError("Not an experiment progress backup")
        if data.get("schema") == "pv_dc_records_v1":
            if not isinstance(data.get("records"), dict):
                raise ValueError("Invalid legacy progress records")
            temporary = Journal({})
            for cid, record in data["records"].items():
                if cid in catalog and isinstance(record, dict):
                    exp = temporary.start(catalog[cid])
                    exp.update(status="Imported legacy record — review", legacy_record=deepcopy(record))
            data = temporary.data
        if data.get("schema") != SCHEMA or not isinstance(data.get("experiments"), dict):
            raise ValueError("Not an experiment progress backup")
        data = deepcopy(data)
        all_reference_ids = set()
        for eid, exp in data["experiments"].items():
            if (not re.fullmatch(r"exp-[a-f0-9]{32}", eid) or not isinstance(exp, dict)
                    or exp.get("experiment_id") != eid or exp.get("case_id") not in catalog):
                raise ValueError("Invalid experiment identity")
            if exp.get("mode") != catalog[exp["case_id"]]["Source mode"] or exp.get("plan") != catalog[exp["case_id"]]:
                raise ValueError("Experiment plan does not match the original catalog")
            exp["physical_ground_truth_verified"] = False
            exp["yolo_detection_verified"] = False
            if not isinstance(exp.get("events"), list) or exp.get("active_stage") not in VIEWS:
                raise ValueError("Invalid experiment event/stage structure")
            for view in VIEWS:
                stage = exp["stages"][view]
                if not all(isinstance(stage.get(k), list) for k in ("preparations", "references", "capture_confirmations")):
                    raise ValueError("Invalid experiment stage")
                for receipt in stage["preparations"]:
                    if (not isinstance(receipt, dict) or receipt["actual"].get("mode") != exp["mode"]
                            or fingerprint(receipt["actual"]) != receipt["state_fingerprint"]
                            or not isinstance(receipt["assessment"].get("configuration_matches_plan"), bool)):
                        raise ValueError("Preparation state fingerprint mismatch")
                ids = set()
                for ref in stage["references"]:
                    if (ref["experiment_id"] != eid or ref["view"] != view or ref["reference_id"] in all_reference_ids
                            or fingerprint(ref["actual"]) != ref["state_fingerprint"]
                            or not re.fullmatch(r"[A-Za-z0-9_-]+\.png", ref["filename"])
                            or not re.fullmatch(r"[a-f0-9]{64}", ref["sha256"])
                            or ref.get("server_export_verified") is not True or not ref.get("export_ready_utc")):
                        raise ValueError("Invalid reference identity")
                    ids.add(ref["reference_id"])
                    all_reference_ids.add(ref["reference_id"])
                    if ref["reference_id"] in blobs and hashlib.sha256(blobs[ref["reference_id"]]).hexdigest() != ref["sha256"]:
                        raise ValueError("Reference log/hash mismatch")
                if stage.get("current_reference_id") and stage["current_reference_id"] not in ids:
                    raise ValueError("Unknown current reference")
                if any(c.get("reference_id") not in ids or c.get("operator_confirmed_recording") is not True
                       or c.get("automatically_verified") is not False for c in stage["capture_confirmations"]):
                    raise ValueError("Invalid recording confirmation")
                if stage.get("video"):
                    video_metadata(stage["video"])
            if exp.get("finished_utc"):
                validator = Journal({KEY: {"schema": SCHEMA, "experiments": {eid: exp}, "active_by_mode": {}}})
                for view in ["FAR"] + (["CLOSE"] if exp["plan"]["CLOSE required?"] == "Yes" else []):
                    ref = validator.current_reference(exp, view)
                    if (exp.get("completion_scope", "operator_capture") != "simulator_only" and not validator.stage_complete(exp, view)):
                        raise ValueError("Finished experiment is missing required evidence")
                    stage = exp["stages"][view]
                    if (not ref or not ref["server_export_verified"] or not ref["export_ready_utc"]
                            or not stage["preparations"] or stage["preparations"][-1]["state_fingerprint"] != ref["state_fingerprint"]
                            or (not stage["preparations"][-1]["assessment"]["configuration_matches_plan"]
                                and stage.get("exception_accepted_fingerprint") != ref["state_fingerprint"])):
                        raise ValueError("Finished experiment has mismatched preparation evidence")
        active = data.get("active_by_mode", {})
        if not isinstance(active, dict) or any(mode not in {"Generated", "Real Farm"} or eid not in data["experiments"]
                or data["experiments"][eid]["mode"] != mode for mode, eid in active.items()):
            raise ValueError("Invalid active experiment map")
        selected = data.get("selected_case_by_mode", {})
        if not isinstance(selected, dict) or any(mode not in {"Generated", "Real Farm"} or cid not in catalog
                or catalog[cid]["Source mode"] != mode for mode, cid in selected.items()):
            raise ValueError("Invalid selected case map")
        slots = data.get("photo_slots", {})
        if not isinstance(slots, dict) or any(not re.fullmatch(r"RF(?:0[1-9]|1[0-4])", key)
                or value not in {f"image_{i:02}" for i in range(1, 15)} for key, value in slots.items()):
            raise ValueError("Invalid prepared photo mapping")
        self.session[KEY] = data
        self.session[BLOBS] = blobs
        self.data, self.blobs = data, blobs
        return len(data["experiments"])
