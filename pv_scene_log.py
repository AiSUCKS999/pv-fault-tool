"""Scene journal, exact clean reference exports and transactional session restore."""
from copy import deepcopy
import csv
import hashlib
import io
import json
import re
import uuid
import zipfile

from PIL import Image

from pv_experiment_log import Journal, utc, tag
from pv_scene_generate import simulator_snapshot, restore_simulator

KEY="pv_scene_journal_v1"
BLOBS="pv_scene_reference_blobs"


def json_bytes(value):
    return json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda a:a.tolist() if hasattr(a,"tolist") else str(a)).encode("utf-8")


class SceneJournal:
    def __init__(self,session):
        self.session=session
        self.data=session.setdefault(KEY,{"schema":KEY,"experiments":{},"active":{}})
        self.blobs=session.setdefault(BLOBS,{})

    def active(self,mode):
        return self.data["experiments"].get(self.data["active"].get(mode))

    def capture(self,ns,acquired,request=None,new=False):
        """Encode every image first. Failed encodes never become saved references."""
        mode=acquired["mode"];old=None if new else self.active(mode)
        if old and old["revisions"][-1]["signature"]==acquired["signature"]:return old
        eid=old["experiment_id"] if old else "scene-"+uuid.uuid4().hex
        revision=len(old["revisions"])+1 if old else 1
        source=tag(acquired["source"]["identity"]);prefix=f"{eid}/{source}_r{revision:03}"
        blobs={};refs=[]
        for stage,frame in [("FAR",acquired["far"]),*[("CLOSE_"+tag(oid),f) for oid,f in acquired["close"].items()]]:
            for modality in ("original","rgb","thermal","healthy_thermal"):
                if modality not in frame:continue
                filename=f"{prefix}_{stage}_{modality}.png"
                png=ns["export_current_view_bytes"](frame[modality])
                with Image.open(io.BytesIO(png)) as decoded:
                    decoded.load()
                    if decoded.size!=frame[modality].size or decoded.convert("RGB").tobytes()!=frame[modality].convert("RGB").tobytes():
                        raise ValueError("Reference export differs from the acquired image")
                blobs[filename]=png
                refs.append({"filename":filename,"sha256":hashlib.sha256(png).hexdigest(),"stage":stage,
                    "modality":modality,"size":list(frame[modality].size),"crop_bounds":frame["bounds"],
                    "annotations":False,"encoded_verified":True,"download_saved_confirmed":False})
        maskname=prefix+"_PV_mask.png";blobs[maskname]=ns["export_current_view_bytes"](acquired["mask"])
        refs.append({"filename":maskname,"sha256":hashlib.sha256(blobs[maskname]).hexdigest(),"stage":"FAR","modality":"mask",
                     "encoded_verified":True,"download_saved_confirmed":False})
        snapshot=simulator_snapshot(ns)
        snapshot["state"].pop("real_farm_image",None)
        snapshot["state"].pop("real_farm_image_path",None)
        snapshot=json.loads(json_bytes(snapshot))
        item={"revision":revision,"created_utc":utc(),"signature":acquired["signature"],"source":deepcopy(acquired["source"]),
              "faults":deepcopy(acquired["faults"]),"geometry":deepcopy(acquired["geometry"]),"references":refs,
              "rgb_report":deepcopy(acquired["rgb_report"]),"simulator":snapshot,
              "geometry_verified":False,"synthetic_only":True,"physical_ground_truth_verified":False}
        # JSON validation precedes any journal mutation too.
        item=json.loads(json_bytes(item))
        exp=deepcopy(old) if old else {"experiment_id":eid,"mode":mode,"created_utc":utc(),"revisions":[],"events":[],
             "physical_ground_truth_verified":False,"yolo_detection_verified":False}
        exp.update(status="Prepared",updated_utc=utc(),request=deepcopy(request or exp.get("request",{})))
        exp["revisions"].append(item)
        exp["events"].append({"utc":utc(),"action":"references_encoded","revision":revision})
        self.blobs.update(blobs);self.data["experiments"][eid]=exp;self.data["active"][mode]=eid
        return exp

    def record_presentation(self,mode,value):
        exp=self.active(mode)
        if not exp or value.get("signature")!=exp["revisions"][-1]["signature"]:return
        settings=value.get("calibration_settings")
        if settings is not None:
            if settings.get("pattern") not in {"gray","detail","fault"}:return
            # Display settings are observational metadata, never research pixels/labels.
            settings=deepcopy(settings)
            if exp.get("led_calibration")==settings:return
            exp["led_calibration"]=settings
            exp["events"].append({"utc":utc(),"action":"led_calibration_settings","revision":len(exp["revisions"]),"settings":settings})

    def confirm(self,exp,stage,fault_id=None):
        revision=exp["revisions"][-1]
        if stage=="CLOSE" and fault_id not in {f["id"] for f in revision["faults"]}:raise ValueError("Select a current Close fault first")
        exp["events"].append({"utc":utc(),"action":"operator_confirmed_drone_capture","stage":stage,"fault_id":fault_id,
            "revision":revision["revision"],"automatically_verified":False})
        exp["updated_utc"]=utc()

    def finish(self,exp):
        for ref in exp["revisions"][-1]["references"]:
            if hashlib.sha256(self.blobs.get(ref["filename"],b"")).hexdigest()!=ref["sha256"]:raise ValueError("Missing or invalid reference; restore its backup first")
        exp.update(status="Saved",saved_utc=utc())
        exp["events"].append({"action":"experiment_saved","utc":utc(),"external_video_verified":False})

    def bundle(self,historical=True):
        buffer=io.BytesIO();flat=io.StringIO(newline="")
        writer=csv.DictWriter(flat,fieldnames=["experiment_id","revision","mode","source","fault_id","module_id","fault_type","severity","metadata_json"])
        writer.writeheader()
        with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("scene_log.json",json_bytes(self.data))
            for exp in self.data["experiments"].values():
                for rev in exp["revisions"]:
                    for fault in rev["faults"]:
                        writer.writerow({"experiment_id":exp["experiment_id"],"revision":rev["revision"],"mode":exp["mode"],
                            "source":rev["source"]["identity"],"fault_id":fault["id"],"module_id":fault["module_id"],
                            "fault_type":fault["fault_type"],"severity":fault["severity"],"metadata_json":json_bytes(fault).decode()})
                    for ref in rev["references"]:
                        data=self.blobs.get(ref["filename"])
                        if data is None or hashlib.sha256(data).hexdigest()!=ref["sha256"]:raise ValueError("Reference bytes missing or corrupt; session ZIP not created")
                        z.writestr(ref["filename"],data)
                    z.writestr(f"{exp['experiment_id']}/r{rev['revision']:03}_metadata.json",json_bytes(rev))
            z.writestr("faults.csv",flat.getvalue().encode("utf-8-sig"))
            if historical and "pv_dc_experiments_v2" in self.session:
                z.writestr("historical_cases.zip",Journal(self.session).bundle())
        return buffer.getvalue()

    def restore(self,raw):
        """Validate all records and bytes, then merge; no archive extraction."""
        blobs={};history={}
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if len(z.infolist())>12000 or sum(i.file_size for i in z.infolist())>256*1024*1024:raise ValueError("Session backup exceeds restore limit")
                if len(z.namelist())!=len(set(z.namelist())):raise ValueError("Duplicate backup members")
                data=json.loads(z.read("scene_log.json"))
                if data.get("schema")!=KEY or not isinstance(data.get("experiments"),dict):raise ValueError("Not a scene experiment backup")
                for eid,exp in data["experiments"].items():
                    if eid!=exp["experiment_id"] or not re.fullmatch(r"scene-[a-f0-9]{32}",eid):raise ValueError("Invalid experiment identity")
                    if exp["mode"] not in {"Generated","Real Farm"} or not exp["revisions"]:raise ValueError("Invalid experiment record")
                    exp["physical_ground_truth_verified"]=False;exp["yolo_detection_verified"]=False
                    for rev in exp["revisions"]:
                        ids=[f["id"] for f in rev["faults"]]
                        if len(ids)!=len(set(ids)) or any(not 1<=int(f["severity"])<=10 for f in rev["faults"]):raise ValueError("Invalid fault records")
                        rev["geometry_verified"]=False;rev["physical_ground_truth_verified"]=False
                        for ref in rev["references"]:
                            name=ref["filename"]
                            if not re.fullmatch(r"scene-[a-f0-9]{32}/[A-Za-z0-9_-]+\.png",name):raise ValueError("Invalid reference filename")
                            content=z.read(name)
                            if hashlib.sha256(content).hexdigest()!=ref["sha256"]:raise ValueError("Reference hash mismatch")
                            with Image.open(io.BytesIO(content)) as image:image.verify()
                            blobs[name]=content
                if any(eid not in data["experiments"] for eid in data["active"].values()):raise ValueError("Invalid active experiment")
                if "historical_cases.zip" in z.namelist():
                    from pv_data_collection import _load_cases
                    Journal(history).restore(z.read("historical_cases.zip"),_load_cases(),is_zip=True)
        except (KeyError,TypeError,OSError,zipfile.BadZipFile,json.JSONDecodeError) as exc:raise ValueError(f"Invalid backup: {exc}") from exc
        self.data["experiments"].update(data["experiments"]);self.data["active"].update(data["active"]);self.blobs.update(blobs)
        if history:
            journal=Journal(self.session);restored=Journal(history)
            journal.data["experiments"].update(restored.data["experiments"])
            journal.data["active_by_mode"].update(restored.data["active_by_mode"]);journal.blobs.update(restored.blobs)

    def resume_simulator(self,ns,exp):
        rev=exp["revisions"][-1];saved=deepcopy(rev["simulator"])
        if exp["mode"]=="Real Farm":
            ref=next(r for r in rev["references"] if r["stage"]=="FAR" and r["modality"]=="original")
            image=Image.open(io.BytesIO(self.blobs[ref["filename"]])).convert("RGB")
            if ns["real_farm_source_identity"](image)!=rev["source"]["source_pixel_sha256"]:raise ValueError("Source photo identity mismatch")
            state=saved["state"]["real_farm_state"];state.pop("geometry_confirmation",None)
            if state.get("source_rgb_pixel_sha256")!=rev["source"]["source_pixel_sha256"]:raise ValueError("Geometry/source identity mismatch")
            saved["state"].update(real_farm_image=image,real_farm_image_path="restored_session_source.png")
        before=simulator_snapshot(ns)
        try:
            restore_simulator(ns,saved)
            from pv_scene_frames import acquire_scene
            acquired=acquire_scene(ns,exp["mode"])
            expected={(f["id"],f["module_id"],f["fault_type"],f["severity"]) for f in rev["faults"]}
            if {(f["id"],f["module_id"],f["fault_type"],f["severity"]) for f in acquired["faults"]}!=expected:raise ValueError("Restored simulator faults differ from the saved experiment")
        except Exception:
            restore_simulator(ns,before);raise
        ns["st"].session_state["pv_scene_previous_simulator"]=before
        return acquired
