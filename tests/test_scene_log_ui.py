from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_scene_generate import generate_scene
from pv_scene_log import SceneJournal,KEY,BLOBS


@pytest.fixture(scope="module")
def sample():
    ns=runpy.run_path(str(ROOT/"tests/test_collection_acceptance.py"))["namespace"]()
    request={"scene":"grass_open","rows":1,"columns":1,"count":2,"classes":["SoilingDust","SingleDiode"],
        "policy":"Counts per class","counts":{"SoilingDust":1,"SingleDiode":1},"seed":42,"severity":5}
    acquired=generate_scene(ns,"Generated",request)
    return ns,request,acquired


def test_scene_backup_restore_export_and_failed_encode(sample,monkeypatch):
    ns,request,acquired=sample;session={};journal=SceneJournal(session)
    exp=journal.capture(ns,acquired,request,new=True)
    before=deepcopy(journal.data)
    original=ns["export_current_view_bytes"]
    def failed(image):raise ValueError("encode failed")
    monkeypatch.setitem(ns,"export_current_view_bytes",failed)
    with pytest.raises(ValueError,match="encode failed"):journal.capture(ns,acquired,request,new=True)
    assert journal.data==before
    monkeypatch.setitem(ns,"export_current_view_bytes",original)
    fid=exp["revisions"][0]["faults"][0]["id"]
    journal.confirm(exp,"FAR");journal.confirm(exp,"CLOSE",fid);journal.finish(exp)
    assert len(exp["events"])==4 and exp["status"]=="Saved"
    raw=journal.bundle();out=ROOT/"artifacts/experiment_recovery";out.mkdir(exist_ok=True,parents=True)
    (out/"scene_session_example.zip").write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        data=json.loads(z.read("scene_log.json"));record=data["experiments"][exp["experiment_id"]]
        assert {f["id"] for f in record["revisions"][0]["faults"]}=={f["id"] for f in acquired["faults"]}
        assert all(not r["download_saved_confirmed"] for r in record["revisions"][0]["references"])
        assert "faults.csv" in z.namelist()
    reset={};restored=SceneJournal(reset);restored.restore(raw)
    assert restored.data==journal.data and restored.blobs==journal.blobs
    ns["st"].session_state.clear();ns["st"].query_params.clear()
    resumed=restored.resume_simulator(ns,restored.active("Generated"))
    assert {(f["id"],f["severity"]) for f in resumed["faults"]}=={(f["id"],f["severity"]) for f in acquired["faults"]}
    # Corrupt archives cannot overwrite an existing journal.
    broken=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source,zipfile.ZipFile(broken,"w") as target:
        for item in source.namelist():target.writestr(item,b"bad" if item.endswith(".png") else source.read(item))
    before=deepcopy(restored.data)
    with pytest.raises(ValueError,match="hash mismatch"):restored.restore(broken.getvalue())
    assert restored.data==before


def test_no_typing_generated_collection_and_current_review_export():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
    app.query_params.update(location="grass_open",custom_layout="1",row_count="1",panels_per_row="1",section="DATA COLLECTION")
    app.run();assert not app.exception
    assert app.radio(key="collection_mode_Generated").value=="Scene experiments"
    # All ordinary actions are buttons. Configuration has defaults.
    assert not app.text_input and not app.text_area
    app.selectbox(key="scene_setup_Generatedpreset").set_value("Sparse — 1–2").run()
    next(b for b in app.button if b.label=="GENERATE EXPERIMENT").click().run()
    assert not app.exception and not app.error
    journal=SceneJournal(app.session_state.filtered_state)
    exp=journal.active("Generated");eid=exp["experiment_id"]
    assert len(exp["revisions"][0]["faults"])==2
    next(b for b in app.button if b.label=="CONFIRM FAR RECORDED").click().run()
    assert not app.exception
    exp=SceneJournal(app.session_state.filtered_state).active("Generated")
    assert any(e["action"]=="operator_confirmed_drone_capture" for e in exp["events"])
    next(b for b in app.button if b.label=="SAVE EXPERIMENT").click().run()
    assert not app.exception and SceneJournal(app.session_state.filtered_state).active("Generated")["status"]=="Saved"
    for section in ("Review","Export"):
        app.query_params["section"]=section;app.run()
        assert not app.exception
        assert SceneJournal(app.session_state.filtered_state).active("Generated")["experiment_id"]==eid
        assert any(b.label=="DOWNLOAD SESSION ZIP / BACKUP" for b in app.get("download_button"))
    assert not app.text_input


def test_real_review_export_read_camera_collection():
    ns=runpy.run_path(str(ROOT/"tests/test_collection_acceptance.py"))["namespace"]()
    request={"source":"image_05","count":2,"classes":["SoilingDust"],"seed":42,"severity":7}
    acquired=generate_scene(ns,"Real Farm",request)
    assert len(acquired["faults"])==2
    state=ns["st"].session_state["real_farm_state"]
    state["observations"]=[]
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
    app.query_params["base_source"]="real_farm"
    for key in ("real_farm_state","real_farm_image","real_farm_image_path"):
        app.session_state[key]=deepcopy(ns["st"].session_state[key])
    app.session_state["real_farm_section"]="Review"
    app.run();assert not app.exception
    next(b for b in app.button if b.label=="LOAD CURRENT SCENE").click().run()
    assert not app.exception and not app.error
    exp=SceneJournal(app.session_state.filtered_state).active("Real Farm")
    assert {f["id"] for f in exp["revisions"][0]["faults"]}=={o["observation_id"] for o in state["camera_observations"]}
    app.radio(key="real_farm_section").set_value("Export").run()
    assert not app.exception and any(b.label=="DOWNLOAD SESSION ZIP / BACKUP" for b in app.get("download_button"))
