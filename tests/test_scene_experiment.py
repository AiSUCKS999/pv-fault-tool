from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_scene_generate import assignment, choose_targets, generate_scene, restore_simulator
from pv_scene_frames import acquire_scene, presentation_payload
from pv_presentation import presentation_html


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(ROOT/"tests/test_collection_acceptance.py"))["namespace"]()


def request(ns, **overrides):
    codes=list(ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"])+["SingleHotSpot"]
    return dict(scene="grass_open",rows=6,columns=6,count=8,classes=codes,policy="Counts per class",
        counts={c:1 for c in codes},seed=421,severity=7,placement="Spread",**overrides)


def test_explicit_distribution_and_count_validation():
    assert assignment(10,["A","B"],42,"Counts per class",{"A":10,"B":0})==[("A",7)]*10
    assert assignment(20,["A","B"],42)==assignment(20,["A","B"],42)
    assert any(a==b for a,b in zip(assignment(20,["A","B"],42),assignment(20,["A","B"],42)[1:]))
    with pytest.raises(ValueError):assignment(3,["A"],1,"Counts per class",{"A":2})
    with pytest.raises(ValueError):choose_targets([{"position":[0,0]}],2,42)


def test_generated_full_scene_and_recovery(ns):
    ns["st"].session_state.clear();ns["st"].query_params.clear()
    result=generate_scene(ns,"Generated",request(ns))
    faults=result["faults"]
    assert len(faults)==8 and len({f["id"] for f in faults})==8
    assert len({f["module_id"] for f in faults})==8
    assert len({f["row"] for f in faults})>=4 and len({f["array"] for f in faults})>=4
    assert result["source"]["occupied_arrays"]==36 and len(result["geometry"])==144
    assert {f["fault_type"] for f in faults}==set(request(ns)["classes"])
    assert all(f["id"] in result["close"] for f in faults)
    assert all(result["close"][f["id"]]["rgb"].size==result["close"][f["id"]]["thermal"].size for f in faults)
    assert np.any(np.array(result["far"]["thermal"])!=np.array(result["far"]["healthy_thermal"]))
    out=ROOT/"artifacts/experiment_recovery";out.mkdir(exist_ok=True,parents=True)
    for k,img in result["far"].items():
        if k!="bounds":img.save(out/f"generated_6x6_{k}.png")
    (out/"generated_6x6.html").write_text(presentation_html(presentation_payload(ns,result)),encoding="utf-8")
    (out/"generated_6x6_faults.json").write_text(json.dumps(faults,indent=2,default=str))
    # Normal navigation/re-acquisition neither adds faults nor changes identities.
    before=deepcopy(ns["st"].session_state["random_fault_records"])
    assert acquire_scene(ns,"Generated") is result
    assert ns["st"].session_state["random_fault_records"]==before
    bad=request(ns);bad.update(count=145,policy="Random")
    with pytest.raises(ValueError,match="only"):generate_scene(ns,"Generated",bad)
    assert ns["st"].session_state["random_fault_records"]==before
    healthy=request(ns);healthy.update(count=0,counts={},clear_existing=True)
    clean=generate_scene(ns,"Generated",healthy)
    assert clean["faults"]==[]
    assert np.array_equal(np.array(clean["far"]["thermal"]),np.array(clean["far"]["healthy_thermal"]))
    restore_simulator(ns,ns["st"].session_state["pv_scene_previous_simulator"])
    assert ns["st"].session_state["random_fault_records"]==before


def test_real_multiple_faults_same_geometry_and_photograph(ns):
    ns["st"].session_state.clear();ns["st"].query_params.clear()
    ns["st"].session_state["real_farm_prepared_picker"]="image_05";ns["real_farm_catalog_selected"]()
    state=ns["st"].session_state["real_farm_state"];image=ns["st"].session_state["real_farm_image"]
    geometry=deepcopy(state["modules"]);original=image.tobytes()
    targets=ns["real_farm_camera_candidates"](state,image)[1]
    manual=ns["real_farm_camera_new_observation"](state,image,targets[0])
    manual["diagnostic_fault_type"]="SingleDiode";manual_id=manual["observation_id"]
    req=request(ns);req.update(source="image_05")
    result=generate_scene(ns,"Real Farm",req)
    assert len(result["faults"])==9 and manual_id in {f["id"] for f in result["faults"]}
    assert state["modules"]==geometry and image.tobytes()==original
    assert len({f["module_id"] for f in result["faults"]})==9
    assert all(result["close"][f["id"]]["rgb"].size==result["close"][f["id"]]["thermal"].size for f in result["faults"])
    out=ROOT/"artifacts/experiment_recovery"
    for k,img in result["far"].items():
        if k!="bounds":img.save(out/f"real_05_{k}.png")
    (out/"real_05.html").write_text(presentation_html(presentation_payload(ns,result)),encoding="utf-8")
    (out/"real_05_faults.json").write_text(json.dumps(result["faults"],indent=2,default=str))
    ns["real_farm_camera_remove_random"](state)
    assert [o["observation_id"] for o in state["camera_observations"]]==[manual_id]
    assert state["modules"]==geometry and image.tobytes()==original
