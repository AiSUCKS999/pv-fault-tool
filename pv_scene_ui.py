"""Button-driven full-farm experiments, sharing the live simulator state."""
from copy import deepcopy
import hashlib

from pv_presentation import show_presentation
from pv_scene_frames import acquire_scene, current_signature, presentation_payload
from pv_scene_generate import generate_scene, restore_simulator, simulator_snapshot
from pv_scene_log import SceneJournal, json_bytes


def _error(ns,action):
    try:action()
    except (ValueError,RuntimeError,KeyError,OSError) as exc:
        ns["st"].session_state["pv_scene_error"]=str(exc)


def _capture(ns,mode,request=None,new=False):
    acquired=acquire_scene(ns,mode)
    exp=SceneJournal(ns["st"].session_state).capture(ns,acquired,request,new=new)
    ns["st"].session_state["pv_scene_presentation_"+mode]=(acquired["signature"],presentation_payload(ns,acquired))
    return exp


def _generate(ns,mode,request):
    def action():
        before=simulator_snapshot(ns)
        try:
            generate_scene(ns,mode,request)
            _capture(ns,mode,request,new=True)
        except Exception:
            restore_simulator(ns,before);raise
    _error(ns,action)


def _remove(ns,mode,clear=False):
    def action():
        before=simulator_snapshot(ns);st=ns["st"]
        if mode=="Generated":
            loc=ns["selected_location"]()["id"]
            for key in ("random_fault_records","panel_cell_fault_records"):
                st.session_state[key]=[r for r in st.session_state.get(key,[]) if r.get("location_id")!=loc or (not clear and r.get("placement_origin")!="scene_random")]
        else:
            state=st.session_state["real_farm_state"]
            if clear:state["camera_observations"]=[];state.pop("camera_random_batch",None)
            else:ns["real_farm_camera_remove_random"](state)
        try:_capture(ns,mode)
        except Exception:restore_simulator(ns,before);raise
        st.session_state["pv_scene_previous_simulator"]=before
    _error(ns,action)


def setup(ns,mode):
    st=ns["st"];key="scene_setup_"+mode
    request={}
    with st.expander("Experiment setup",expanded=SceneJournal(st.session_state).active(mode) is None):
        if mode=="Generated":
            cols=st.columns([2,1,1]);locations=[loc for loc in ns["LOCATIONS"] if loc["id"] in ns["LAYOUT_CAPS"]]
            ids=[loc["id"] for loc in locations];current=ns["selected_location"]()["id"]
            request["scene"]=cols[0].selectbox("Terrain",ids,index=ids.index(current) if current in ids else 0,
                format_func=lambda x:ns["location_by_id"](x)["name"],key=key+"scene")
            request["rows"]=cols[1].number_input("Rows",1,6,ns["custom_row_count"](current),key=key+"rows")
            request["columns"]=cols[2].number_input("Columns",1,6,ns["custom_panels_per_row"](current),key=key+"columns")
        else:
            entries=ns["real_farm_catalog_entries"]();ids=[e["id"] for e in entries]
            current=st.session_state.get("real_farm_state",{}).get("prepared_catalog_id")
            if st.session_state.get("real_farm_image") is not None and not current:ids=["Current upload"]+ids
            source=st.selectbox("Source photograph",ids,index=ids.index(current) if current in ids else 0,
                format_func=lambda x:next((e["label"] for e in entries if e["id"]==x),x),key=key+"source")
            request["source"]=None if source=="Current upload" else source
        columns=st.columns(3)
        preset=columns[0].selectbox("Fault count",["Healthy — 0","Sparse — 1–2","Moderate — 3–6","Dense — 7–12","Custom"],index=2,key=key+"preset")
        limits={"Healthy — 0":(0,0,0),"Sparse — 1–2":(1,2,2),"Moderate — 3–6":(3,6,6),"Dense — 7–12":(7,12,8),"Custom":(0,4096,8)}
        low,high,value=limits[preset]
        request["count"]=0 if high==0 else columns[1].number_input("Faulted modules",low,high,value,key=key+"count_"+preset)
        request["seed"]=columns[2].number_input("Seed",0,2147483647,42,key=key+"seed")
        codes=[*ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"],"SingleHotSpot","PartialShading"]
        request["classes"]=st.multiselect("Fault classes",codes,default=list(ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"]),format_func=ns["fault_display_name"],key=key+"classes")
        columns=st.columns(3)
        request["policy"]=columns[0].selectbox("Class distribution",["Random","Counts per class","Balanced"],key=key+"policy")
        request["placement"]=columns[1].selectbox("Placement",["Spread","Random","Clustered"],key=key+"placement")
        request["severity_distribution"]=columns[2].selectbox("Severity distribution",["Fixed","Uniform"],key=key+"severity_distribution")
        if request["severity_distribution"]=="Fixed":request["severity"]=st.slider("Severity",1,10,7,key=key+"severity")
        else:request["severity_limits"]=st.slider("Severity range",1,10,(3,10),key=key+"severity_limits")
        if request["policy"]=="Counts per class":
            request["counts"]={}
            columns=st.columns(3)
            for i,code in enumerate(request["classes"]):
                request["counts"][code]=columns[i%3].number_input(ns["fault_display_name"](code),0,4096,1,key=key+"class_"+code)
        request["clear_existing"]=st.checkbox("Clear active faults before generating",value=False,key=key+"clear")
        st.button("GENERATE EXPERIMENT",type="primary",use_container_width=True,on_click=_generate,args=(ns,mode,request),key=key+"generate")


def _restore_backup(ns,raw):
    _error(ns,lambda:SceneJournal(ns["st"].session_state).restore(raw))


def _confirm_current(ns,mode,stage):
    def action():
        journal=SceneJournal(ns["st"].session_state);exp=journal.active(mode)
        value=ns["st"].session_state.get("scene_display_"+mode) or {}
        revision=exp["revisions"][-1]
        if value.get("calibration", "off")!="off":raise ValueError("Exit LED calibration before confirming an experiment recording")
        if (stage=="CLOSE" and value.get("signature")!=revision["signature"]) or current_signature(ns,mode)!=revision["signature"]:raise ValueError("Display changed; wait for the current scene before confirming")
        index=int(value.get("index",0));faults=revision["faults"]
        if stage=="CLOSE" and (value.get("distance")!="CLOSE" or not 0<=index<len(faults)):raise ValueError("Select a current Close target")
        journal.confirm(exp,stage,faults[index]["id"] if stage=="CLOSE" else None)
        exp["events"][-1]["presentation"]={key:value.get(key) for key in ("distance","mode","fit","zoom","clean")}
        if exp.get("led_calibration"):
            exp["events"][-1]["led_calibration"]=deepcopy(exp["led_calibration"])
    _error(ns,action)


def render_scene_workflow(ns,mode,section="DATA COLLECTION"):
    st=ns["st"];journal=SceneJournal(st.session_state)
    if section in {"Faults","DATA COLLECTION"}:
        from pv_ros_mission import render_prepare
        render_prepare(ns,mode)
        setup(ns,mode)
    if message:=st.session_state.pop("pv_scene_error",None):st.error(message)
    ready=mode=="Generated" or st.session_state.get("real_farm_image") is not None
    exp=journal.active(mode)
    if ready:
        saved=st.session_state.get("pv_scene_presentation_"+mode)
        stale=not saved or saved[0]!=current_signature(ns,mode)
        if stale:
            st.button("LOAD CURRENT SCENE",on_click=lambda:_error(ns,lambda:_capture(ns,mode)),key="scene_load_"+mode,type="primary")
        else:
            show_presentation(ns,saved[1],key=mode)
            frames=acquire_scene(ns,mode)
            from pv_ros_mission import render_export
            render_export(ns,frames)
            view=st.session_state.get("pv_scene_view_"+mode,{})
            faults=frames["faults"];idx=min(max(int(view.get("index",0)),0),max(0,len(faults)-1))
            selected=faults[idx] if faults else None
            if selected:
                st.caption(f"{selected['id']} · {selected['fault_type']} · severity {selected['severity']} · {selected['rgb_classification']}")
            if section in {"Faults","DATA COLLECTION"}:
                buttons=st.columns(3)
                buttons[0].button("REMOVE RANDOM FAULTS",on_click=_remove,args=(ns,mode),key="scene_remove_"+mode)
                buttons[1].button("CLEAR CAMERA FAULTS" if mode=="Real Farm" else "CLEAR ALL FAULTS",on_click=_remove,args=(ns,mode,True),key="scene_clear_"+mode)
                previous=st.session_state.get("pv_scene_previous_simulator")
                def undo():
                    def action():
                        restore_simulator(ns,previous);_capture(ns,mode)
                    _error(ns,action)
                buttons[2].button("RESTORE PREVIOUS SCENE",disabled=previous is None,on_click=undo,key="scene_undo_"+mode)
            if section=="Review":
                st.subheader("Current faults")
                st.dataframe([{k:f.get(k) for k in ("id","module_id","fault_type","severity","rgb_classification")} for f in faults],hide_index=True,use_container_width=True)
                st.caption(f"{len(faults)} fault instances · {frames['source']['identity']} · geometry unverified")
                with st.expander("Diagnostics",expanded=False):st.json({"source":frames["source"],"faults":faults,"rgb":frames["rgb_report"]})
            if exp:
                st.caption(f"{exp['experiment_id']} · {exp['status']} · revision {len(exp['revisions'])}")
                columns=st.columns(3)
                columns[0].button("CONFIRM FAR RECORDED",on_click=_confirm_current,args=(ns,mode,"FAR"),key="scene_far_confirm_"+mode)
                columns[1].button("CONFIRM CLOSE RECORDED",disabled=not selected or view.get("distance")!="CLOSE",
                    on_click=_confirm_current,args=(ns,mode,"CLOSE"),key="scene_close_confirm_"+mode)
                columns[2].button("SAVE EXPERIMENT",on_click=lambda:_error(ns,lambda:journal.finish(exp)),key="scene_finish_"+mode,type="primary")
                completed={e.get("fault_id") for e in exp["events"] if e["action"]=="operator_confirmed_drone_capture" and e.get("stage")=="CLOSE" and e.get("revision")==len(exp["revisions"])}
                st.progress(len(completed)/len(faults) if faults else 1,text=f"Close recordings confirmed: {len(completed)}/{len(faults)}")
                if exp.get("status")=="Saved":
                    next_request=deepcopy(exp.get("request",{}))
                    if "count" in next_request:
                        next_request["seed"]=(int(next_request.get("seed",42))+1)%2147483648
                        st.button("NEXT EXPERIMENT",on_click=_generate,args=(ns,mode,next_request),key="scene_next_"+mode)
    if exp or journal.data["experiments"]:
        try:
            bundle=journal.bundle()
            st.download_button("DOWNLOAD SESSION ZIP / BACKUP",bundle,file_name="pv_scene_session.zip",mime="application/zip",key="scene_bundle_"+mode)
        except ValueError as exc:st.error(str(exc))
    st.caption("Session progress is temporary. Download the ZIP to keep it. Recording buttons confirm external drone captures.")
    with st.expander("Help / Diagnostics",expanded=False):
        from pathlib import Path
        st.markdown(Path(__file__).with_name("APP_DIAGNOSTICS.md").read_text(encoding="utf-8"))
    with st.expander("Restore session",expanded=False):
        upload=st.file_uploader("Experiment ZIP",type=["zip"],key="scene_restore_file_"+mode)
        if upload:st.button("RESTORE PROGRESS",on_click=_restore_backup,args=(ns,upload.getvalue()),key="scene_restore_"+mode)
        ids=[eid for eid,e in journal.data["experiments"].items() if e["mode"]==mode]
        if ids:
            chosen=st.selectbox("Saved experiment",ids,key="scene_resume_id_"+mode)
            def resume():
                def action():
                    saved_exp=journal.data["experiments"][chosen]
                    frames=journal.resume_simulator(ns,saved_exp)
                    journal.data["active"][mode]=chosen
                    st.session_state["pv_scene_presentation_"+mode]=(frames["signature"],presentation_payload(ns,frames))
                _error(ns,action)
            st.button("RESTORE SCENE",on_click=resume,key="scene_resume_"+mode)
