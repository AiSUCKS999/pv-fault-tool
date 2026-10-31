"""Explicit scene requests mapped to existing fault records and engines."""
from copy import deepcopy
import uuid

import numpy as np

from pv_scene_frames import acquire_scene


def assignment(count, classes, seed, policy="Random", counts=None, severity="Fixed", level=7, limits=(1,10)):
    if int(count)!=count or count<0:raise ValueError("Fault count must be a nonnegative integer")
    if count and not classes:raise ValueError("Select fault classes")
    rng=np.random.default_rng(int(seed)^0x505646)
    if policy=="Counts per class":
        counts=counts or {}
        if set(counts)-set(classes) or any(int(n)!=n or n<0 for n in counts.values()):raise ValueError("Invalid class counts")
        codes=[c for c in classes for _ in range(int(counts.get(c,0)))]
        if len(codes)!=count:raise ValueError("Class counts must sum to total fault count")
        rng.shuffle(codes)
    elif policy=="Balanced":
        codes=[]
        while len(codes)<count:codes.extend(rng.permutation(classes).tolist())
        codes=codes[:count]
    elif policy=="Random":codes=rng.choice(classes,size=count,replace=True).tolist() if count else []
    else:raise ValueError("Unsupported class distribution")
    low,high=map(int,limits)
    if not 1<=int(level)<=10 or not 1<=low<=high<=10:raise ValueError("Severity must be 1–10")
    severity_rng=np.random.default_rng(int(seed)^0x534556)
    values=[int(level)]*count if severity=="Fixed" else severity_rng.integers(low,high+1,count).tolist()
    return list(zip(codes,values))


def choose_targets(pool,count,seed,placement="Spread"):
    if count>len(pool):raise ValueError(f"Requested {count} faulted modules; only {len(pool)} unoccupied valid targets available")
    if count<0:raise ValueError("Fault count must be nonnegative")
    if not count:return []
    rng=np.random.default_rng(int(seed))
    if placement=="Random":return [pool[i] for i in rng.choice(len(pool),count,replace=False)]
    points=np.float64([m["position"] for m in pool]);scale=np.ptp(points,axis=0);scale[scale==0]=1
    points=(points-points.min(0))/scale
    chosen=[int(rng.integers(len(pool)))];distance=np.full(len(pool),np.inf)
    while len(chosen)<count:
        distance=np.minimum(distance,np.sum((points-points[chosen[-1]])**2,axis=1))
        distance[chosen]=-1 if placement=="Spread" else np.inf
        chosen.append(int(np.argmax(distance) if placement=="Spread" else np.argmin(distance)))
    return [pool[i] for i in chosen]


def simulator_snapshot(ns):
    st=ns["st"]
    keys=("random_fault_records","panel_cell_fault_records","real_farm_state","real_farm_image","real_farm_image_path")
    return {"query":dict(st.query_params),"state":{k:deepcopy(st.session_state[k]) for k in keys if k in st.session_state}}


def restore_simulator(ns,saved):
    st=ns["st"]
    for key in ("random_fault_records","panel_cell_fault_records","real_farm_state","real_farm_image","real_farm_image_path"):
        st.session_state.pop(key,None)
    st.session_state.update(deepcopy(saved["state"]))
    st.query_params.clear();st.query_params.update(saved["query"])
    for key in list(st.session_state):
        if key.startswith(("pv_scene_frames_","pv_scene_presentation_","real_farm_camera_acquisitions","real_farm_rgb_preview_cache","real_farm_camera_layers")):
            st.session_state.pop(key,None)


def generate_scene(ns,mode,request):
    """Transactional batch: validate/render before retaining the requested scene."""
    allowed=set(ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"])|{"SingleHotSpot","PartialShading"}
    classes=list(request.get("classes",[]));count=int(request["count"]);seed=int(request["seed"])
    if set(classes)-allowed:raise ValueError("Unsupported fault class requested")
    assigned=assignment(count,classes,seed,request.get("policy","Random"),request.get("counts"),
        request.get("severity_distribution","Fixed"),request.get("severity",7),request.get("severity_limits",(1,10)))
    saved=simulator_snapshot(ns);st=ns["st"];batch="scene-"+uuid.uuid4().hex[:12]
    try:
        if mode=="Generated":
            scene=request["scene"];rows=int(request["rows"]);cols=int(request["columns"])
            if not 1<=rows<=6 or not 1<=cols<=6:raise ValueError("Supported generated layouts are 1–6 rows and columns")
            asset=ns["ASSET_DIR"]/"solar-renders"/scene/f"{rows}x{cols}.webp"
            fallback=ns["ASSET_DIR"]/"solar-renders"/f"{rows}x{cols}.webp"
            if not asset.is_file() and not (scene=="grass_open" and fallback.is_file()):raise ValueError("Exact approved layout asset is unavailable")
            if scene=="rooftop_warehouse" and rows==cols==6 and "SoilingDust" in [a[0] for a in assigned]:
                raise ValueError("Known baseline failure: rooftop 6×6 SoilingDust remains blocked")
            st.session_state["thermal_renderer_mode"]="Current Thermal"
            st.query_params.update(location=scene,custom_layout="1",row_count=str(rows),panels_per_row=str(cols))
            loc=ns["location_by_id"](scene)
            current=ns["thermal_records_for_distance"](ns["records_for_location"](scene,"thermal"),"far")
            manual=[r for r in current if r.get("placement_origin")!="scene_random"]
            if request.get("clear_existing") or count==0:manual=[]
            if any((r.get("layout",{}).get("row_count"),r.get("layout",{}).get("panels_per_row"))!=(rows,cols) for r in manual):
                raise ValueError("Existing manual faults use another layout. Restore it or explicitly clear active faults")
            occupied={(r["row_id"],r["panel_number_in_row"],r["module_number_in_panel"]) for r in manual}
            pool=[]
            for row in range(1,rows+1):
                for col in range(1,cols+1):
                    for module in range(1,5):
                        rid,cell=ns["default_cell_for_array"](scene,row,col,module,20)
                        if (rid,col,module) not in occupied:
                            pool.append({"row":rid,"cell":cell,"position":[col+(module-1)*.08,row]})
            chosen=choose_targets(pool,count,seed,request.get("placement","Spread"))
            additions=[]
            for i,(target,(fault,severity)) in enumerate(zip(chosen,assigned),1):
                if fault in {"SurfaceObstruction","SoilingDust"}:
                    from pv_far_surface import generated_surface_records
                    contribution=generated_surface_records(ns,loc,fault,severity,target["row"],target["cell"]["id"])
                else:
                    contribution=[ns["fault_record"](loc,"thermal",ns["row_by_id"](scene,target["row"]),target["cell"],fault,severity)]
                for record in contribution:
                    record.update(experiment_fault_id=f"{batch}-F{i:03}",placement_origin="scene_random",placement_seed=seed,random_batch_id=batch)
                additions.extend(contribution)
            other=[r for r in st.session_state.get("random_fault_records",[]) if r.get("location_id")!=scene]
            st.session_state["random_fault_records"]=other+manual+additions
            st.session_state["panel_cell_fault_records"]=[r for r in st.session_state.get("panel_cell_fault_records",[]) if r.get("location_id")!=scene]
            rid,cell=ns["default_cell_for_array"](scene,1,1,1,1)
            st.query_params.update(selected_pv=rid,selected_cell=cell["id"],selected_array="",selected_module="",cell_active="0")
        else:
            source=request.get("source")
            current=st.session_state.get("real_farm_state",{})
            if source and current.get("prepared_catalog_id")!=source:
                st.session_state["real_farm_prepared_picker"]=source
                ns["real_farm_catalog_selected"]()
            state=st.session_state["real_farm_state"];image=st.session_state["real_farm_image"]
            credible=ns["real_farm_camera_candidates"](state,image)[1]
            if request.get("clear_existing") or count==0:state["camera_observations"]=[]
            ns["real_farm_camera_generate_random"](state,image,credible,count,seed,classes,
                policy=request.get("policy","Random"),counts=request.get("counts"),
                severity_distribution=request.get("severity_distribution","Fixed"),severity=request.get("severity",7),
                severity_limits=request.get("severity_limits",(1,10)),placement=request.get("placement","Spread"))
        acquired=acquire_scene(ns,mode)
        st.session_state["pv_scene_previous_simulator"]=saved
        st.session_state["pv_scene_request_"+mode]=deepcopy(request)
        return acquired
    except Exception:
        restore_simulator(ns,saved)
        raise
