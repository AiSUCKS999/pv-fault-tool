"""Portable access to already prepared RGB sources; never runs a detector."""


def real_farm_catalog_entries():
    path=ROOT/"assets/real_farm_prepared/catalog.json"
    return json.loads(path.read_text(encoding="utf-8")).get("images",[]) if path.is_file() else []


def real_farm_catalog_asset(relative,expected_sha=None):
    root=(ROOT/"assets/real_farm_prepared").resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError("Prepared asset leaves dataset directory")
    if expected_sha and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError("Prepared asset does not match its saved SHA-256")
    return path


def real_farm_catalog_selected():
    chosen=st.session_state.get("real_farm_prepared_picker")
    if not chosen or chosen=="upload":return
    entry=next(e for e in real_farm_catalog_entries() if e["id"]==chosen)
    path=real_farm_catalog_asset(entry["rgb"],entry["rgb_file_sha256"])
    image=ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if real_farm_source_identity(image)!=entry["rgb_pixel_sha256"]:raise ValueError("Prepared RGB identity mismatch")
    state=real_farm_default_state(path,image,path.name,entry["rgb_file_sha256"][:16])
    if not real_farm_camera_saved_geometry(state,image):raise ValueError("Prepared source geometry is unavailable")
    state["prepared_catalog_id"]=chosen
    section=st.session_state.get("real_farm_section","Farm Setup")
    for key in list(st.session_state):
        if key.startswith("real_farm_") and key not in {"real_farm_prepared_picker","real_farm_section"}:
            st.session_state.pop(key,None)
    st.session_state.update(real_farm_image=image,real_farm_image_path=str(path),real_farm_state=state,
        real_farm_section=section,real_farm_undo_stack=[],real_farm_redo_stack=[])


def real_farm_camera_random_modules(state,image,credible):
    """Only source-supported, resolved modules not occupied by manual observations."""
    manual=[o for o in state.get("camera_observations",[]) if o.get("placement_origin")!="random"]
    result=[]
    for module in sorted(credible,key=lambda m:m["module_id"]):
        if module.get("physical_module_unresolved") or module.get("simulation_only"):continue
        q=np.float32(module["outer_polygon_rgb"])
        if any(o["module_id"]==module["module_id"] or
               cv2.pointPolygonTest(q,tuple(map(float,o["center_rgb_px"])),False)>=0 for o in manual):continue
        result.append(module)
    return result


def real_farm_camera_forget_observation_widgets(state,observations):
    source=state.get("source_rgb_pixel_sha256","")[:12]
    prefixes=tuple(f"real_farm_camera_observation_{source}_{o['observation_id']}_" for o in observations)
    if prefixes:
        for key in list(st.session_state):
            if key.startswith(prefixes):st.session_state.pop(key,None)


def real_farm_camera_remove_random(state):
    real_farm_camera_forget_observation_widgets(state,[o for o in state.get("camera_observations",[]) if o.get("placement_origin")=="random"])
    state["camera_observations"]=[o for o in state.get("camera_observations",[]) if o.get("placement_origin")!="random"]
    state.pop("camera_random_batch",None)


def real_farm_camera_random_fault_types(fault_types=None):
    options=[*CLOSE_DIAGNOSTIC_FAULT_TYPES,"SingleHotSpot","PartialShading"]
    requested=list(CLOSE_DIAGNOSTIC_FAULT_TYPES if fault_types is None else fault_types)
    if not requested or any(fault not in options for fault in requested):
        raise ValueError("Select at least one supported fault class")
    return [fault for fault in options if fault in requested]


def real_farm_camera_generate_random(state,image,credible,count,seed,fault_types=None, *, policy="Random", counts=None, severity_distribution="Uniform", severity=7, severity_limits=(1,10), placement="Random"):
    """Place the requested distribution on distinct valid targets; preserve manual faults."""
    from pv_scene_generate import assignment, choose_targets
    fault_types=real_farm_camera_random_fault_types(fault_types) if count else []
    assigned=assignment(int(count),fault_types,seed,policy,counts,severity_distribution,severity,severity_limits)
    pool=real_farm_camera_random_modules(state,image,credible)
    if int(count)>len(pool):raise ValueError(f"Requested {count} faults; only {len(pool)} unoccupied valid modules available")
    if not pool and count:raise ValueError("No resolved modules for random placement. Use Manual Select to confirm a provisional PV target.")
    selected=choose_targets([dict(m,position=np.float32(m["active_polygon_rgb"]).mean(0).tolist()) for m in pool],int(count),seed,placement)
    proposals=[]
    # Independent module streams keep relative positions varied and stable even
    # when target selection order changes. Morphology still uses the experiment seed.
    for module in selected:
        module_seed=seed_for(int(seed),module["module_id"],"camera-placement-v3")
        proposals.extend(real_farm_generate_observations([module],1,module_seed))
    lookup={m["module_id"]:m for m in pool};rng=np.random.default_rng(int(seed));validated=[]
    for proposal in proposals:
        module=lookup[proposal["module_id"]];q=np.float32(module["active_polygon_rgb"])
        low=np.maximum(0,np.floor(q.min(0)).astype(int)-1);high=np.minimum(image.size,np.ceil(q.max(0)).astype(int)+2)
        bounds=tuple(map(int,(*low,*high)))
        support=real_farm_module_mask(image.size,[module],True,bounds)>0
        point=np.float32(proposal["center_rgb_px"]);px,py=np.rint(point-low).astype(int)
        valid=(0<=py<support.shape[0] and 0<=px<support.shape[1] and support[py,px] and
               cv2.pointPolygonTest(q,tuple(map(float,point)),False)>=0)
        if not valid:
            distance=cv2.distanceTransform(np.uint8(support),cv2.DIST_L2,5)
            yy,xx=np.nonzero(support&(distance>=max(1.,float(distance.max())*.45)))
            if not len(xx):continue
            index=int(rng.integers(len(xx)));point=np.float32([xx[index],yy[index]])+low
        target=dict(module);target["simulation_anchor_rgb"]=point.astype(float).tolist()
        validated.append((target,proposal))
    if len(validated)!=int(count):raise ValueError("Some requested targets lack valid PV support; no batch was replaced")
    real_farm_camera_observation_source(state,image)
    real_farm_camera_remove_random(state)
    identity=[real_farm_source_identity(image),int(seed),int(count),"controlled-distribution-v3",fault_types,assigned,placement]
    batch=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    for (target,proposal),(fault,level) in zip(validated,assigned):
        observation=real_farm_camera_new_observation(state,image,target,severity=level)
        observation.update(seed=int(seed),placement_origin="random",random_batch_id=batch,
            diagnostic_fault_type=None if fault=="SingleHotSpot" else fault)
    state["camera_random_batch"]={"id":batch,"seed":int(seed),"count":len(validated),
        "assignment_policy":policy,"placement_distribution":placement,"severity_distribution":severity_distribution,
        "fault_types":fault_types,"fault_counts":{fault:sum(code==fault for code,_ in assigned) for fault in fault_types if any(code==fault for code,_ in assigned)}}
    return len(validated)


def real_farm_camera_generate_random_clicked(state,image,credible):
    # Button callbacks run before the camera radio is instantiated. Updating
    # its value in the button's render branch raises a Streamlit widget error.
    error_key="real_farm_camera_random_error"
    st.session_state.pop(error_key,None)
    try:
        real_farm_camera_generate_random(state,image,credible,
            st.session_state["real_farm_camera_random_count"],
            st.session_state["real_farm_camera_random_seed"],
            st.session_state.get("real_farm_camera_random_fault_types"))
    except ValueError as exc:
        st.session_state[error_key]=str(exc)
        return
    real_farm_camera_show_thermal_result(image,paired=True)


def real_farm_camera_random_controls(state,image,credible):
    pool=real_farm_camera_random_modules(state,image,credible)
    if not pool:
        st.info("Random faults need resolved module geometry. Use Manual Select to confirm a provisional PV polygon.")
    fault_types=st.multiselect("Fault classes",[*CLOSE_DIAGNOSTIC_FAULT_TYPES,"SingleHotSpot"],
        default=list(CLOSE_DIAGNOSTIC_FAULT_TYPES),format_func=fault_display_name,
        key="real_farm_camera_random_fault_types")
    if not fault_types:st.info("Select at least one fault class to generate.")
    count_key="real_farm_camera_random_count"
    if count_key in st.session_state and not 1<=st.session_state[count_key]<=max(1,len(pool)):
        st.session_state[count_key]=max(1,min(int(st.session_state[count_key]),max(1,len(pool))))
    st.number_input("Number of faults",min_value=1,max_value=max(1,len(pool)),value=min(len(CLOSE_DIAGNOSTIC_FAULT_TYPES),max(1,len(pool))),step=1,key=count_key)
    st.number_input("Random seed",min_value=0,max_value=2147483647,value=42,step=1,key="real_farm_camera_random_seed")
    changed=False
    st.button("Generate Random Faults",disabled=not pool or not fault_types,use_container_width=True,
        on_click=real_farm_camera_generate_random_clicked,args=(state,image,credible))
    error=st.session_state.pop("real_farm_camera_random_error",None)
    if error:st.warning(error)
    if st.button("Remove Random Faults",disabled=not any(o.get("placement_origin")=="random" for o in state.get("camera_observations",[])),use_container_width=True):
        real_farm_camera_remove_random(state);changed=True
    counts={}
    for observation in state.get("camera_observations",[]):
        if observation.get("placement_origin")=="random":
            fault=observation.get("diagnostic_fault_type") or "SingleHotSpot"
            counts[fault]=counts.get(fault,0)+1
    if counts:st.caption("Random faults: "+" · ".join(f"{fault_display_name(fault)} × {count}" for fault,count in counts.items()))
    return changed
