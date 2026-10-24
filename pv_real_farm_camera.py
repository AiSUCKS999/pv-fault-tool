"""Exploratory camera capture, independent of USER-verified annotation masks.

Uses existing material, frame/grid recovery, sensor and production fault code.
No camera candidate or rendering copy grants a physical_review approval.
"""


def real_farm_camera_structure_checks(state,image,candidates):
    """Reject cell-sized grid proposals; this validates, never refits geometry.

    New crystalline-module proposals need repeated internal structure in BOTH
    directions. Thin-film, tiny, obscured or featureless faces remain uncertain.
    A saved source-matched correction project retains its existing geometry.
    """
    source=real_farm_source_identity(image)
    triage=state.get("geometry_review_triage",{})
    records=triage.get("by_module",{}) if triage.get("source")==source else {}
    reviewed=sum(records.get(m["module_id"],{}).get("polygon")==real_farm_polygon_signature(m)
                 for m in candidates)
    if candidates and reviewed>=len(candidates)*.75:return {}
    signature=real_farm_geometry_signature(state.get("modules",[]))
    cached=state.get("camera_structure_check",{})
    if cached.get("source")==source and cached.get("geometry")==signature and cached.get("version")==2:return cached["rejected"]
    rgb=real_farm_rgb_array(image);rejected={}
    for module in candidates:
        q=np.float32(module["active_polygon_rgb"])
        lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        w,h=float((lengths[0]+lengths[2])/2),float((lengths[1]+lengths[3])/2)
        if min(w,h)<40:
            rejected[module["module_id"]]="Insufficient internal module evidence; may be a cell fragment"
            continue
        ratio=min(1.,256./max(w,h));width=max(8,round(w*ratio));height=max(8,round(h*ratio))
        target=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
        crop=cv2.warpPerspective(rgb,cv2.getPerspectiveTransform(q,target),(width,height))
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY).astype(np.float32)
        counts=[]
        for axis in (0,1):
            gradient=np.abs(cv2.Sobel(gray,cv2.CV_32F,1 if axis==0 else 0,0 if axis==0 else 1,ksize=3))
            inner=gradient[max(2,height//10):-max(2,height//10),max(2,width//10):-max(2,width//10)]
            profile=inner.mean(axis=axis)
            threshold=max(2.,float(np.median(profile)+np.std(profile)*.65))
            points=np.flatnonzero(profile>threshold)
            groups=np.split(points,np.where(np.diff(points)>3)[0]+1)
            counts.append(sum(bool(len(group)) for group in groups))
        if min(counts)<3:
            rejected[module["module_id"]]="No repeated internal cell structure in both directions; physical module unresolved"
    initial_rejections=len(rejected)
    unreliable=bool(candidates and initial_rejections/len(candidates)>.25)
    if unreliable:
        rejected={m["module_id"]:"Scene-level cell/module confusion; automatic geometry withheld" for m in candidates}
    state["camera_structure_check"]={"version":2,"source":source,"geometry":signature,"rejected":rejected,
        "cell_scale_rejections":initial_rejections,"scene_geometry_unreliable":unreliable,
        "method":"internal_structure_safety_gate_not_module_verification"}
    return rejected


def real_farm_camera_candidates(state, image):
    candidates=real_farm_recovered_modules(state.get("modules",[]),include_unaccepted=True,image_size=image.size)
    triage=state.get("geometry_review_triage",{})
    records=triage.get("by_module",{}) if triage.get("source")==real_farm_source_identity(image) else {}
    structure_rejected=real_farm_camera_structure_checks(state,image,candidates)
    credible=[]
    issues=real_farm_geometry_review_issues(state,image) if candidates else {}
    for m in candidates:
        if m["module_id"] in structure_rejected:continue
        q=np.float32(m.get("outer_polygon_rgb",[]))
        if q.shape!=(4,2) or not real_farm_mechanical_validation(q,image.size)["mechanically_valid"]:continue
        if m.get("mask_source")=="SEARCH_ROI" or m.get("refinement_status","accepted")!="accepted":continue
        record=records.get(m["module_id"],{})
        supported=(record.get("polygon")==real_farm_polygon_signature(m) and
                   record.get("category") in {"confirmed_valid","warning_contradicted"})
        if record.get("category")=="weak_evidence" and record.get("polygon")==real_farm_polygon_signature(m):continue
        if m.get("physical_module_unresolved") and not supported:continue
        if real_farm_rgb_boundary_issues(m,image):continue
        if any(any(word in reason for word in ("Overlap with","Duplicate with","Unusual local module size","Invalid or nonconvex")) for reason in issues.get(m["module_id"],[])):continue
        # Cropped perimeters cannot anchor a confidently positioned fault.
        if q.min()<1 or q[:,0].max()>image.width-2 or q[:,1].max()>image.height-2:continue
        if float(m.get("pv_support_fraction",1))<.65:continue
        credible.append(m)
    return candidates,credible


def real_farm_camera_local_target(state,image,point,allow_point=False):
    """Bounded local PV/frame recovery for simulation, never annotation approval.

    Reuse the existing RGB support classifier and physical-boundary checks.
    Unlike table recovery, do not close across thin physical frame gutters.
    An unresolved support patch is explicitly NOT a physical module polygon.
    """
    from pv_module_recovery import material_mask,ordered_quad,physical_boundary_issues
    x,y=map(float,point)
    if not (0<=x<image.width and 0<=y<image.height):return None
    # Existing table orientation/extent bounds the local native search window.
    tables=[t for t in state.get("module_tables",[]) if t.get("quad") and
            cv2.pointPolygonTest(np.float32(t["quad"]),(x,y),False)>=0]
    span=192.
    if tables:
        q=np.float32(tables[0]["quad"])
        edges_table=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        span=float(np.clip(min(edges_table)*2.5,64,480))
    x1,y1=max(0,int(x-span)),max(0,int(y-span))
    x2,y2=min(image.width,int(x+span+1)),min(image.height,int(y+span+1))
    rgb=real_farm_rgb_array(image)[y1:y2,x1:x2]
    support=material_mask(rgb)
    # Existing cool-blue PV evidence can be vetoed by the generic vegetation
    # contrast test on cyan glass. Retain unequivocally blue support locally;
    # never close across neutral frames or alter the scene/annotation masks.
    r,g,b=(rgb[:,:,i].astype(np.int16) for i in range(3))
    support=np.maximum(support,np.uint8((b>=g-4)&(b>r+6))*255)
    px,py=int(round(x))-x1,int(round(y))-y1
    px,py=min(px,support.shape[1]-1),min(py,support.shape[0]-1)
    if not support[py,px]:return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(support,8)
    component=np.uint8(labels==labels[py,px])*255
    contours,hierarchy=cv2.findContours(component,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contour=max((c for i,c in enumerate(contours) if hierarchy[0,i,3]<0),key=cv2.contourArea)
    hull=cv2.convexHull(contour)
    approx=cv2.approxPolyDP(hull,cv2.arcLength(hull,True)*.018,True)
    q=ordered_quad(approx.reshape(4,2) if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(hull)))
    edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    short,long=float(min(edges)),float(max(edges))
    area=max(1.,abs(cv2.contourArea(q)))
    box=cv2.boundingRect(contour)
    clipped=box[0]<=1 or box[1]<=1 or box[0]+box[2]>=support.shape[1]-1 or box[1]+box[3]>=support.shape[0]-1
    # Keep merged tables, unresolved fragments and clipped components out of
    # module-dependent diagnoses. No size/count target creates new frames.
    resolved=(not clipped and short>=12 and 1.25<=long/max(short,1)<=3.3 and
              cv2.contourArea(contour)/area>.78 and not physical_boundary_issues(rgb,q))
    kind="module" if resolved else "pv_point"
    if not resolved:
        if not allow_point:return None
        # Local PV support bounds the exploratory footprint. This patch has no
        # claimed physical perimeter or electrical-section/cell labels.
        distance=cv2.distanceTransform(component,cv2.DIST_L2,3)
        radius=float(distance[py,px])
        if radius<3:return None
        r=radius*.9
        q=np.float32([[px-r,py-r],[px+r,py-r],[px+r,py+r],[px-r,py+r]])
    q+=np.float32([x1,y1])
    if not real_farm_mechanical_validation(q,image.size)["mechanically_valid"]:return None
    mid="SIM-"+hashlib.sha256(np.rint(q).astype(np.int32).tobytes()).hexdigest()[:8]
    target=real_farm_module_record(mid,q.tolist(),image.size,inset_ratio=.018,provenance_note="provisional_local_RGB_support")
    # Refined visible support only clips the fault; never changes the master.
    local_contour=contour.reshape(-1,2)+np.int32([x1,y1])
    target.update(mask_source="PROVISIONAL_SIMULATION_SUPPORT",simulation_only=True,
        training_verified=False,geometry_status="PROVISIONAL",placement_kind=kind,
        requires_physical_review=True,physical_module_unresolved=not resolved,
        refined_mask_contours_rgb=[local_contour.tolist()],
        refined_mask_holes_rgb=[(c.reshape(-1,2)+np.int32([x1,y1])).tolist() for i,c in enumerate(contours) if hierarchy[0,i,3]>=0],
        refinement_status="accepted",
        simulation_anchor_rgb=[x,y],simulation_context_radius=max((short*1.5 if kind=="pv_point" else long*1.25),float(min(edges_table)) if tables else short),source_rgb_pixel_sha256=real_farm_source_identity(image))
    target["manual_final_polygon_rgb"]=None
    target["geometry_confidence"]=min(target["geometry_confidence"],.5) if resolved else 0.
    target["confidence"].update(label="PROVISIONAL_SIMULATION_ONLY",module_polygon="PROVISIONAL_RGB_SUPPORT" if resolved else "UNKNOWN_POINT_TARGET",cells="NOT_AVAILABLE",electrical_sections="NOT_AVAILABLE")
    target["provenance"].update(created_by="Exploratory camera target",geometry_method="LOCAL_RGB_SUPPORT" if resolved else "PV_POINT_FOOTPRINT_ONLY")
    return target

def real_farm_camera_simulation_targets(state,image,candidates,credible):
    """Camera-only recovery; canonical modules/reviews/verified export untouched."""
    identity=real_farm_source_identity(image)
    signature=real_farm_geometry_signature(state.get("modules",[]))
    cache=state.get("camera_simulation",{})
    if cache.get("source")!=identity or cache.get("geometry")!=signature or cache.get("version")!=4:
        cache={"version":4,"source":identity,"geometry":signature,"targets":[],"automatic_attempted":False}
        state["camera_simulation"]=cache
    if not credible and not cache["automatic_attempted"]:
        cache["automatic_attempted"]=True
        center=np.float32(image.size)/2
        nearest=sorted(candidates,key=lambda m:np.linalg.norm(np.float32(m["outer_polygon_rgb"]).mean(0)-center))[:12]
        points=[np.float32(proposal["active_polygon_rgb"]).mean(0) for proposal in nearest]
        if not points:points=[np.float32(t["quad"]).mean(0) for t in state.get("module_tables",[])[:12] if t.get("quad")]
        for point in points:
            target=real_farm_camera_local_target(state,image,point)
            if target:
                target["simulation_anchor_rgb"]=np.float32(target["active_polygon_rgb"]).mean(0).tolist()
                cache["targets"].append(target);break
    return [t for t in cache["targets"] if t.get("source_rgb_pixel_sha256")==identity]

def real_farm_camera_polygon_target(state,image,polygon,anchor=None):
    """Rebuild only an exploratory target's transform after corner editing."""
    if not real_farm_mechanical_validation(polygon,image.size)["mechanically_valid"]:return None
    q=np.float32(polygon)
    if anchor is None or cv2.pointPolygonTest(q,tuple(map(float,anchor)),False)<=0:
        anchor=q.mean(0).tolist()
    support=real_farm_camera_local_target(state,image,anchor,allow_point=True)
    if support is None:return None
    mid="SIM-"+hashlib.sha256(np.rint(q).astype(np.int32).tobytes()).hexdigest()[:8]
    target=real_farm_module_record(mid,polygon,image.size,inset_ratio=.018,provenance_note="provisional_selected_RGB_polygon")
    target.update(simulation_only=True,training_verified=False,geometry_status="PROVISIONAL",
        placement_kind="module",requires_physical_review=True,physical_module_unresolved=False,
        simulation_anchor_rgb=list(anchor),simulation_context_radius=float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).max()*1.25),
        source_rgb_pixel_sha256=real_farm_source_identity(image),
        refined_mask_contours_rgb=support["refined_mask_contours_rgb"],
        refined_mask_holes_rgb=support["refined_mask_holes_rgb"],
        mask_source="PROVISIONAL_SIMULATION_SUPPORT",refinement_status="accepted")
    target["provenance"].update(created_by="Exploratory camera target",geometry_method="SELECTED_RGB_POLYGON")
    target["confidence"].update(label="PROVISIONAL_SIMULATION_ONLY",module_polygon="SELECTED_RGB_POLYGON")
    return target

def real_farm_camera_pick_target(state,image,selected):
    """One click and confirmation, without editing the farm's annotation set."""
    from pv_geometry_component import quad_editor
    refresh=False
    with st.expander("Select / adjust PV polygon on RGB",expanded=True):
        st.caption("Click an outlined proposal or another PV surface. Adjust its four corners if needed, save the outline, then confirm to create a hotspot. Simulation only; not training verification.")
        identity=real_farm_source_identity(image)
        cache=state["camera_simulation"]
        pending=cache.get("pending")
        if pending:
            if st.button("Select a different PV surface"):
                cache.pop("pending",None);cache.pop("edit_error",None);pending=None
        elif selected and st.button("Adjust selected target polygon"):
            cache["pending"]=json.loads(json.dumps(selected));pending=cache["pending"]
        event=quad_editor(image=data_uri(image),origin=[0,0],crop_size=list(image.size),source_size=list(image.size),
            source_identity=identity,revision=cache["geometry"],selection_only=pending is None,target_edit=pending is not None,
            polygon=(pending or selected or {}).get("outer_polygon_rgb",[]),selected_id=(pending or {}).get("module_id"),
            neighbors=[m["outer_polygon_rgb"] for m in state.get("modules",[])],
            detail="Click PV support",key="real_farm_camera_point_v2_"+identity[:12],default=None)
        if event and event.get("event_id")!=cache.get("handled_event"):
            cache["handled_event"]=event.get("event_id")
            if event.get("source_identity")==identity and event.get("revision")==cache["geometry"] and event.get("action")=="pick":
                point=event.get("point",[])
                matches=[m for m in state.get("modules",[]) if len(point)==2 and cv2.pointPolygonTest(np.float32(m["outer_polygon_rgb"]),tuple(map(float,point)),False)>=0]
                proposal=min(matches,key=lambda m:abs(real_farm_polygon_area(m["outer_polygon_rgb"]))) if matches else None
                cache["pending"]=(real_farm_camera_polygon_target(state,image,proposal["outer_polygon_rgb"],point) if proposal else
                                  real_farm_camera_local_target(state,image,point,allow_point=True))
                cache["pick_failed"]=cache["pending"] is None
                cache.pop("edit_error",None)
                pending=cache["pending"];refresh=True
            elif event.get("source_identity")==identity and event.get("revision")==cache["geometry"] and event.get("action")=="save" and pending:
                corrected=real_farm_camera_polygon_target(state,image,event.get("polygon",[]),pending.get("simulation_anchor_rgb"))
                if corrected:
                    cache["pending"]=corrected;pending=corrected;cache.pop("edit_error",None);refresh=True
                else:cache["edit_error"]="Outline must be a valid four-corner polygon on visible PV support. Adjust the corners and save again."
        if cache.get("pick_failed"):st.warning("No supported PV surface at that location. Click inside a visible module face.")
        if cache.get("edit_error"):st.warning(cache["edit_error"])
        if pending:
            st.caption("PROVISIONAL module perimeter — not verified ground truth." if pending["placement_kind"]=="module" else
                       "PROVISIONAL PV point: perimeter unresolved. Hotspot only; footprint scale is approximate.")
            if st.button("Confirm polygon and add hotspot",use_container_width=True,disabled=bool(cache.get("edit_error"))):
                cache["targets"]=[t for t in cache["targets"] if t["module_id"]!=pending["module_id"]]+[pending]
                cache["selected_target"]=pending["module_id"]
                real_farm_camera_new_observation(state,image,pending)
                cache.pop("pending",None);refresh=True

    return refresh

def real_farm_camera_new_observation(state,image,target,severity=7):
    observations=state.setdefault("camera_observations",[])
    observation=real_farm_add_module_observation([target],observations,target["module_id"],severity,42)
    anchor=target.get("simulation_anchor_rgb")
    if anchor is not None and cv2.pointPolygonTest(np.float32(target["active_polygon_rgb"]),tuple(map(float,anchor)),False)>0:
        uv=cv2.perspectiveTransform(np.float32(anchor).reshape(1,1,2),np.float64(target["homography_rgb_to_module"]))[0,0]
        observation.update(local_uv=uv.astype(float).tolist(),center_rgb_px=list(anchor))
    observation.update(diagnostic_fault_type=None,source_rgb_pixel_sha256=real_farm_source_identity(image),
        target_geometry=json.loads(json.dumps(target)),simulation_only=True,training_verified=False,
        geometry_status="PROVISIONAL" if target.get("simulation_only") else "UNVERIFIED",
        footprint_rgb={"status":"awaiting reference-backed scalar placement"})
    observations.append(observation)
    state["camera_select_observation_queue"]=observation["observation_id"]
    return observation

def real_farm_camera_observation_source(state,image):
    source=real_farm_source_identity(image)
    observations=state.setdefault("camera_observations",[])
    previous=state.get("camera_observation_source",state.get("source_rgb_pixel_sha256"))
    foreign=[o for o in observations if o.get("source_rgb_pixel_sha256",previous)!=source]
    if foreign:
        state.setdefault("camera_detached_observations",[]).extend(foreign)
        state["camera_observations"]=[o for o in observations if o not in foreign]
        state.pop("camera_selected_observation",None)
    state["camera_observation_source"]=source
    for observation in state["camera_observations"]:
        observation.setdefault("source_rgb_pixel_sha256",source)
        if observation.get("diagnostic_fault_type")=="SingleHotSpot":observation["diagnostic_fault_type"]=None
        observation.update(simulation_only=True,training_verified=False)

def real_farm_camera_observation_field(layers,observations):
    """Existing fault engine; retain bounded native footprint evidence per RFH."""
    field=layers["master_scalar"].copy()
    lookup={m["module_id"]:m for m in layers["accepted_modules"]}
    for observation in observations:
        module=lookup.get(observation.get("module_id"))
        if module is None:continue
        q=np.float32(module["active_polygon_rgb"])
        x1,y1=np.maximum(0,np.floor(q.min(0)).astype(int)-2)
        x2,y2=np.minimum([field.shape[1],field.shape[0]],np.ceil(q.max(0)).astype(int)+3)
        before=field[y1:y2,x1:x2].copy()
        diagnosis=observation.get("diagnostic_fault_type") or "SingleHotSpot"
        observation["applied_faults"]={}
        real_farm_apply_production_fault(field,module,observation,diagnosis,in_place=True)
        delta=field[y1:y2,x1:x2]-before
        peak=float(np.max(abs(delta),initial=0))
        support=np.uint8(abs(delta)>max(peak*.10,1e-5))*255
        contours,_=cv2.findContours(support,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        yy,xx=np.nonzero(support)
        observation["footprint_rgb"]={"status":"rendered synthetic relative scalar", "diagnosis":diagnosis,
            "threshold_fraction_of_peak":.10,"native_pixels":len(xx),
            "bbox":[int(xx.min()+x1),int(yy.min()+y1),int(xx.max()+x1+1),int(yy.max()+y1+1)] if len(xx) else None,
            "contours":[(c.reshape(-1,2)+[x1,y1]).tolist() for c in contours],
            "reference_id":observation.get("production_reference_ids",{}).get(diagnosis)}
        if diagnosis=="SingleHotSpot":observation["hotspot_footprint_rgb"]=json.loads(json.dumps(observation["footprint_rgb"]))
    return field

def real_farm_camera_observation_controls(state,image,targets,view):
    observations=state["camera_observations"]
    if not observations:return None
    ids=[o["observation_id"] for o in observations]
    key="real_farm_camera_observation_"+real_farm_source_identity(image)[:12]
    queued=state.pop("camera_select_observation_queue",None)
    if queued in ids:st.session_state[key]=queued
    if st.session_state.get(key) not in ids:st.session_state[key]=state.get("camera_selected_observation") if state.get("camera_selected_observation") in ids else ids[0]
    index=ids.index(st.session_state[key])
    nav=st.columns(2)
    if nav[0].button("Previous hotspot",disabled=len(ids)<2,use_container_width=True):st.session_state[key]=ids[(index-1)%len(ids)]
    if nav[1].button("Next hotspot",disabled=len(ids)<2,use_container_width=True):st.session_state[key]=ids[(index+1)%len(ids)]
    chosen=st.selectbox("Hotspot observation",ids,key=key)
    state["camera_selected_observation"]=chosen
    observation=next(o for o in observations if o["observation_id"]==chosen)
    module=next((m for m in targets if m["module_id"]==observation["module_id"]),None)
    st.caption(f"{chosen} · Module {observation['module_id']} · {observation.get('geometry_status','UNVERIFIED')}")
    if view=="Close View":
        options=["SingleHotSpot",*CLOSE_DIAGNOSTIC_FAULT_TYPES]
        if module and module.get("placement_kind")=="pv_point":
            options=["SingleHotSpot"]
            st.caption("Perimeter unresolved: hotspot symptom only. Module/section diagnoses require a fitted physical module.")
        current=observation.get("diagnostic_fault_type") or "SingleHotSpot"
        if current not in options:current="SingleHotSpot"
        diagnosis=st.selectbox("Actual fault for this hotspot",options,index=options.index(current),
            format_func=lambda code:"Thermal Hotspot Symptom / Unassigned" if code=="SingleHotSpot" else real_farm_diagnostic_label(code),
            key=f"{key}_{chosen}_diagnosis")
        severity=st.slider("Observation severity",1,10,int(observation.get("severity",7)),key=f"{key}_{chosen}_severity")
        observation.update(diagnostic_fault_type=None if diagnosis=="SingleHotSpot" else diagnosis,severity=severity)
        st.caption("Reference-backed morphology · synthetic placement and relative intensity" if diagnosis in {"SingleHotSpot","SingleDiode","SingleByPassed"} else "Synthetic diagnostic pattern · unvalidated")
    diagnosis=observation.get("diagnostic_fault_type")
    st.caption("Current diagnosis: "+(real_farm_diagnostic_label(diagnosis) if diagnosis else "Unassigned hotspot symptom"))
    return observation


def real_farm_camera_saved_geometry(state,image):
    """Reuse the newest matching local project; never replace edited live work."""
    capture=state.get("camera_capture",{})
    if state.get("modules") and not (
        capture.get("geometry")==real_farm_geometry_signature(state["modules"])
        and not state.get("geometry_review_triage")
        and state.get("camera_analysis_source")==real_farm_source_identity(image)
    ):return False
    identity=real_farm_source_identity(image)
    paths=sorted(REAL_FARM_PROJECT_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in paths:
        try:project=real_farm_load_project(path)
        except (OSError,ValueError):continue
        if project.get("source_rgb_pixel_sha256")!=identity or project.get("image")!={"width":image.width,"height":image.height}:continue
        if not project.get("modules"):continue
        for key in ("modules","module_tables","geometry_review_triage","full_farm_review","bounded_review",
                    "physical_geometry_review_required","panel_detection_method"):
            if key in project:state[key]=project[key]
        state["camera_geometry_origin"]={"type":"saved_source_matched_project","path":str(path),"source":identity}
        state["camera_recovery_message"]=""
        state["camera_observations"]=[]
        return True
    return False


def real_farm_camera_analyze(state,image,progress):
    """Recover once for a fresh upload; never replace existing reviewed geometry."""
    real_farm_camera_saved_geometry(state,image)
    if not state.get("modules"):
        if real_farm_source_is_reviewed(image):
            result=real_farm_legacy_recover_full_farm(state,image,progress)
        else:
            result=real_farm_recover_region_frames(state,image,progress)
        state["camera_recovery_message"]=result.get("message","")
        # Keep inferred visible support/vegetation holes on new proposals.
        # This attaches support, without accepting any physical boundary.
        vegetation=real_farm_material_masks(image)["vegetation"]
        visibility=np.uint8(vegetation==0)*255
        for m in state.get("modules",[]):
            if not m.get("refined_mask_contours_rgb"):
                unresolved=m.get("physical_module_unresolved",False)
                real_farm_attach_instance_support(m,visibility)
                m.update(requires_physical_review=True,physical_module_unresolved=unresolved)
        del vegetation,visibility
    state["camera_analysis_source"]=real_farm_source_identity(image)


def real_farm_camera_warm_background(layers,image,modules):
    """Use the existing sunlit substrate response outside physical PV only."""
    warm=real_farm_background_scalar(image,sunlit_terrain=True)
    outer=real_farm_module_mask(image.size,modules,active=False)
    distance=cv2.distanceTransform(np.uint8(outer==0),cv2.DIST_L2,3)
    # Keep an unchanged sampling guard around glass AND frames. The existing
    # optical/Far sampling kernels cannot pull edited substrate into PV pixels.
    for start in range(0,image.height,128):
        rows=slice(start,start+128)
        weight=np.clip((distance[rows]-6.)/4.,0,1)
        weight=weight*weight*(3.-2.*weight)
        delta=warm[rows]-layers["background_scalar"][rows]
        layers["master_scalar"][rows] += delta*weight


def real_farm_camera_build_layers(state,image):
    candidates,credible=real_farm_camera_candidates(state,image)
    # The established renderer consumes ephemeral simulation geometry. Clearing
    # its rendering gate on COPIES does not approve stored annotation geometry.
    structure_rejected=real_farm_camera_structure_checks(state,image,candidates)
    render_modules=[]
    for module in candidates:
        if module["module_id"] in structure_rejected:continue
        if module.get("mask_source")=="SEARCH_ROI" or module.get("refinement_status","accepted")!="accepted":continue
        if not real_farm_mechanical_validation(module.get("outer_polygon_rgb",[]),image.size)["mechanically_valid"]:continue
        item=dict(module)
        item.update(requires_physical_review=False,physical_module_unresolved=False)
        render_modules.append(item)
    # Recover the established native scene path; the experimental reference
    # profile imposed one cool PV mean and a hot generic substrate everywhere.
    # Defaults match the retained pre-experiment established-renderer output.
    layers=real_farm_build_thermal_master(image,render_modules,include_material_masks=False)
    real_farm_camera_warm_background(layers,image,candidates)
    for field in ("background_scalar","pv_scalar","accepted_parent_regions","rejected_modules","material_masks"):
        layers.pop(field,None)
    if structure_rejected or not candidates:
        layers["pv_module_union_mask"]=np.maximum(layers["pv_module_union_mask"],real_farm_material_masks(image)["pv_scene_candidate"])
    values=layers["master_scalar"]
    layers.update(accepted_modules=credible,camera_candidates=candidates,
        camera_source=real_farm_source_identity(image),proxy_acquired=False,
        mode="SYNTHETIC_RELATIVE_THERMAL_PROXY",
        scalar_audit={"model":"Current Thermal: established native material/scene/module renderer",
            "signal_units":"synthetic relative signal; not measured temperature",
            "native_canvas":list(image.size),"display_range":layers["display_range"],
            "p1_median_p99":[round(float(v),3) for v in np.percentile(values,[1,50,99])],
            "clipped_fraction":float(np.mean((values<=0)|(values>=255)))},
        camera_material_revision="real-background-sunlit-v13")
    return layers


def real_farm_camera_visible_delta(base, delta, support, fault_type):
    """Explicit synthetic visibility adaptation, before the source homography.

    Scale localized signatures around their own energy center, preserving
    orientation and shape. Extended/string/crack signatures keep their extent.
    The caller still clips to visible active PV, including occlusion holes.
    """
    scale={"SingleHotSpot":2.2,"SingleDiode":1.5}.get(fault_type,1.0)
    gain=2.6 if fault_type=="SingleHotSpot" else 2.2
    delta=np.asarray(delta,np.float32)
    support=np.asarray(support,np.float32)
    if scale!=1.0:
        energy=np.square(delta);total=float(energy.sum())
        if total>1e-8:
            h,w=delta.shape
            cx=float(energy.sum(0) @ np.arange(w,dtype=np.float32))/total
            cy=float(energy.sum(1) @ np.arange(h,dtype=np.float32))/total
            transform=np.float32([[scale,0,cx*(1-scale)],[0,scale,cy*(1-scale)]])
            delta=cv2.warpAffine(delta,transform,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            support=cv2.warpAffine(support,transform,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Preserve a rounded hot core and graded cold signatures, rather than
    # clipping the whole residual to an undifferentiated white/black patch.
    headroom=np.maximum(np.where(delta>=0,254.-base,base-25.),1.)
    delta=headroom*np.tanh(delta*gain/headroom)
    return delta.astype(np.float32),support,{"version":"legible-v1","gain":gain,
        "local_scale":scale,"signal":"synthetic relative scalar","training_verified":False}


def real_farm_camera_sensor(field,distance,support=None):
    """Acquire the established native scalar once, then sample the Far view."""
    field=apply_sensor_model(field,"real_farm",distance,support)
    if distance=="far":
        h,w=field.shape
        small=cv2.resize(field,(max(1,round(w*.5)),max(1,round(h*.5))),interpolation=cv2.INTER_AREA)
        return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)
    return field


def real_farm_camera_acquisition(layers,observations,distance,bounds=None,grayscale=False):
    config=json.dumps([{k:o.get(k) for k in ("observation_id","module_id","severity","seed","local_uv","diagnostic_fault_type","production_reference_ids","camera_visibility")} for o in observations],sort_keys=True)
    target_geometry=tuple((m["module_id"],real_farm_polygon_signature(m)) for m in layers.get("accepted_modules",[]) if
        m.get("simulation_only") and any(o.get("module_id")==m["module_id"] for o in observations))
    key=(id(layers["master_scalar"]),config,target_geometry,distance,bounds,grayscale,"real-hotspot-active-footprint-v1-audit")
    cache=st.session_state.get("real_farm_camera_acquisitions",{})
    audits=layers.setdefault("acquisition_audits",{})
    if not observations:layers.pop("acquisition_audit",None)
    if key in cache and (not observations or key in audits):
        if observations:layers["acquisition_audit"]=audits[key]
        return cache[key]
    field=real_farm_camera_observation_field(layers,observations) if observations else layers["master_scalar"]
    sensor=real_farm_camera_sensor(field,distance,layers["pv_module_union_mask"])
    # Inspect the same scalar planes actually sent to display, before palette.
    if observations:
        baseline=real_farm_camera_sensor(layers["master_scalar"],distance,layers["pv_module_union_mask"])
        crop=bounds or (0,0,field.shape[1],field.shape[0])
        x1,y1,x2,y2=crop
        pre=(field[y1:y2,x1:x2]-layers["master_scalar"][y1:y2,x1:x2])
        post=(sensor[y1:y2,x1:x2]-baseline[y1:y2,x1:x2])
        footprints={}
        for name,delta in (("pre_sensor",pre),("post_sensor",post)):
            yy,xx=np.nonzero(delta>max(float(delta.max())*.1,1e-5))
            footprints[name]={"tenth_peak_pixels":int(len(xx)),
                "bbox_rgb":[int(xx.min()+x1),int(yy.min()+y1),int(xx.max()+x1+1),int(yy.max()+y1+1)] if len(xx) else None}
        layers["acquisition_audit"]={"distance":distance,"bounds":list(crop),"display_range":layers["display_range"],
            "pre_sensor_peak":float(pre.max()),"post_sensor_peak":float(post.max()),
            "pre_sensor_energy":float(np.sum(pre,dtype=np.float64)),"post_sensor_energy":float(np.sum(post,dtype=np.float64)),
            "pre_sensor_half_peak_pixels":int(np.count_nonzero(pre>pre.max()*.5)),
            "post_sensor_half_peak_pixels":int(np.count_nonzero(post>post.max()*.5)),
            "scalar_footprint":footprints,
            "faults":[o.get("production_fault_debug",{}) for o in observations]}
        del baseline,pre,post
    if bounds:
        x1,y1,x2,y2=bounds;sensor=sensor[y1:y2,x1:x2]
    if grayscale:
        low,high=layers["display_range"]
        acquired=Image.fromarray(np.uint8(np.clip((sensor-low)/(high-low)*255.,0,255))).convert("RGB")
    else:
        acquired=real_farm_palette_image(sensor,layers["display_range"],distance)
    config=json.dumps([{k:o.get(k) for k in ("observation_id","module_id","severity","seed","local_uv","diagnostic_fault_type","production_reference_ids","camera_visibility")} for o in observations],sort_keys=True)
    target_geometry=tuple((m["module_id"],real_farm_polygon_signature(m)) for m in layers.get("accepted_modules",[]) if
        m.get("simulation_only") and any(o.get("module_id")==m["module_id"] for o in observations))
    key=(id(layers["master_scalar"]),config,target_geometry,distance,bounds,grayscale,"real-hotspot-active-footprint-v1-audit")
    # Keep at most two actual acquired images (normally Far and selected Close).
    # Adding target geometry to the cache key must not turn navigation into an
    # unbounded cache of native crops/images.
    cache={k:v for k,v in cache.items() if k[0]==key[0] and k[1]==key[1] and k[3]!=distance}
    cache[key]=acquired
    cache=dict(list(cache.items())[-2:])
    st.session_state["real_farm_camera_acquisitions"]=cache
    layers["acquisition_audits"]={k:v for k,v in audits.items() if k in cache}
    if observations:layers["acquisition_audits"][key]=layers["acquisition_audit"]
    return acquired


def real_farm_camera_workflow(state,image,section="Thermal"):
    # Camera capture uses the recovered Current renderer. A hidden Generated
    # Mode selector must not silently select a legacy sensor or module field.
    st.session_state["thermal_renderer_mode"]="Current Thermal"
    st.caption("Synthetic relative thermal · not measured temperature.")
    source=real_farm_source_identity(image)
    if st.session_state.get("real_farm_camera_ui_source")!=source:
        # A new upload starts at the full Far scene, with no hidden healthy/RGB
        # display flags or previous source's acquisition buffers/selection.
        for widget_key in list(st.session_state):
            if widget_key.startswith("real_farm_camera_"):st.session_state.pop(widget_key,None)
        st.session_state["real_farm_camera_ui_source"]=source
    key=(source,real_farm_geometry_signature(state.get("modules",[])),"real-background-sunlit-v13-workflow-v1")
    cached=st.session_state.get("real_farm_camera_layers")
    generate=st.button("Generate Thermal",type="primary",use_container_width=True) if section=="Thermal" or not cached else False
    if generate:
        st.session_state.pop("real_farm_camera_layers",None)
        st.session_state.pop("real_farm_acquired_images",None)
        st.session_state.pop("real_farm_thermal_master_layers",None)
        st.session_state.pop("real_farm_camera_acquisitions",None)
        cached=None
        with st.status("Generating thermal…",expanded=False) as status:
            try:
                real_farm_camera_analyze(state,image,status.write)
            except (ValueError,cv2.error) as exc:
                from pv_memory import is_allocation_failure
                if is_allocation_failure(exc):raise
                state["camera_recovery_message"]=str(exc)
                status.write("Module recovery incomplete. Whole-scene camera capture remains available.")
            layers=real_farm_camera_build_layers(state,image)
            key=(source,real_farm_geometry_signature(state.get("modules",[])),"real-background-sunlit-v13-workflow-v1")
            st.session_state["real_farm_camera_layers"]=(key,layers)
            state["camera_capture"]={"source":source,"geometry":key[1],"native_size":list(image.size),
                "automatic_candidates":len(layers["camera_candidates"]),"fault_position_candidates":len(layers["accepted_modules"]),
                "training_verified":False}
            status.update(label="Thermal ready",state="complete",expanded=False)
        cached=(key,layers)
    if not cached or cached[0][0]!=source:
        st.image(image,caption=f"Original RGB · {image.width} × {image.height}",use_container_width=True)
        return
    if cached[0]!=key:
        layers=real_farm_camera_build_layers(state,image)
        st.session_state["real_farm_camera_layers"]=(key,layers)
    else:layers=cached[1]
    candidates=layers["camera_candidates"]
    credible=list(layers.setdefault("credible_modules",list(layers["accepted_modules"])))
    provisional=real_farm_camera_simulation_targets(state,image,candidates,credible)
    targets=credible+provisional
    layers["accepted_modules"]=targets
    state.setdefault("camera_capture",{}).update(automatic_candidates=len(candidates),fault_position_candidates=len(targets),geometry_unreliable=bool(state.get("camera_structure_check",{}).get("scene_geometry_unreliable")),training_verified=False,mode=layers.get("mode"),scalar_audit=layers.get("scalar_audit",{}))
    st.caption(f"{len(candidates)} module proposals · {len(targets)} simulation targets · geometry unverified")
    origin=state.get("camera_geometry_origin",{})
    if origin.get("type")=="prepared_source_matched_project":
        st.caption("Prepared RGB geometry reused · "+origin.get("status","NEEDS REVIEW")+" · not user-verified ground truth")
    if layers.get("camera_surface_mode"):st.caption("Uncertain outlines may contain cell fragments. Thermal preview uses source-supported PV surfaces; these are not verified individual-module labels.")
    if state.get("camera_structure_check",{}).get("scene_geometry_unreliable") or not credible:
        st.caption("Some module boundaries could not be identified reliably. You can still simulate faults by selecting a module. Verified dataset export requires geometry review.")
    target_ids=[m["module_id"] for m in targets]
    queued=state["camera_simulation"].pop("selected_target",None)
    if queued in target_ids:st.session_state["real_farm_camera_target"]=queued
    if target_ids:
        if st.session_state.get("real_farm_camera_target") not in target_ids:
            center=np.float32(image.size)/2
            selected=min(targets,key=lambda m:np.linalg.norm(np.float32(m["outer_polygon_rgb"]).mean(0)-center))
            st.session_state["real_farm_camera_target"]=selected["module_id"]
        mid=st.selectbox("Placement module",target_ids,key="real_farm_camera_target",
            format_func=lambda value:value+(" · PROVISIONAL" if value.startswith("SIM-") else ""))
    else:mid=None
    selected=next((m for m in targets if m["module_id"]==mid),None)
    refresh=real_farm_camera_pick_target(state,image,selected) if section=="Faults" else False
    if selected and selected.get("simulation_only"):
        st.caption("PROVISIONAL — exploratory simulation; excluded from verified training labels.")
    real_farm_camera_observation_source(state,image)
    observations=state["camera_observations"]
    if section=="Faults":
        if st.button("Add hotspot observation",disabled=not mid,use_container_width=True):
            real_farm_camera_new_observation(state,image,selected)
        if st.button("Clear Camera Faults",disabled=not observations):
            state["camera_observations"]=[];observations=[]
    target_lookup={m["module_id"]:m for m in targets}
    for observation in observations:
        if observation["module_id"] in target_lookup:
            observation.setdefault("target_geometry",json.loads(json.dumps(target_lookup[observation["module_id"]])))
        observation.pop("camera_visibility",None)
    view=st.radio("Camera acquisition",["Far View","Close View"],horizontal=True,key="real_farm_camera_distance")
    active_observation=real_farm_camera_observation_controls(state,image,targets,view)
    if active_observation:mid=active_observation["module_id"]
    unavailable=[o for o in observations if o.get("module_id") not in target_lookup or
        (target_lookup[o["module_id"]].get("placement_kind")=="pv_point" and o.get("diagnostic_fault_type"))]
    if unavailable:st.warning("Some observations lack the geometry needed for their selected diagnosis. Their records are retained; use hotspot-only placement or review that target.")
    observations=[o for o in observations if o not in unavailable]
    secondary=st.columns(2)
    original=secondary[0].checkbox("View Original RGB",value=False,key="real_farm_camera_original")
    outlines=secondary[1].checkbox("Show Module Outlines",value=False,key="real_farm_camera_outlines")
    diagnostics=st.expander("Diagnostics",expanded=False)
    with diagnostics:
        grayscale=st.checkbox("Grayscale Signal",value=False,key="real_farm_camera_grayscale")
        healthy_only=st.checkbox("Healthy Baseline",value=False,key="real_farm_camera_healthy")
    display="Original RGB" if original else "Grayscale Signal" if grayscale else "Synthetic Thermal"
    bounds=None
    selected=next((m for m in targets if m["module_id"]==mid),None)
    if display=="Original RGB":shown=image;caption="Full original RGB"
    elif display=="Automatic Modules":
        if state.get("camera_structure_check",{}).get("scene_geometry_unreliable"):
            shown=image;caption="Unreliable cell-grid proposals withheld; no physical module labels available"
        else:
            shown=real_farm_annotated_preview(image,candidates,mid,max_width=image.width,show_labels=True)
            caption=("Saved module geometry" if state.get("camera_geometry_origin") else "Automatic module proposals")+" · unverified · amber requires review"
    else:
        try:
            bounds=None
            if view=="Close View":
                if selected:
                    context=candidates
                    if selected.get("simulation_only"):
                        area=abs(real_farm_polygon_area(selected["outer_polygon_rgb"]))
                        context=[selected]+[m for m in candidates if abs(real_farm_polygon_area(m["outer_polygon_rgb"]))>=area*.5]
                    bounds=real_farm_close_context_bounds(selected,context,(image.height,image.width),.75 if selected.get("simulation_only") else .25)
                    if selected.get("simulation_only"):
                        cx,cy=np.float32(selected["active_polygon_rgb"]).mean(0)
                        radius=selected["simulation_context_radius"]
                        bounds=(max(0,min(bounds[0],int(cx-radius))),max(0,min(bounds[1],int(cy-radius))),
                                min(image.width,max(bounds[2],int(cx+radius+1))),min(image.height,max(bounds[3],int(cy+radius+1))))
                else:
                    w,h=image.size;bounds=(max(0,w//2-500),max(0,h//2-350),min(w,w//2+500),min(h,h//2+350))
            shown=real_farm_camera_acquisition(layers,[] if healthy_only else observations,"close" if bounds else "far",bounds,grayscale=display=="Grayscale Signal")
        except FaultGeometryError as exc:
            st.error(f"Synthetic fault could not be rendered accurately: {exc}")
            shown=real_farm_camera_acquisition(layers,[],"close" if bounds else "far",bounds,grayscale=display=="Grayscale Signal")
        caption=(f"Close View · native-source thermal detail · {shown.width} × {shown.height}" if bounds else f"Far View · {image.width} × {image.height}")
        caption += " · Healthy" if healthy_only else " · With synthetic fault" if observations else " · Healthy"
    if outlines:
        overlay=np.array(shown);offset=np.float32(bounds[:2] if bounds else (0,0))
        for module in candidates+provisional:
            polygon=np.rint(np.float32(module["outer_polygon_rgb"])-offset).astype(np.int32)
            cv2.polylines(overlay,[polygon],True,(0,255,255) if module["module_id"]==mid else (255,180,30),1,cv2.LINE_AA)
        shown=Image.fromarray(overlay)
    st.image(shown,caption=caption,width=shown.width,output_format="PNG")
    st.download_button("Download camera view",export_current_view_bytes(shown),file_name=f"camera_{display.replace(' ','_')}_{view.replace(' ','_')}.png",mime="image/png")
    with diagnostics:
        st.json(layers.get("scalar_audit",{}))
        if layers.get("acquisition_audit"):
            st.download_button("Download scalar comparison",json.dumps(layers["acquisition_audit"],indent=2),file_name="camera_scalar_comparison.json",mime="application/json")
        st.download_button("Download exact scalar PV mask",export_current_view_bytes(Image.fromarray(layers.get("camera_composited_pv_mask",layers["pv_module_union_mask"]))),file_name="scalar_PV_mask_UNVERIFIED.png",mime="image/png")
        st.download_button("Download module outlines",export_current_view_bytes(real_farm_annotated_preview(image,candidates,max_width=image.width,show_labels=False)),file_name="module_proposals_UNVERIFIED.png",mime="image/png")
        st.download_button("Download original RGB",export_current_view_bytes(image),file_name="original_RGB.png",mime="image/png")
        st.download_button("Download pre-palette proxy codes",export_current_view_bytes(Image.fromarray(np.uint8(np.clip(layers["master_scalar"],0,255)))),file_name="synthetic_proxy_codes_NOT_TEMPERATURE.png",mime="image/png")
        st.caption(f"{len(candidates)} module proposals · {len(targets)} simulation targets. Synthetic relative signal, not measured temperature. Training-label export requires separate explicit verification.")
        st.download_button("Download unverified analysis",real_farm_project_json(state),file_name="camera_analysis_UNVERIFIED.json",mime="application/json")
    if refresh:st.rerun()
