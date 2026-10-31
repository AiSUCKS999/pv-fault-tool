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
    if cached.get("source")==source and cached.get("geometry")==signature and cached.get("version")==3:return cached["rejected"]
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
    # A weak region must not veto unrelated modules that pass this same check.
    # Preserve individual rejections and all subsequent placement safeguards.
    state["camera_structure_check"]={"version":3,"source":source,"geometry":signature,"rejected":rejected,
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
    if nav[0].button("Previous fault",disabled=len(ids)<2,use_container_width=True):st.session_state[key]=ids[(index-1)%len(ids)]
    if nav[1].button("Next fault",disabled=len(ids)<2,use_container_width=True):st.session_state[key]=ids[(index+1)%len(ids)]
    diagnosis_by_id={o["observation_id"]:o.get("diagnostic_fault_type") or "SingleHotSpot" for o in observations}
    chosen=st.selectbox("Fault observation",ids,key=key,
        format_func=lambda value:f"{value} · {fault_display_name(diagnosis_by_id[value])}")
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
        diagnosis=st.selectbox("Actual fault for this observation",options,index=options.index(current),
            format_func=lambda code:"Thermal Hotspot Symptom / Unassigned" if code=="SingleHotSpot" else real_farm_diagnostic_label(code),
            key=f"{key}_{chosen}_diagnosis")
        severity=st.slider("Observation severity",1,10,int(observation.get("severity",7)),key=f"{key}_{chosen}_severity")
        st.caption("Reference-backed morphology · synthetic placement and relative intensity" if diagnosis in {"SingleHotSpot","SingleDiode","SingleByPassed"} else "Synthetic diagnostic pattern · unvalidated")
        apply_col,remove_col=st.columns(2)
        if apply_col.button("Generate / Apply Fault",type="primary",use_container_width=True,
                            key=f"{key}_{chosen}_apply"):
            observation.update(diagnostic_fault_type=None if diagnosis=="SingleHotSpot" else diagnosis,severity=severity)
            state["updated_at"]=real_farm_now()
        if remove_col.button("Remove / Reset Fault",disabled=not observation.get("diagnostic_fault_type"),
                             use_container_width=True,key=f"{key}_{chosen}_remove"):
            observation["diagnostic_fault_type"]=None
            observation.pop("applied_faults",None)
            st.session_state.pop(f"{key}_{chosen}_diagnosis",None)
            st.session_state.pop(f"{key}_{chosen}_severity",None)
            state["updated_at"]=real_farm_now()
            st.rerun()
    diagnosis=observation.get("diagnostic_fault_type")
    st.caption("Current diagnosis: "+(real_farm_diagnostic_label(diagnosis) if diagnosis else "Unassigned hotspot symptom"))
    return observation


def real_farm_camera_saved_geometry(state,image):
    """Reuse the newest matching local project; never replace edited live work."""
    # A camera refresh must not replace live geometry or erase observations.
    if state.get("modules"):return False
    identity=real_farm_source_identity(image)
    if state.get("camera_prepared_source")==identity:return False
    prepared_root=ROOT/"assets"/"real_farm_prepared"
    index=prepared_root/"index.json"
    if index.is_file():
        try:
            entry=json.loads(index.read_text(encoding="utf-8")).get("sources",{}).get(identity)
            if entry and entry.get("image_size")==list(image.size):
                path=(prepared_root/entry["project"]).resolve()
                if not path.is_relative_to(prepared_root.resolve()):raise ValueError("Prepared project leaves dataset directory")
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["project_sha256"]:raise ValueError("Prepared project hash changed")
                project=real_farm_load_project(path)
                if project.get("source_rgb_pixel_sha256")!=identity or project.get("image")!={"width":image.width,"height":image.height}:raise ValueError("Prepared project/source mismatch")
                origin_type="prepared_source_matched_project"
                # A later correction saved through the existing review workflow
                # takes precedence over an older prepared copy of this source.
                newer=[] if entry.get("frozen") or entry.get("pinned") else sorted(REAL_FARM_PROJECT_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
                for saved in newer:
                    if saved.stat().st_mtime<=path.stat().st_mtime:continue
                    try:updated=real_farm_load_project(saved)
                    except (OSError,ValueError):continue
                    if updated.get("source_rgb_pixel_sha256")==identity and updated.get("image")==project["image"] and updated.get("modules"):
                        project,path,origin_type=updated,saved,"saved_source_matched_project"
                        break
                for key in ("modules","module_tables","geometry_review_triage","full_farm_review","bounded_review","physical_geometry_review_required","panel_detection_method","camera_simulation","camera_geometry_partial","dataset_preparation","camera_preparation_response"):
                    if key in project:state[key]=project[key]
                state.update(camera_prepared_source=identity,camera_geometry_origin={"type":origin_type,"path":str(path),"source":identity,"status":entry["status"] if origin_type=="prepared_source_matched_project" else "SAVED GEOMETRY"},camera_recovery_message="",camera_observations=[])
                return True
        except (OSError,ValueError,KeyError,TypeError) as exc:
            state["camera_recovery_message"]="Prepared geometry could not be reused: "+str(exc)
    paths=sorted(REAL_FARM_PROJECT_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in paths:
        try:project=real_farm_load_project(path)
        except (OSError,ValueError):continue
        if project.get("source_rgb_pixel_sha256")!=identity or project.get("image")!={"width":image.width,"height":image.height}:continue
        if not project.get("modules"):continue
        for key in ("modules","module_tables","geometry_review_triage","full_farm_review","bounded_review",
                    "physical_geometry_review_required","panel_detection_method","camera_preparation_response"):
            if key in project:state[key]=project[key]
        state["camera_geometry_origin"]={"type":"saved_source_matched_project","path":str(path),"source":identity}
        state["camera_recovery_message"]=""
        state["camera_observations"]=[]
        return True
    # Existing corrected baselines are also saved geometry. Do not rerun a
    # coarse recovery on their exact source and lose physical module splits.
    registry=ROOT/"assets"/"geometry_reviews"/"real_farm_iteration4.json"
    if registry.exists():
        records=json.loads(registry.read_text(encoding="utf-8-sig")).get("sources",[])
        baseline_root=(ROOT/"artifacts"/"real_farm_geometry_iteration4").resolve()
        for record in records:
            if record.get("rgb_sha256")!=identity or record.get("image_size")!=list(image.size):continue
            scene=record.get("scene","")
            if not scene or Path(scene).name!=scene or scene in {".",".."}:continue
            # Retain the project's existing D: junction for the Long baseline.
            baseline=baseline_root/scene/"after_modules.json"
            if not baseline.is_file():continue
            modules=json.loads(baseline.read_text(encoding="utf-8-sig"))
            if not isinstance(modules,list) or not modules:continue
            state.update(modules=modules,physical_geometry_review_required=True,
                panel_detection_method="SOURCE_MATCHED_SAVED_GEOMETRY",
                camera_geometry_origin={"type":"saved_source_matched_baseline","path":str(baseline),"source":identity},
                camera_recovery_message="",camera_observations=[])
            return True
    return False


def real_farm_camera_analyze(state,image,progress):
    """Recover once for a fresh upload; never replace existing reviewed geometry."""
    real_farm_camera_saved_geometry(state,image)
    if not state.get("modules") and state.get("camera_prepared_source")!=real_farm_source_identity(image):
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


def real_farm_camera_surface_regions(image):
    """Camera-only PV surface envelopes, never individual module annotations."""
    from pv_module_recovery import material_mask
    rgb=real_farm_rgb_array(image)
    support=material_mask(rgb)
    # Same local PV evidence as click placement; retain glass under cyan glare.
    for start in range(0,image.height,128):
        rows=slice(start,start+128)
        r,g,b=(rgb[rows,:,i].astype(np.int16) for i in range(3))
        support[rows]=np.maximum(support[rows],np.uint8((b>=g-4)&(b>r+6))*255)
    # Existing native 3px grouping, not preview-scale dilation or grid slicing.
    support=cv2.morphologyEx(support,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(support,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    regions=[]
    for contour in contours:
        if cv2.contourArea(contour)<120:continue
        q=cv2.boxPoints(cv2.minAreaRect(contour))
        # Keep the actual table transform across the image boundary. Raster
        # ROIs clip its visible support; clamping corners shears the surface.
        if not real_farm_mechanical_validation(q,image.size)["mechanically_valid"]:continue
        record=real_farm_module_record("SURFACE-"+str(len(regions)+1),q.tolist(),image.size,inset_ratio=0.,provenance_note="camera_surface_envelope_not_a_module")
        # Keep the actual contour, including occlusion holes, inside its envelope.
        # Bright neutral frames/glare inside this connected PV hull are not
        # holes in the glass. Preserve vegetation and genuine dark occlusions.
        low=np.maximum(0,np.floor(q.min(0)).astype(int));high=np.minimum(image.size,np.ceil(q.max(0)).astype(int)+1)
        x1,y1=low;x2,y2=high
        local=support[y1:y2,x1:x2].copy()
        hsv=cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2HSV)
        hull=np.zeros(local.shape,np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(contour).reshape(-1,2)-low,255)
        local[(hull>0)&(hsv[:,:,1]<48)&(hsv[:,:,2]>82)]=255
        # Blue cast shadows are not glass. Use a lower-tail compatibility
        # guard relative to the brighter observed glass, not the former 5th
        # percentile cutoff that bisected normal cell texture. This is support
        # classification only; RGB brightness never sets the thermal signal.
        gray=cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
        observed=gray[local>0]
        if len(observed):local[gray<.7*float(np.percentile(observed,75))]=0
        # Update the one shared support plane; attachment crops its own ROI.
        support[y1:y2,x1:x2]=local
        real_farm_attach_instance_support(record,support)
        record.update(camera_surface_only=True,camera_table_surface=True,requires_physical_review=False,physical_module_unresolved=False)
        regions.append(record)
    return regions


def real_farm_camera_source_frames(layers,image,regions):
    """Map only observed neutral frame/cell ridges into the existing cool response.

    Known cell fragments must not create their own procedural perimeter. Sample
    the established healthy model's frame response instead of inventing lines.
    Work in native source ROIs; fine source structure is never enlarged first.
    """
    rgb=real_farm_rgb_array(image);field=layers["master_scalar"]
    for region in regions:
        q=np.float32(region["outer_polygon_rgb"])
        x1,y1=np.maximum(0,np.floor(q.min(0)).astype(int))
        x2,y2=np.minimum([image.width,image.height],np.ceil(q.max(0)).astype(int)+1)
        part=rgb[y1:y2,x1:x2]
        if not part.size:continue
        neutral=part.min(2).astype(np.float32)
        # Positive source ridges locate real reflective frames/busbars. This
        # changes apparent frame response, not a brightness-to-temperature map.
        detail=np.maximum(neutral-cv2.GaussianBlur(neutral,(0,0),1.0),0)
        visible=real_farm_module_mask(image.size,[region],True,(int(x1),int(y1),int(x2),int(y2)))>0
        values=detail[visible]
        if not len(values):continue
        ridge=np.clip(detail/max(2.,float(np.percentile(values,98))),0,1)
        # Suppress texture inside blue glass; retain neutral physical structures.
        chroma=part.max(2).astype(float)-part.min(2)
        ridge*=np.clip((65.-chroma)/45.,0,1)*visible
        model=healthy_thermal_module_field((128,192),seed_for(tuple(q.ravel()),"source-frame-response"),164,196,interior_detail=.65)
        frame=float(np.median(np.r_[model[0,:],model[-1,:],model[:,0],model[:,-1]]))
        roi=field[y1:y2,x1:x2]
        roi-=np.maximum(roi-frame,0)*ridge
    return field


def real_farm_camera_module_edge_response(layers,image,modules,require_frame_evidence=False,edge_width_fraction=None):
    """Thin, inner-only physical-frame cooling; no geometry or mask changes.

    One max-strength response at shared seams/corners prevents doubled borders.
    Uncertain polygons receive cooling only at existing source-supported ridges.
    All samples read the unchanged healthy field before sparse min-composition.
    """
    field=layers["master_scalar"];visible=layers["pv_module_union_mask"]
    rgb=real_farm_rgb_array(image);updates=[]
    for module in modules:
        outer=np.float32(module.get("outer_polygon_rgb",[]))
        active=np.float32(module.get("active_polygon_rgb",[]))
        if outer.shape!=(4,2) or active.shape!=(4,2):continue
        short=float(min(np.linalg.norm(np.roll(outer,-1,axis=0)-outer,axis=1)))
        if short<4:continue
        low=np.maximum(0,np.floor(outer.min(0)).astype(int)-3)
        high=np.minimum(image.size,np.ceil(outer.max(0)).astype(int)+4)
        x1,y1=low;x2,y2=high
        shape=(int(y2-y1),int(x2-x1))
        if min(shape)<=0:continue
        yy,xx=np.mgrid[:shape[0],:shape[1]].astype(np.float32)
        distance=np.full(shape,np.inf,np.float32)
        inset=np.zeros(shape,np.float32)
        frame_evidence=np.ones(shape,np.float32)
        if require_frame_evidence:
            part=rgb[y1:y2,x1:x2]
            neutral=part.min(2).astype(np.float32)
            ridge=np.clip((neutral-cv2.GaussianBlur(neutral,(0,0),1.0))/8.,0,1)
            chroma=part.max(2).astype(np.float32)-neutral
            frame_evidence=ridge*np.clip((82.-chroma)/50.,0,1)
        for i in range(4):
            a,b=outer[i]-low,outer[(i+1)%4]-low;edge=b-a
            t=np.clip(((xx-a[0])*edge[0]+(yy-a[1])*edge[1])/max(float(edge@edge),1e-6),0,1)
            d=np.hypot(xx-a[0]-t*edge[0],yy-a[1]-t*edge[1])
            if require_frame_evidence:
                # Do not cool an uncertain diagonal/cell-fragment edge merely
                # because it crosses a few real frames. Require sustained
                # source ridge support along that existing polygon side.
                along=np.linspace(.08,.92,max(8,int(np.linalg.norm(edge))))
                ex=np.clip(np.rint(a[0]+along*edge[0]).astype(int),0,shape[1]-1)
                ey=np.clip(np.rint(a[1]+along*edge[1]).astype(int),0,shape[0]-1)
                if np.mean(frame_evidence[ey,ex]>.25)<.45:continue
            rim=np.linalg.norm(active[i]-outer[i])*(1-t)+np.linalg.norm(active[(i+1)%4]-outer[(i+1)%4])*t
            inset=np.where(d<distance,rim,inset);distance=np.minimum(distance,d)
        inside=np.zeros(shape,np.uint8)
        cv2.fillConvexPoly(inside,np.rint(outer-low).astype(np.int32),1)
        seed=seed_for(tuple(np.round(outer,2).ravel()),"physical-frame-continuity")
        variation=thermal_coherent_field(shape,seed,(5,7))
        widths=thermal_coherent_field(shape,seed_for(seed,"width"),(4,6))
        # Source-native pixel width follows projected dimensions. The rim and
        # inner diffusion form ONE band, not separate outer/active outlines.
        width=np.clip(short*.04,.8,2.)
        if edge_width_fraction is not None:
            # A source-bound frame measurement can retain its projected width
            # at native resolution instead of the small-preview two-pixel cap.
            width=np.clip(short*float(edge_width_fraction),.8,max(2.,short*.06))
        width*=np.clip(1+widths*.12,.8,1.2)
        reach=np.minimum(short*.12,np.maximum(width*2.,inset+width))
        t=np.clip(distance/np.maximum(reach,.1),0,1)
        weight=(1-t)*(1-t)*(1+2*t)
        weight*=np.clip(.92+variation*.08,.72,1.)
        if require_frame_evidence:weight*=frame_evidence
        weight*=np.uint8((inside>0)&(visible[y1:y2,x1:x2]>0))
        roi=field[y1:y2,x1:x2]
        # Same apparent-frame response as the existing continuity model.
        ceiling=roi-np.maximum(roi-(99.+variation*4.),0)*weight
        y,x=np.nonzero(ceiling<roi)
        updates.append((y+y1,x+x1,ceiling[y,x]))
    for y,x,ceiling in updates:field[y,x]=np.minimum(field[y,x],ceiling)
    return field


def real_farm_camera_physical_rim_support(layers,image,modules):
    """Keep confirmed frame material out of the substrate composition.

    Extend the adjacent healthy PV field across its physical rim before the
    existing edge response. Read one unchanged plane; average shared rims.
    Active glass, occlusion holes and exterior pixels are never replaced.
    """
    if not modules:return
    quads=np.concatenate([np.float32(m["outer_polygon_rgb"]) for m in modules])
    low=np.maximum(0,np.floor(quads.min(0)).astype(int)-1)
    high=np.minimum(image.size,np.ceil(quads.max(0)).astype(int)+2)
    x1,y1=map(int,low);x2,y2=map(int,high)
    if x2<=x1 or y2<=y1:return
    field=layers["master_scalar"];original=field[y1:y2,x1:x2].copy()
    bounds=(x1,y1,x2,y2);total=np.zeros_like(original);count=np.zeros(original.shape,np.uint8)
    all_glass=real_farm_module_mask(image.size,modules,active=True,bounds=bounds)>0
    for module in modules:
        active=real_farm_module_mask(image.size,[module],active=True,bounds=bounds)>0
        if not np.any(active):continue
        outer=real_farm_module_mask(image.size,[module],active=False,bounds=bounds)>0
        rim=outer&~all_glass
        for hole in module.get("refined_mask_holes_rgb",[]):
            excluded=np.zeros(original.shape,np.uint8)
            cv2.fillPoly(excluded,[np.rint(np.float32(hole)-low).astype(np.int32)],1)
            rim[excluded>0]=False
        _,labels=cv2.distanceTransformWithLabels(np.uint8(~active),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        yy,xx=np.nonzero(active);lookup=np.zeros(int(labels.max())+1,np.float32)
        lookup[labels[yy,xx]]=original[yy,xx]
        total[rim]+=lookup[labels[rim]];count[rim]+=1
    valid=count>0;field[y1:y2,x1:x2][valid]=total[valid]/count[valid]


def real_farm_camera_uncovered_surfaces(layers,image,modules):
    """Render isolated source-supported PV omissions, without annotation repair.

    An absent detector polygon must not assign normal glass the generic cool
    blue-object background prior. Reuse the existing healthy surface model only
    for a module-sized, source-supported gap between comparable PV surfaces.
    These envelopes never enter selectable geometry or verified export labels.
    """
    from pv_module_recovery import material_mask
    if not modules:return
    rgb=real_farm_rgb_array(image)
    outer=real_farm_module_mask(image.size,modules,active=False)
    missing=cv2.bitwise_and(material_mask(rgb),cv2.bitwise_not(outer))
    missing=cv2.morphologyEx(missing,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    missing[outer>0]=0
    contours,_=cv2.findContours(missing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    quads=[np.float32(m["outer_polygon_rgb"]) for m in modules]
    typical=float(np.median([cv2.contourArea(q) for q in quads]))
    centers=np.array([q.mean(0) for q in quads])
    regions=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not .55*typical<=area<=1.8*typical:continue
        x,y,w,h=cv2.boundingRect(contour)
        # Partial image-edge surfaces and thin inter-table strips are not gaps.
        if x<=0 or y<=0 or x+w>=image.width or y+h>=image.height:continue
        q=cv2.boxPoints(cv2.minAreaRect(contour))
        lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        short=float(min(lengths))
        if area/max(cv2.contourArea(q),1)<.8:continue
        adjacent=[]
        for side in range(4):
            points=q[side]+np.linspace(.2,.8,9)[:,None]*(q[(side+1)%4]-q[side])
            # Check existing physical surfaces just beyond each proposed side.
            outward=points-q.mean(0)
            outward/=np.maximum(np.linalg.norm(outward,axis=1)[:,None],1.)
            points=np.rint(points+outward*max(3.,short*.08)).astype(int)
            points=np.clip(points,[0,0],[image.width-1,image.height-1])
            adjacent.append(np.mean(outer[points[:,1],points[:,0]]>0)>.5)
        if not ((adjacent[0] and adjacent[2]) or (adjacent[1] and adjacent[3])):continue
        # Do not reinterpret a genuinely dark/shaded/covered surface as healthy.
        neighbors=np.argsort(np.linalg.norm(centers-q.mean(0),axis=1))[:3]
        level=[]
        for index in neighbors:
            center=np.rint(centers[index]).astype(int)
            level.append(float(np.median(rgb[max(0,center[1]-3):center[1]+4,max(0,center[0]-3):center[0]+4])))
        local=rgb[y:y+h,x:x+w];support=missing[y:y+h,x:x+w]>0
        if not .7<=float(np.median(local[support]))/max(float(np.median(level)),1.)<=1.4:continue
        # Same support rule as existing camera surface envelopes: neutral
        # reflective frame pixels inside the PV hull are not substrate holes.
        hull=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(contour).reshape(-1,2)-[x,y],255)
        hsv=cv2.cvtColor(local,cv2.COLOR_RGB2HSV)
        visible=(hull>0)&(hsv[:,:,1]<48)&(hsv[:,:,2]>82)&(outer[y:y+h,x:x+w]==0)
        missing[y:y+h,x:x+w][visible]=255
        record=real_farm_module_record("SURFACE-GAP-"+str(len(regions)+1),q.tolist(),image.size,inset_ratio=0.,provenance_note="camera_surface_envelope_not_a_module")
        real_farm_attach_instance_support(record,missing)
        record.update(camera_surface_only=True,requires_physical_review=False,physical_module_unresolved=False)
        regions.append(record)
    if not regions:return
    # Native source coordinates and the established scalar renderer: no palette
    # edits, flat fills, new fault models, or changes to existing module pixels.
    scalar,support=real_farm_module_scalar_layer(image,regions,interior_detail=.65)
    real_farm_camera_module_edge_response({"master_scalar":scalar,"pv_module_union_mask":support},image,regions)
    support[outer>0]=0
    field=layers["master_scalar"]
    for start in range(0,image.height,128):
        rows=slice(start,start+128);valid=support[rows]>0
        field[rows][valid]=scalar[rows][valid]
        layers["pv_module_union_mask"][rows][valid]=255
    layers["camera_uncovered_surface_count"]=len(regions)


def real_farm_camera_build_layers(state,image):
    candidates,credible=real_farm_camera_candidates(state,image)
    response=state.get("camera_preparation_response",{})
    native_frames=(response.get("mode")=="native_physical_frames" and
        response.get("source")==real_farm_source_identity(image) and
        response.get("geometry")==real_farm_geometry_signature(state.get("modules",[])))
    # The established renderer consumes ephemeral simulation geometry. Clearing
    # its rendering gate on COPIES does not approve stored annotation geometry.
    structure_rejected=real_farm_camera_structure_checks(state,image,candidates)
    render_modules=[]
    # A known-unreliable proposal must not create a synthetic physical frame.
    # Saved supported geometry retains the established per-module path.
    uncertain=bool(state.get("camera_geometry_partial")) or not candidates or len(structure_rejected)/len(candidates)>.25
    surfaces=real_farm_camera_surface_regions(image) if uncertain else []
    for module in credible if uncertain else candidates:
        if module.get("mask_source")=="SEARCH_ROI" or module.get("refinement_status","accepted")!="accepted":continue
        if not real_farm_mechanical_validation(module.get("outer_polygon_rgb",[]),image.size)["mechanically_valid"]:continue
        item=dict(module);item.update(requires_physical_review=False,physical_module_unresolved=False)
        if native_frames and module["module_id"] in response.get("physical_frame_support_ids",[]):
            item["camera_native_raster_support"]=True
        render_modules.append(item)
    if native_frames and not uncertain:
        # A source-bound, inspected small-module preparation supplies physical
        # perimeters. Use one native frame response instead of stacking the
        # canonical edge field, rim reconciliation and continuity cooling.
        # The existing source background and module interiors remain in use.
        layers=real_farm_build_thermal_master(image,render_modules,include_material_masks=False,interior_detail=.65,edge_cooling_scale=0.)
        if response.get("source_cell_ridges_are_neutral_glass") is not True:
            real_farm_camera_source_frames(layers,image,render_modules)
        rim_ids=set(response.get("physical_frame_support_ids",[]))
        if rim_ids:real_farm_camera_physical_rim_support(layers,image,[m for m in render_modules if m["module_id"] in rim_ids])
        real_farm_camera_module_edge_response({"master_scalar":layers["master_scalar"],
            "pv_module_union_mask":real_farm_module_mask(image.size,render_modules,active=False)},image,render_modules,
            edge_width_fraction=response.get("physical_frame_width_fraction"))
    elif uncertain:
        # Reuse the active thermal model on source-supported surface envelopes.
        # No envelope is promoted to a selectable/verified physical module.
        layers=real_farm_build_thermal_master(image,surfaces,include_material_masks=False,interior_detail=.65,edge_cooling_scale=0.)
        # Preserve the existing material/exposure field. Forcing the sunlit
        # preset over every uncertain upload was saturating all ground lanes.
        real_farm_camera_source_frames(layers,image,surfaces)
    else:
        layers=real_farm_build_thermal_master(image,render_modules,include_material_masks=False,interior_detail=.65)
        real_farm_camera_warm_background(layers,image,candidates)
        real_farm_reconcile_frame_rims(layers["master_scalar"],image,render_modules)
        real_farm_camera_uncovered_surfaces(layers,image,render_modules)
    # Cell-grid proposals are not physical module perimeters. In uncertain
    # scenes keep the source-frame response above, without stamping proposal
    # edges into the thermal signal. Supported module scenes retain their rims.
    if not uncertain and not native_frames:real_farm_camera_module_edge_response(layers,image,candidates)
    layers["camera_surface_mode"]=bool(uncertain)
    for field in ("background_scalar","pv_scalar","accepted_parent_regions","rejected_modules","material_masks"):
        layers.pop(field,None)
    layers["camera_composited_pv_mask"]=layers["pv_module_union_mask"]
    if structure_rejected or not candidates:
        layers["pv_module_union_mask"]=np.maximum(layers["pv_module_union_mask"],real_farm_material_masks(image)["pv_scene_candidate"])
    values=layers["master_scalar"]
    layers.update(accepted_modules=credible,camera_candidates=candidates,
        camera_source=real_farm_source_identity(image),proxy_acquired=False,
        mode="SYNTHETIC_RELATIVE_THERMAL_PROXY",
        scalar_audit={"model":"Current Thermal: native source-supported surfaces" if uncertain else "Current Thermal: established native material/scene/module renderer",
            "signal_units":"synthetic relative signal; not measured temperature",
            "native_canvas":list(image.size),"display_range":layers["display_range"],
            "p1_median_p99":[round(float(v),3) for v in np.percentile(values,[1,50,99])],
            "clipped_fraction":float(np.mean((values<=0)|(values>=255)))},
        camera_material_revision="real-native-source-surfaces-v24-edges-v1-gap-v1-table-v2")
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


def real_farm_camera_image_view_changed(image):
    from pv_real_farm_rgb_view import view_mode_changed
    view_mode_changed(st)
    if st.session_state.get("real_farm_camera_display") in {"Thermal","RGB + Thermal"}:
        st.session_state["real_farm_camera_pending_result_source"]=real_farm_source_identity(image)


def real_farm_camera_show_thermal_result(image,paired=False):
    """Select the fault result before widgets render; prepare pixels on rerun."""
    st.session_state["real_farm_camera_display"]="RGB + Thermal" if paired else "Thermal"
    st.session_state["real_farm_camera_distance"]="Far View"
    real_farm_camera_image_view_changed(image)


def real_farm_camera_workflow(state,image,section="Thermal"):
    # Camera capture uses the recovered Current renderer. A hidden Generated
    # Mode selector must not silently select a legacy sensor or module field.
    st.session_state["thermal_renderer_mode"]="Current Thermal"
    from pv_real_farm_rgb_view import annotated_rgb_fault_preview, view_bounds
    from pv_scene_frames import presentation_panel
    presentation_panel(globals(), "Real Farm")
    source=real_farm_source_identity(image)
    if st.session_state.get("real_farm_camera_ui_source")!=source:
        # A new upload starts at the full Far scene, with no hidden healthy/RGB
        # display flags or previous source's acquisition buffers/selection.
        for widget_key in list(st.session_state):
            if widget_key.startswith("real_farm_camera_"):st.session_state.pop(widget_key,None)
        st.session_state["real_farm_camera_ui_source"]=source
    display_key="real_farm_camera_display"
    if display_key not in st.session_state:
        st.session_state[display_key]="Original RGB" if st.session_state.get("real_farm_camera_original") else "Thermal"
    image_view=st.radio("Image view",["RGB + Thermal","Thermal","RGB faults","Original RGB"],horizontal=True,key=display_key,
        on_change=real_farm_camera_image_view_changed,args=(image,))
    paired=image_view=="RGB + Thermal"
    thermal_view=image_view in {"Thermal","RGB + Thermal"}
    st.caption("Paired RGB surface effects / marked fault locations and synthetic thermal response." if paired else
        "Synthetic relative thermal · not measured temperature." if thermal_view else
        "RGB surface effects and marked thermal/electrical fault locations · simulation only." if image_view=="RGB faults" else "Original RGB photograph.")
    preview=st.container()
    pending_result=st.session_state.get("real_farm_camera_pending_result_source")
    frame_width=state.get("camera_preparation_response",{}).get("physical_frame_width_fraction")
    render_revision="real-native-source-surfaces-v24-edges-v1-gap-v1-table-v2"+(f"-frame-width-{float(frame_width):.6f}" if frame_width is not None else "")
    key=(source,real_farm_geometry_signature(state.get("modules",[])),render_revision)
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
            key=(source,real_farm_geometry_signature(state.get("modules",[])),render_revision)
            st.session_state["real_farm_camera_layers"]=(key,layers)
            state["camera_capture"]={"source":source,"geometry":key[1],"native_size":list(image.size),
                "automatic_candidates":len(layers["camera_candidates"]),"fault_position_candidates":len(layers["accepted_modules"]),
                "training_verified":False}
            status.update(label="Thermal ready",state="complete",expanded=False)
        cached=(key,layers)
    if pending_result==source and (not cached or cached[0]!=key or "master_scalar" not in cached[1]):
        # Random placement already has resolved geometry. Prepare its thermal
        # image without running detection again or replacing live observations.
        with st.status("Preparing fault images…",expanded=False) as status:
            layers=real_farm_camera_build_layers(state,image)
            cached=(key,layers)
            st.session_state["real_farm_camera_layers"]=cached
            status.update(label="Fault images ready",state="complete",expanded=False)
    if not cached or cached[0][0]!=source:
        if image_view!="Thermal" or section=="Faults":
            candidates,credible=real_farm_camera_candidates(state,image)
            layers={"camera_candidates":candidates,"accepted_modules":credible,"mode":"rgb_preview_only"}
        else:
            if section=="Faults" and state.get("modules"):
                candidates,credible=real_farm_camera_candidates(state,image)
                provisional=real_farm_camera_simulation_targets(state,image,candidates,credible)
                selected=next(iter(credible+provisional),None)
                real_farm_camera_observation_source(state,image)
                st.caption("Saved geometry loaded. You can select a PV polygon now; Generate Thermal to view its hotspot.")
                if real_farm_camera_pick_target(state,image,selected):st.rerun()
                st.caption(f"{len(state.get('camera_observations',[]))} hotspot observations · Generate Thermal to view")
            with preview:
                st.image(image,caption=f"Original RGB · {image.width} × {image.height}",use_container_width=True)
            return
    else:
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
    view=st.radio("Camera acquisition",["Far View","Close View"],horizontal=True,key="real_farm_camera_distance")
    if section=="Faults" and view=="Far View":
        from pv_far_surface import real_controls
        real_controls(globals(),state,image,selected,layers)
    placement_key="real_farm_camera_placement"
    if st.session_state.get(placement_key)=="Random Hotspots":
        st.session_state[placement_key]="Random Faults"
    placement=st.radio("Fault placement",["Manual Select","Random Faults"],horizontal=True,key=placement_key) if section=="Faults" else "Manual Select"
    refresh=real_farm_camera_pick_target(state,image,selected) if section=="Faults" and placement=="Manual Select" else False
    if selected and selected.get("simulation_only"):
        st.caption("PROVISIONAL — exploratory simulation; excluded from verified training labels.")
    real_farm_camera_observation_source(state,image)
    observations=state["camera_observations"]
    if section=="Faults":
        if placement=="Random Faults":
            refresh=real_farm_camera_random_controls(state,image,credible) or refresh
            observations=state["camera_observations"]
        elif st.button("Add fault observation",disabled=not mid,use_container_width=True):
            real_farm_camera_new_observation(state,image,selected)
        if st.button("Clear Camera Faults",disabled=not observations):
            real_farm_camera_forget_observation_widgets(state,observations)
            state.pop("camera_random_batch",None)
            state["camera_observations"]=[];observations=[]
    target_lookup={m["module_id"]:m for m in targets}
    for observation in observations:
        if observation["module_id"] in target_lookup:
            observation.setdefault("target_geometry",json.loads(json.dumps(target_lookup[observation["module_id"]])))
        observation.pop("camera_visibility",None)
    active_observation=real_farm_camera_observation_controls(state,image,targets,view)
    if active_observation:mid=active_observation["module_id"]
    unavailable=[o for o in observations if o.get("module_id") not in target_lookup or
        (target_lookup[o["module_id"]].get("placement_kind")=="pv_point" and o.get("diagnostic_fault_type"))]
    if unavailable:st.warning("Some observations lack the geometry needed for their selected diagnosis. Their records are retained; use hotspot-only placement or review that target.")
    observations=[o for o in observations if o not in unavailable]
    secondary=st.columns(2)
    outlines=secondary[0].checkbox("Show Module Outlines",value=False,key="real_farm_camera_outlines")
    diagnostics=st.expander("Diagnostics",expanded=False)
    with diagnostics:
        grayscale=st.checkbox("Grayscale Signal",value=False,key="real_farm_camera_grayscale",disabled=not thermal_view)
        healthy_only=st.checkbox("Healthy Baseline",value=False,key="real_farm_camera_healthy")
    display=image_view if not thermal_view else "Grayscale Signal" if grayscale else "Synthetic Thermal"
    bounds=None
    rgb_report=None
    rgb_shown=None
    thermal_result_ready=False
    selected=next((m for m in targets if m["module_id"]==mid),None)
    if display in {"Original RGB","RGB faults"}:
        shown=image
        if display=="RGB faults" and not healthy_only:
            shown,rgb_report=annotated_rgb_fault_preview(globals(),image,targets,observations)
            with diagnostics:
                st.json(rgb_report)
        bounds=view_bounds(globals(),image,selected,candidates,view)
        if bounds:shown=shown.crop(bounds)
        rgb_label="simulated RGB surface appearance" if display=="RGB faults" and not healthy_only else "original RGB photograph"
        if rgb_report and rgb_report.get("annotated"):rgb_label+=" · marked thermal/electrical fault locations"
        caption=("Close View" if bounds else "Far View")+" · "+rgb_label
    elif display=="Automatic Modules":
        if state.get("camera_structure_check",{}).get("scene_geometry_unreliable"):
            shown=image;caption="Unreliable cell-grid proposals withheld; no physical module labels available"
        else:
            shown=real_farm_annotated_preview(image,candidates,mid,max_width=image.width,show_labels=True)
            caption=("Saved module geometry" if state.get("camera_geometry_origin") else "Automatic module proposals")+" · unverified · amber requires review"
    elif "master_scalar" not in layers:
        shown=image;caption="Original RGB · thermal frame not generated yet"
        st.info("Generate a visible Far condition above to prepare both views, or use Generate Thermal for the existing thermal workflow.")
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
            thermal_result_ready=True
        except FaultGeometryError as exc:
            st.error(f"Synthetic fault could not be rendered accurately: {exc}")
            shown=real_farm_camera_acquisition(layers,[],"close" if bounds else "far",bounds,grayscale=display=="Grayscale Signal")
        caption=(f"Close View · native-source thermal detail · {shown.width} × {shown.height}" if bounds else f"Far View · {image.width} × {image.height}")
        caption += " · Fault preview unavailable; healthy baseline shown" if not thermal_result_ready else " · Healthy" if healthy_only else " · With synthetic fault" if observations else " · Healthy"
    if paired:
        rgb_shown=image
        if not healthy_only:
            rgb_shown,rgb_report=annotated_rgb_fault_preview(globals(),image,targets,observations)
            with diagnostics:st.json(rgb_report)
        if bounds:rgb_shown=rgb_shown.crop(bounds)
    if outlines:
        outlined=[]
        for frame in ([shown,rgb_shown] if paired else [shown]):
            overlay=np.array(frame);offset=np.float32(bounds[:2] if bounds else (0,0))
            for module in candidates+provisional:
                polygon=np.rint(np.float32(module["outer_polygon_rgb"])-offset).astype(np.int32)
                cv2.polylines(overlay,[polygon],True,(0,255,255) if module["module_id"]==mid else (255,180,30),1,cv2.LINE_AA)
            outlined.append(Image.fromarray(overlay))
        shown=outlined[0]
        if paired:rgb_shown=outlined[1]
    with preview:
        if paired:
            rgb_column,thermal_column=st.columns(2)
            with rgb_column:
                st.markdown("**RGB faults**")
                st.image(rgb_shown,caption=("Close View" if bounds else "Far View")+" · RGB surface effects and marked fault locations" if not healthy_only else "Original RGB · healthy baseline",width="stretch",output_format="PNG")
                st.download_button("Download RGB faults",export_current_view_bytes(rgb_shown),file_name=f"camera_RGB_faults_{view.replace(' ','_')}.png",mime="image/png")
            with thermal_column:
                st.markdown("**Thermal faults**")
                st.image(shown,caption=caption,width="stretch",output_format="PNG")
                st.download_button("Download thermal faults",export_current_view_bytes(shown),file_name=f"camera_Thermal_faults_{view.replace(' ','_')}.png",mime="image/png")
        else:
            st.image(shown,caption=caption,width="stretch",output_format="PNG")
            st.download_button("Download camera view",export_current_view_bytes(shown),file_name=f"camera_{display.replace(' ','_')}_{view.replace(' ','_')}.png",mime="image/png")
        if rgb_report is not None:
            st.caption(f"{len(rgb_report['rendered'])} RGB surface effect(s) · {len(rgb_report.get('annotated',[]))} marked thermal/electrical location(s)")
            if rgb_report.get("annotated"):
                st.caption("Location markers are annotations, not visible physical damage. Dust, obstruction and cracks use simulated RGB surface effects.")
            if rgb_report["unavailable"]:
                st.info(f"{len(rgb_report['unavailable'])} observation(s) could not be shown in RGB; see details below.")
                with st.expander("RGB preview details",expanded=False):
                    for skipped in rgb_report["unavailable"]:
                        st.write(f"{skipped['observation_id']}: {skipped['reason']}")
        if not paired and observations:
            st.button("Show RGB + thermal",on_click=real_farm_camera_show_thermal_result,args=(image,True))
    with diagnostics:
        st.json(layers.get("scalar_audit",{}))
        if "master_scalar" in layers:
            if layers.get("acquisition_audit"):
                st.download_button("Download scalar comparison",json.dumps(layers["acquisition_audit"],indent=2),file_name="camera_scalar_comparison.json",mime="application/json")
            st.download_button("Download exact scalar PV mask",export_current_view_bytes(Image.fromarray(layers["camera_composited_pv_mask"])),file_name="scalar_PV_mask_UNVERIFIED.png",mime="image/png")
            st.download_button("Download module outlines",export_current_view_bytes(real_farm_annotated_preview(image,candidates,max_width=image.width,show_labels=False)),file_name="module_proposals_UNVERIFIED.png",mime="image/png")
        st.download_button("Download original RGB",export_current_view_bytes(image),file_name="original_RGB.png",mime="image/png")
        if "master_scalar" in layers:
            st.download_button("Download pre-palette proxy codes",export_current_view_bytes(Image.fromarray(np.uint8(np.clip(layers["master_scalar"],0,255)))),file_name="synthetic_proxy_codes_NOT_TEMPERATURE.png",mime="image/png")
        st.caption(f"{len(candidates)} module proposals · {len(targets)} simulation targets. Synthetic relative signal, not measured temperature. Training-label export requires separate explicit verification.")
        st.download_button("Download unverified analysis",real_farm_project_json(state),file_name="camera_analysis_UNVERIFIED.json",mime="application/json")
    if pending_result==source and thermal_result_ready:
        st.session_state.pop("real_farm_camera_pending_result_source",None)
    if refresh:st.rerun()
