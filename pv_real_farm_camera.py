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
    key=(id(layers["master_scalar"]),config,distance,bounds,grayscale)
    cache=st.session_state.get("real_farm_camera_acquisitions",{})
    if key in cache:return cache[key]
    field=real_farm_faulted_master(layers,observations) if observations else layers["master_scalar"]
    sensor=real_farm_camera_sensor(field,distance,layers["pv_module_union_mask"])
    # Inspect the same scalar planes actually sent to display, before palette.
    if observations:
        baseline=real_farm_camera_sensor(layers["master_scalar"],distance,layers["pv_module_union_mask"])
        crop=bounds or (0,0,field.shape[1],field.shape[0])
        x1,y1,x2,y2=crop
        pre=(field[y1:y2,x1:x2]-layers["master_scalar"][y1:y2,x1:x2])
        post=(sensor[y1:y2,x1:x2]-baseline[y1:y2,x1:x2])
        layers["acquisition_audit"]={"distance":distance,"bounds":list(crop),"display_range":layers["display_range"],
            "pre_sensor_peak":float(pre.max()),"post_sensor_peak":float(post.max()),
            "pre_sensor_energy":float(np.sum(pre,dtype=np.float64)),"post_sensor_energy":float(np.sum(post,dtype=np.float64)),
            "pre_sensor_half_peak_pixels":int(np.count_nonzero(pre>pre.max()*.5)),
            "post_sensor_half_peak_pixels":int(np.count_nonzero(post>post.max()*.5)),
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
    key=(id(layers["master_scalar"]),config,distance,bounds,grayscale)
    cache={k:v for k,v in cache.items() if k[0]==key[0] and k[1]==key[1] and k[2]!=distance}
    cache[key]=acquired;st.session_state["real_farm_camera_acquisitions"]=cache
    return acquired


def real_farm_camera_workflow(state,image):
    # Camera capture uses the recovered Current renderer. A hidden Generated
    # Mode selector must not silently select a legacy sensor or module field.
    st.session_state["thermal_renderer_mode"]="Current Thermal"
    st.caption("Synthetic relative thermal proxy \u00b7 experimental appearance \u00b7 not measured temperature.")
    source=real_farm_source_identity(image)
    key=(source,real_farm_geometry_signature(state.get("modules",[])),"real-background-sunlit-v13")
    cached=st.session_state.get("real_farm_camera_layers")
    generate=st.button("Generate Thermal",type="primary",use_container_width=True)
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
            key=(source,real_farm_geometry_signature(state.get("modules",[])),"real-background-sunlit-v13")
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
    candidates=layers["camera_candidates"];credible=layers["accepted_modules"]
    state.setdefault("camera_capture",{}).update(automatic_candidates=len(candidates),fault_position_candidates=len(credible),geometry_unreliable=bool(state.get("camera_structure_check",{}).get("scene_geometry_unreliable")),training_verified=False,mode=layers.get("mode"),scalar_audit=layers.get("scalar_audit",{}))
    if state.get("camera_geometry_origin",{}).get("type")=="saved_source_matched_project":st.caption("Using saved geometry matched to this RGB image. Labels remain unverified.")
    if len(credible)<len(candidates):st.caption("Some module boundaries are uncertain. Fault placement is limited to supported targets.")
    if state.get("camera_recovery_message") and not credible and not state.get("camera_structure_check",{}).get("scene_geometry_unreliable"):st.warning(state["camera_recovery_message"])
    target_ids=[m["module_id"] for m in credible]
    if target_ids:
        if st.session_state.get("real_farm_camera_target") not in target_ids:
            center=np.float32(image.size)/2
            selected=min(credible,key=lambda m:np.linalg.norm(np.float32(m["outer_polygon_rgb"]).mean(0)-center))
            st.session_state["real_farm_camera_target"]=selected["module_id"]
        mid=st.selectbox("Fault target module (automatic confidence)",target_ids,key="real_farm_camera_target")
    else:
        mid=None;st.warning("Automatic module geometry is unreliable for this image. Thermal capture only; fault placement is unavailable.")
    controls=st.columns(3)
    fault=controls[0].selectbox("Synthetic fault",["SingleHotSpot",*CLOSE_DIAGNOSTIC_FAULT_TYPES],format_func=real_farm_diagnostic_label,key="real_farm_camera_fault")
    severity=controls[1].slider("Fault severity",1,10,7,key="real_farm_camera_severity")
    observations=state.setdefault("camera_observations",[])
    if controls[2].button("Apply Synthetic Fault",disabled=not mid):
        observation=real_farm_add_module_observation(credible,observations,mid,severity,42)
        observation["diagnostic_fault_type"]=fault
        state["camera_observations"]=[o for o in observations if o["module_id"]!=mid]+[observation]
        observations=state["camera_observations"]
    if st.button("Clear Camera Faults",disabled=not observations):
        state["camera_observations"]=[];observations=[]
    observations=[o for o in observations if o.get("module_id") in target_ids]
    for observation in observations:observation.pop("camera_visibility",None)
    view=st.radio("Camera acquisition",["Far View","Close View"],horizontal=True,key="real_farm_camera_distance")
    display=st.radio("Camera display",["Synthetic Thermal","Grayscale Signal","Original RGB","Automatic Modules"],horizontal=True,key="real_farm_camera_display")
    outlines=st.checkbox("Show module outlines",value=False,key="real_farm_camera_outlines",disabled=bool(state.get("camera_structure_check",{}).get("scene_geometry_unreliable"))) if display=="Synthetic Thermal" else False
    healthy_only=st.checkbox("Show healthy baseline",value=False,key="real_farm_camera_healthy") if display in {"Synthetic Thermal","Grayscale Signal"} else False
    selected=next((m for m in credible if m["module_id"]==mid),None)
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
                if selected:bounds=real_farm_close_context_bounds(selected,candidates,(image.height,image.width),.25)
                else:
                    w,h=image.size;bounds=(max(0,w//2-500),max(0,h//2-350),min(w,w//2+500),min(h,h//2+350))
            shown=real_farm_camera_acquisition(layers,[] if healthy_only else observations,"close" if bounds else "far",bounds,grayscale=display=="Grayscale Signal")
        except FaultGeometryError as exc:
            st.error(f"Synthetic fault could not be rendered accurately: {exc}")
            shown=real_farm_camera_acquisition(layers,[],"close" if bounds else "far",bounds,grayscale=display=="Grayscale Signal")
        caption=(f"Close View · source-coordinate crop · {shown.width} × {shown.height} · no super-resolution" if bounds else f"Far View · {image.width} × {image.height}")
        caption += " · Healthy" if healthy_only else " · With synthetic fault" if observations else " · Healthy"
    if outlines and not state.get("camera_structure_check",{}).get("scene_geometry_unreliable"):
        overlay=np.array(shown);offset=np.float32(bounds[:2] if bounds else (0,0))
        for module in candidates:
            polygon=np.rint(np.float32(module["outer_polygon_rgb"])-offset).astype(np.int32)
            cv2.polylines(overlay,[polygon],True,(0,255,255) if module["module_id"]==mid else (255,180,30),1,cv2.LINE_AA)
        shown=Image.fromarray(overlay)
    st.image(shown,caption=caption,use_container_width=True)
    st.download_button("Download camera view",export_current_view_bytes(shown),file_name=f"camera_{display.replace(' ','_')}_{view.replace(' ','_')}.png",mime="image/png")
    with st.expander("Advanced"):
        st.json(layers.get("scalar_audit",{}))
        if layers.get("acquisition_audit"):
            st.download_button("Download scalar comparison",json.dumps(layers["acquisition_audit"],indent=2),file_name="camera_scalar_comparison.json",mime="application/json")
        st.download_button("Download pre-palette proxy codes",export_current_view_bytes(Image.fromarray(np.uint8(np.clip(layers["master_scalar"],0,255)))),file_name="synthetic_proxy_codes_NOT_TEMPERATURE.png",mime="image/png")
        st.caption(f"{len(candidates)} module proposals · {len(credible)} eligible fault targets. Synthetic relative signal, not measured temperature. Training-label export requires separate explicit verification.")
        st.download_button("Download unverified analysis",real_farm_project_json(state),file_name="camera_analysis_UNVERIFIED.json",mime="application/json")
        st.button("Advanced Geometry Review",on_click=real_farm_open_geometry)
