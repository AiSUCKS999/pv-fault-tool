"""Direct thermal-display capture. Never recolor or infer temperature from it."""


def real_thermal_decode(data):
    original=ImageOps.exif_transpose(Image.open(BytesIO(data)))
    scalar=None;window=None
    if original.mode in {"I","F","I;16","I;16B","I;16L"}:
        raw=np.asarray(original,np.float32)
        if not np.isfinite(raw).all():raise ValueError("Thermal display contains non-finite sensor codes")
        low,high=float(raw.min()),float(raw.max());window=[low,high]
        # A display window is not a temperature calibration. Raw upload retained.
        gray=np.uint8(np.clip((raw-low)/max(high-low,1.)*255.,0,255))
        display=Image.fromarray(gray).convert("RGB");scalar=gray.astype(np.float32)
    else:
        display=original.convert("RGB")
        rgb=np.asarray(display);span=rgb.max(2).astype(np.int16)-rgb.min(2)
        if float(np.mean(span<=3))>.999:scalar=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    return display,scalar,window


def real_thermal_geometry(project,image):
    source=real_farm_source_identity(image)
    if project.get("source_rgb_pixel_sha256")!=source:raise ValueError("Geometry does not match this thermal capture's pixels")
    if project.get("image")!={"width":image.width,"height":image.height}:raise ValueError("Geometry uses different image dimensions")
    candidates=real_farm_recovered_modules(project.get("modules",[]),include_unaccepted=True,image_size=image.size)
    approved=[]
    for module in candidates:
        review=module.get("physical_review",{})
        if review.get("source")!=source or review.get("polygon")!=real_farm_polygon_signature(module):continue
        if module.get("physical_module_unresolved") or module.get("refinement_status","accepted")!="accepted":continue
        if not real_farm_mechanical_validation(module.get("outer_polygon_rgb",[]),image.size)["mechanically_valid"]:continue
        approved.append(module)
    return approved


def real_farm_real_thermal_workflow():
    st.caption("Real thermal display image \u00b7 non-radiometric \u00b7 no absolute temperatures")
    upload=st.file_uploader("Upload a complete thermal image",type=["png","jpg","jpeg","tif","tiff"],key="real_thermal_upload")
    if upload is None:return
    data=upload.getvalue();digest=hashlib.sha256(data).hexdigest()
    capture=st.session_state.get("real_thermal_capture")
    if not capture or capture["digest"]!=digest:
        try:
            display,scalar,window=real_thermal_decode(data)
        except (OSError,ValueError) as exc:
            st.error(str(exc));return
        capture={"digest":digest,"image":display,"scalar":scalar,"window":window,
                 "modules":[],"observations":[],"geometry_digest":None,"name":upload.name}
        st.session_state["real_thermal_capture"]=capture
    image=capture["image"];scalar=capture["scalar"]
    with st.expander("Geometry / source details"):
        st.caption("Original pixels are retained. Existing anomalies remain in the baseline. JPEG/PNG display codes are not temperatures.")
        geometry=st.file_uploader("Matching thermal geometry project (optional)",type=["json"],key="real_thermal_geometry_upload")
        if geometry is not None:
            geometry_digest=hashlib.sha256(geometry.getvalue()).hexdigest()
            if geometry_digest!=capture["geometry_digest"]:
                try:
                    project=real_farm_load_project(geometry)
                    capture["modules"]=real_thermal_geometry(project,image)
                    capture["observations"]=[];capture["geometry_digest"]=geometry_digest
                except (ValueError,KeyError,TypeError) as exc:
                    st.error(str(exc))
        st.caption(f"{len(capture['modules'])} source-matched inspected module positions. Preview needs no geometry.")
        if capture["window"] is not None:st.write({"uncalibrated_sensor_code_display_window":capture["window"]})
        st.download_button("Download original capture",data,file_name=upload.name,key="real_thermal_original_download")
    modules=capture["modules"];observations=capture["observations"]
    selected=None
    if modules:
        target=st.selectbox("Thermal module",[m["module_id"] for m in modules],key="real_thermal_target")
        selected=next(m for m in modules if m["module_id"]==target)
        fault=st.selectbox("Synthetic diagnosis",["SingleHotSpot",*CLOSE_DIAGNOSTIC_FAULT_TYPES],format_func=real_farm_diagnostic_label,key="real_thermal_fault")
        severity=st.slider("Severity",1,10,7,key="real_thermal_severity")
        if st.button("Add synthetic fault",disabled=scalar is None):
            observation=real_farm_add_module_observation(modules,observations,target,severity,42)
            observation["diagnostic_fault_type"]=fault
            capture["observations"]=[o for o in observations if o["module_id"]!=target]+[observation]
            observations=capture["observations"]
            st.session_state["real_thermal_compare"]="With synthetic faults"
    else:
        st.caption("Fault injection unavailable: no inspected geometry registered to this thermal capture. Original preview remains available.")
    if modules and scalar is None:st.caption("Colorized pixels are preserved. Fault injection needs a known scalar display mapping.")
    distance=st.radio("Thermal view",["Far View","Close View"],horizontal=True,key="real_thermal_distance")
    bounds=None
    if distance=="Close View":
        if selected:bounds=real_farm_close_context_bounds(selected,modules,(image.height,image.width),.25)
        else:
            w,h=image.size;bounds=(w//4,h//4,max(w//4+1,w*3//4),max(h//4+1,h*3//4))
    displayed=image
    if observations and scalar is not None:
        choice=st.radio("Compare",["Original capture","With synthetic faults","Side by side"],horizontal=True,key="real_thermal_compare")
        layers={"master_scalar":scalar,"accepted_modules":modules}
        try:
            modified=real_farm_faulted_master(layers,observations)
            delta=modified-scalar;changed=np.abs(delta)>1e-5
            pixels=np.array(image);gray=np.uint8(np.clip(np.rint(modified),0,255))
            pixels[changed]=gray[changed,None]
            altered=Image.fromarray(pixels)
            if choice=="Side by side":
                columns=st.columns(2)
                columns[0].image(image.crop(bounds) if bounds else image,caption="Original captured baseline",use_container_width=True)
                columns[1].image(altered.crop(bounds) if bounds else altered,caption="Synthetic diagnosis added; non-radiometric",use_container_width=True)
                displayed=altered
            elif choice=="With synthetic faults":displayed=altered
        except FaultGeometryError as exc:st.error(str(exc))
    shown=displayed.crop(bounds) if bounds else displayed
    if not observations or st.session_state.get("real_thermal_compare")!="Side by side":
        st.image(shown,caption=f"Original capture / relative overlays \u00b7 {shown.width} \u00d7 {shown.height} pixels",use_container_width=True)
    st.download_button("Download displayed thermal PNG",export_current_view_bytes(shown),file_name="real_thermal_display.png",mime="image/png",key="real_thermal_png_download")
    with st.expander("Supplied thermal reference"):
        st.image(str(ROOT/"assets/thermal_references/real_thermal_target.png"),caption="Supervisor reference; cyan markup is part of the supplied image",use_container_width=True)
