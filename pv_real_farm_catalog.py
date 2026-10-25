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


def real_farm_camera_generate_random(state,image,credible,count,seed):
    """Reuse the existing distinct-module randomizer and validate visible support."""
    real_farm_camera_observation_source(state,image)
    pool=real_farm_camera_random_modules(state,image,credible)
    if not pool:raise ValueError("No resolved modules for random placement. Use Manual Select to confirm a provisional PV target.")
    proposals=real_farm_generate_observations(pool,min(int(count),len(pool)),int(seed))
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
    if not validated:raise ValueError("No visible PV support for the requested random targets. Use Manual Select.")
    real_farm_camera_remove_random(state)
    batch=hashlib.sha256(f"{real_farm_source_identity(image)}:{int(seed)}:{int(count)}".encode()).hexdigest()[:16]
    for target,proposal in validated:
        real_farm_camera_new_observation(state,image,target,severity=proposal["severity"])
        state["camera_observations"][-1].update(seed=int(seed),placement_origin="random",random_batch_id=batch)
    state["camera_random_batch"]={"id":batch,"seed":int(seed),"count":len(validated)}
    return len(validated)


def real_farm_camera_random_controls(state,image,credible):
    pool=real_farm_camera_random_modules(state,image,credible)
    if not pool:
        st.info("Random hotspots need resolved module geometry. Use Manual Select to confirm a provisional PV polygon.")
    count_key="real_farm_camera_random_count"
    if count_key in st.session_state and not 1<=st.session_state[count_key]<=max(1,len(pool)):
        st.session_state[count_key]=max(1,min(int(st.session_state[count_key]),max(1,len(pool))))
    count=st.number_input("Number of hotspots",min_value=1,max_value=max(1,len(pool)),value=min(3,max(1,len(pool))),step=1,key=count_key)
    seed=st.number_input("Random seed",min_value=0,max_value=2147483647,value=42,step=1,key="real_farm_camera_random_seed")
    changed=False
    if st.button("Generate Random Hotspots",disabled=not pool,use_container_width=True):
        try:
            real_farm_camera_generate_random(state,image,credible,count,seed)
            st.session_state["real_farm_camera_distance"]="Far View"
            changed=True
        except ValueError as exc:st.warning(str(exc))
    if st.button("Remove Random Hotspots",disabled=not any(o.get("placement_origin")=="random" for o in state.get("camera_observations",[])),use_container_width=True):
        real_farm_camera_remove_random(state);changed=True
    st.caption("Regenerate replaces only the random batch. Manually placed observations are preserved.")
    return changed
