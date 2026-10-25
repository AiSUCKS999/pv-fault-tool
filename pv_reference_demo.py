"""Healthy-only Streamlit adapter for the existing reference-driven POC.

No fault engine, detector, geometry recovery, or production scalar changes.
Only two small camera images are retained per browser session.
"""

REFERENCE_DEMO_OPTIONS = ["Current Thermal", "Reference-Driven Thermal (Experimental)", "Legacy Thermal (Fallback)"]


def reference_demo_selected():
    return st.session_state.get("thermal_renderer_mode", "Current Thermal") == REFERENCE_DEMO_OPTIONS[1]


def reference_demo_selector():
    st.markdown("""<style>
        [data-testid="stRadio"] label p,
        [data-testid="stCheckbox"] label p,
        [data-testid="stCaptionContainer"] p { color: #cbd5e1 !important; }
        </style>""", unsafe_allow_html=True)
    if "thermal_renderer_mode" not in st.session_state:
        st.session_state["thermal_renderer_mode"] = REFERENCE_DEMO_OPTIONS[
            1 if qp_get("thermal_renderer", "current") == "reference" else 2 if qp_get("thermal_renderer", "current") == "legacy" else 0
        ]
    st.selectbox("Thermal renderer", REFERENCE_DEMO_OPTIONS, key="thermal_renderer_mode",
                 on_change=reference_demo_mode_changed)
    if reference_demo_selected():
        st.caption("Experimental Reference-Driven Thermal · Healthy scenes only")


def reference_demo_mode_changed():
    st.query_params["thermal_renderer"] = "reference" if reference_demo_selected() else "legacy" if st.session_state.get("thermal_renderer_mode") == REFERENCE_DEMO_OPTIONS[2] else "current"


def reference_demo_notice():
    st.info("Experimental Reference-Driven Thermal — healthy rendering only. Fault injection, diagnosis and "
            "fault datasets are unsupported in this mode. Switch to Current Thermal for fault demonstrations. "
            "Existing faults are retained and hidden here.")


def reference_demo_master(image, modules, signature, active_support=None):
    import hashlib
    from pv_thermal_reference_poc import render_healthy

    profile_path = ROOT / "config" / "reference_thermal_profile.json"
    profile_bytes = profile_path.read_bytes()
    key = (str(signature), image.size, hashlib.sha256(profile_bytes).hexdigest(),
           (ROOT / "pv_thermal_reference_poc.py").stat().st_mtime_ns)
    cache = st.session_state.get("reference_demo_cache", {})
    if key in cache:
        return cache[key]
    if not modules:
        raise ValueError("No accepted physical module geometry is available for this scene.")
    if len(modules) > 1200:
        raise ValueError("The experimental meeting preview supports at most 1,200 modules. Use a smaller scene.")
    # Preserve the source aspect/framing. Camera signal is bounded independently
    # of the upload size, and the same cached acquisition supplies close crops.
    ratio = min(1., 960. / max(image.size))
    size = (max(8, round(image.width * ratio)), max(8, round(image.height * ratio)))
    with st.spinner("Rendering experimental healthy thermal…"):
        signal, details = render_healthy(image, modules, json.loads(profile_bytes), size,
                                         active_support=active_support)
        lut = np.asarray(thermal_lut(), np.uint8)
        result = Image.fromarray(lut[np.clip(np.rint(signal), 0, 255).astype(np.uint8)])
        del signal, details
    cache = dict(list(cache.items())[-1:])
    cache[key] = result
    st.session_state["reference_demo_cache"] = cache
    st.session_state["reference_demo_render_count"] = st.session_state.get("reference_demo_render_count", 0) + 1
    return result


def reference_demo_generated(location_id, render_size=None):
    image = solar_render_image(location_id, None)
    if image is None:
        st.error("Experimental rendering requires an existing generated RGB scene. Choose a rendered scene or Current Thermal.")
        st.stop()
    mask = solar_render_panel_mask(image, location_id).astype(np.float32) / 255.
    mask = (np.clip(mask * 1.18, 0, .98) * 255).astype(np.uint8)
    arrays = solar_render_metadata_array_geometries(mask, load_solar_render_metadata(location_id))
    if not arrays:
        st.error("This scene has no usable array geometry for experimental rendering. Switch to Current Thermal.")
        st.stop()
    modules = []
    for key, geometry in arrays.items():
        region = {"box": geometry["box"], "polygon": geometry["polygon"]}
        for number, module in array_module_polygons(region).items():
            modules.append({"module_id": f"{key}-{number}", "outer_polygon_rgb": module_outer_polygon(module),
                            "active_polygon_rgb": module_active_polygon(module)})
    signature = ("generated", location_id, solar_render_cache_signature(location_id),
                 custom_row_count(location_id), custom_panels_per_row(location_id))
    try:
        master = reference_demo_master(image, modules, signature)
    except (ValueError, OSError) as exc:
        st.error(f"Experimental thermal unavailable: {exc}")
        st.stop()
    if render_size:
        ratio = min(render_size[0] / image.width, render_size[1] / image.height)
        size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    else:
        size = image.size
    return master.resize(size, Image.Resampling.BILINEAR)


def reference_demo_real_panel(state, image, modules, section):
    import hashlib

    st.markdown("#### Experimental Reference-Driven Thermal")
    reference_demo_notice()
    accepted = st.session_state.get("real_farm_accepted_modules", [])
    if not accepted:
        st.info("Use Farm Setup to detect and accept modules first. Rendering does not rerun detection.")
        return
    mask = st.session_state.get("real_farm_final_pv_mask")
    signature = ("real_farm", state.get("source_rgb_sha256_prefix"),
                 real_farm_geometry_signature(accepted), hashlib.sha256(memoryview(mask)).hexdigest())
    try:
        master = reference_demo_master(image, accepted, signature, active_support=mask)
    except (ValueError, OSError) as exc:
        st.error(f"Experimental thermal unavailable: {exc}")
        return
    selected_id = st.selectbox("Selected real module", [m["module_id"] for m in accepted],
                               key="reference_demo_real_module")
    selected = next(m for m in accepted if m["module_id"] == selected_id)
    distance = st.radio("Experimental view", ["Far View", "Close View"], horizontal=True,
                        key="reference_demo_real_distance")
    display = master
    if distance == "Close View":
        sx, sy = master.width / image.width, master.height / image.height
        polygon = np.asarray(selected["outer_polygon_rgb"], np.float32) * [sx, sy]
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        margin = np.maximum((hi - lo) * .65, 12)
        bounds = (max(0, int(lo[0] - margin[0])), max(0, int(lo[1] - margin[1])),
                  min(master.width, int(hi[0] + margin[0]) + 1), min(master.height, int(hi[1] + margin[1]) + 1))
        display = master.crop(bounds)
        st.image(display, caption=f"Healthy Close View · {selected_id} · same thermal acquisition", width=700)
    else:
        st.image(display, caption=f"Healthy experimental preview · {len(accepted)} accepted modules", use_container_width=True)
    st.caption("Relative thermal appearance; not measured temperature. The existing masks and source framing are retained. "
               "This single field-reference profile is experimental across other materials.")
    st.download_button("Download Experimental Healthy Thermal", export_current_view_bytes(master),
                       file_name="real_farm_reference_healthy.png", mime="image/png")
    if section == "Export":
        st.caption("Experimental export is a healthy preview PNG only. Use Current Thermal for fault datasets.")


def reference_demo_load_local_image():
    path = ROOT / "artifacts" / "thermal_realism_v13_background" / "dense_source_preview.png"
    image = Image.open(path).convert("RGB")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    st.session_state["real_farm_image"] = image
    st.session_state["real_farm_image_path"] = str(path)
    st.session_state["real_farm_state"] = real_farm_default_state(path, image, path.name, digest)
    st.session_state["real_farm_selected_module"] = None
    for name in ("real_farm_modules", "real_farm_accepted_modules", "real_farm_final_pv_mask",
                 "real_farm_thermal_master_layers", "real_farm_thermal_master_signature",
                 "reference_demo_real_module", "real_farm_target_module", "real_farm_target_module_select",
                 "real_farm_uploaded_name", "real_farm_close_view", "real_farm_selected_observation"):
        st.session_state.pop(name, None)
    st.session_state["real_farm_undo_stack"] = []
    st.session_state["real_farm_redo_stack"] = []
