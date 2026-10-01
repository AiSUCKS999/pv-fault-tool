# Real patch-bank loading, patch selection, and crop status helpers.
def texture_group_for_location(location_id):
    if location_id == "rooftop":
        return "rooftop"
    if location_id == "farm_lake":
        return "floating"
    if location_id in {"agri_rows", "scenario_1"}:
        return "ground"
    if location_id == "desert_farm":
        return "field"
    return "ground"


@st.cache_data(show_spinner=False)
def real_panel_textures(group):
    textures = []
    group = group or "ground"
    paths = []
    paths.extend(sorted(ASSET_DIR.glob(f"pv_real_texture_{group}_*.png")))
    if group != "ground":
        paths.extend(sorted(ASSET_DIR.glob("pv_real_texture_ground_*.png")))
    if PANEL_TEXTURE_DIR.exists():
        paths.extend(sorted(PANEL_TEXTURE_DIR.glob("*.png"))[:140])
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            patch = Image.open(path).convert("RGB")
        except OSError:
            continue
        arr = np.array(patch)
        if arr.shape[0] >= 12 and arr.shape[1] >= 20:
            textures.append(arr)
    return textures


@st.cache_data(show_spinner=False)
def actual_panel_model_textures():
    if not ACTUAL_PANEL_MODEL_TEXTURE.exists():
        return []
    try:
        patch = Image.open(ACTUAL_PANEL_MODEL_TEXTURE).convert("RGB")
    except OSError:
        return []
    return [np.array(patch)]


def actual_panel_model_enabled():
    return ACTUAL_PANEL_MODEL_TEXTURE.exists()


@st.cache_data(show_spinner=False)
def patch_bank_manifest():
    path = PATCH_BANK_DIR / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def fault_patch_counts():
    counts = {}
    if FAULT_PATCH_DIR.exists():
        for path in sorted(FAULT_PATCH_DIR.iterdir()):
            if path.is_dir():
                counts[path.name] = len(list(path.glob("*.png")))
    for key, value in patch_bank_manifest().get("fault_counts", {}).items():
        counts.setdefault(key, int(value))
    return counts


@st.cache_data(show_spinner=False)
def fault_patch_paths(fault_type):
    classes = [canonical_fault_type(fault_type)] + PATCH_FALLBACKS.get(canonical_fault_type(fault_type), [])
    paths = []
    for class_name in classes:
        class_dir = FAULT_PATCH_DIR / class_name
        if class_dir.exists():
            paths.extend(str(path) for path in sorted(class_dir.glob("*.png")))
        if paths:
            break
    return paths


@st.cache_data(show_spinner=False)
def load_fault_patch(path):
    try:
        image = Image.open(path)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            return np.array(image.convert("RGBA"))
        return np.array(image.convert("RGB"))
    except OSError:
        return None


def selected_fault_patch(fault_type, seed):
    paths = fault_patch_paths(fault_type)
    if not paths:
        return None
    fault_type = canonical_fault_type(fault_type)
    if fault_type in CRACKING_FAULTS | HOTSPOT_FAULTS:
        alpha_paths = []
        for path in paths:
            try:
                image = Image.open(path)
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    alpha_paths.append(path)
            except OSError:
                continue
        if alpha_paths:
            paths = alpha_paths
    if fault_type in CRACKING_FAULTS:
        glass_breakage_paths = [
            path for path in paths
            if "pv_v1_glass_breakage" in Path(path).stem
        ]
        if glass_breakage_paths:
            paths = glass_breakage_paths
        else:
            preferred_paths = [
                path for path in paths
                if Path(path).stem.rsplit("_", 1)[-1] in PREFERRED_CELL_CRACKING_PATCH_SUFFIXES
            ]
            if preferred_paths:
                paths = preferred_paths
    return load_fault_patch(paths[seed % len(paths)])


def patch_bank_status_rows():
    counts = fault_patch_counts()
    rows = []
    for fault_type in USER_SELECTABLE_FAULT_TYPES:
        direct = counts.get(fault_type, 0)
        fallback = next((name for name in PATCH_FALLBACKS.get(fault_type, []) if counts.get(name, 0)), "")
        rows.append(
            {
                "fault_type": fault_display_name(fault_type),
                "direct_patches": direct,
                "fallback_class": fault_display_name(fallback) if fallback else "",
                "status": "direct" if direct else ("fallback" if fallback else "procedural"),
            }
        )
    return rows
