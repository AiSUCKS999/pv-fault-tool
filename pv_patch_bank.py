# Real patch-bank loading, patch selection, and crop status helpers.
def texture_group_for_location(location_id):
    location_id = normalize_location_id(location_id)
    if location_id == "rooftop_warehouse":
        return "rooftop"
    if location_id == "floating_water":
        return "floating"
    if location_id in {"agri_field_new", "grass_open"}:
        return "ground"
    if location_id == "desert_track":
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


def fault_patch_paths(fault_type):
    # Enumerating names is cheap and observes added/removed assets. Image
    # qualification and decoding below are cached by file revision instead.
    classes = [canonical_fault_type(fault_type)] + PATCH_FALLBACKS.get(canonical_fault_type(fault_type), [])
    paths = []
    for class_name in classes:
        class_dir = FAULT_PATCH_DIR / class_name
        if class_dir.exists():
            paths.extend(str(path) for path in sorted(class_dir.glob("*.png")))
        if paths:
            break
    return paths


def fault_patch_revision(path):
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None, None


@st.cache_data(show_spinner=False)
def load_fault_patch(path, file_revision=None):
    # file_revision participates in the cache key even though decoding only
    # needs the path. Replacing an asset must not retain its old pixel array.
    try:
        with Image.open(path) as image:
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                return np.array(image.convert("RGBA"))
            return np.array(image.convert("RGB"))
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def alpha_fault_patch_paths(path_revisions):
    """Qualify transparency once per ordered path set and asset revision."""
    paths = []
    for path, _modified_ns, _size in path_revisions:
        try:
            with Image.open(path) as image:
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    paths.append(path)
        except OSError:
            continue
    return paths


def selected_fault_patch(fault_type, seed):
    paths = fault_patch_paths(fault_type)
    if not paths:
        return None
    fault_type = canonical_fault_type(fault_type)
    if fault_type in CRACKING_FAULTS | HOTSPOT_FAULTS:
        groups = [paths]
        if fault_type in CRACKING_FAULTS:
            groups = [[], [], []]
            for path in paths:
                stem = Path(path).stem
                if "pv_v1_glass_breakage" in stem:
                    groups[0].append(path)
                elif stem.rsplit("_", 1)[-1] in PREFERRED_CELL_CRACKING_PATCH_SUFFIXES:
                    groups[1].append(path)
                else:
                    groups[2].append(path)
        # Alpha qualification precedes crack priority. Qualifying priority
        # groups lazily yields the same candidates without opening lower groups
        # once a higher group contains usable alpha references.
        for group in groups:
            if not group:
                continue
            revisions = tuple((path, *fault_patch_revision(path)) for path in group)
            alpha_paths = alpha_fault_patch_paths(revisions)
            if alpha_paths:
                paths = alpha_paths
                break
        else:
            # Preserve the original unqualified priority fallback when every
            # reference is opaque (or unreadable).
            paths = next(group for group in groups if group)
    path = paths[seed % len(paths)]
    return load_fault_patch(path, fault_patch_revision(path))


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
