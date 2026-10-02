# Panel rows, modules, cells, layout caps, and target geometry.
LOCATIONS = [
    {
        "id": "scenario_1",
        "name": "Location 1 - grass field",
        "terrain": "Original scene rebuilt as realistic grass terrain with the OG PVs removed.",
        "source": "scenario_1_terrain_balanced_topdown.png",
        "pv_source": "scenario_1_og_pv.jpeg",
        "crop": None,
        "blur": 0,
        "color": (1.00, 1.00, 1.00),
        "prebuilt": True,
    },
    {
        "id": "agri_rows",
        "name": "Location 2 - agricultural rows",
        "terrain": "Real top-down green PV field reference prepared for realistic module-level inspection.",
        "source": "agri_green_real_topdown_terrain_ai_filled.png",
        "pv_source": "agri_green_real_topdown_with_pvs.png",
        "crop": None,
        "blur": 0,
        "color": (1.00, 1.00, 1.00),
        "prebuilt": True,
    },
    {
        "id": "desert_farm",
        "name": "Location 3 - desert solar farm",
        "terrain": "Flat real desert reference from the supplied top-down PV image.",
        "source": "desert_background_no_pvs_flat_rough_balanced.png",
        "pv_source": "desert_user_reference_with_pvs.png",
        "crop": None,
        "blur": 0,
        "color": (1.00, 1.00, 1.00),
        "prebuilt": True,
    },
    {
        "id": "farm_lake",
        "name": "Location 4 - offshore water platform",
        "terrain": "Top-down offshore/water platform scene for floating PV placement.",
        "source": "offshore_water_platform_topdown.png",
        "pv_source": "offshore_water_platform_topdown.png",
        "crop": None,
        "blur": 0,
        "color": (1.00, 1.00, 1.00),
        "prebuilt": True,
    },
    {
        "id": "rooftop",
        "name": "Location 5 - real rooftop",
        "terrain": "Cleaned real rooftop reference with the old roof panels removed; OG PVs are reinstalled on the roof.",
        "source": "rooftop_background_balanced_topdown.png",
        "pv_source": "rooftop_with_og_pvs_center_pack_tool.png",
        "crop": None,
        "blur": 0,
        "color": (1.00, 1.00, 1.00),
        "prebuilt": True,
    },
]

# Visible PV rows in the OG scenario image. These drive masking, click targets, and fault placement.
ROW_DEFS = [
    {"row": 1, "x1": 0, "y1": 0, "x2": 763, "y2": 28, "cols": 1},
    {"row": 2, "x1": 0, "y1": 45, "x2": 763, "y2": 136, "cols": 1},
    {"row": 3, "x1": 0, "y1": 164, "x2": 620, "y2": 263, "cols": 1},
    {"row": 4, "x1": 0, "y1": 300, "x2": 376, "y2": 402, "cols": 1},
]

ROOFTOP_ROW_DEFS = [
    {"row": 1, "x1": 188, "y1": 225, "x2": 522, "y2": 247, "cols": 1},
    {"row": 2, "x1": 188, "y1": 268, "x2": 522, "y2": 290, "cols": 1},
    {"row": 3, "x1": 188, "y1": 311, "x2": 522, "y2": 333, "cols": 1},
    {"row": 4, "x1": 188, "y1": 354, "x2": 522, "y2": 376, "cols": 1},
]

DESERT_USER_ROW_DEFS = [
    {"row": 1, "x1": 104, "y1": 47, "x2": 648, "y2": 68, "cols": 1},
    {"row": 2, "x1": 104, "y1": 128, "x2": 648, "y2": 149, "cols": 1},
    {"row": 3, "x1": 104, "y1": 209, "x2": 648, "y2": 230, "cols": 1},
    {"row": 4, "x1": 104, "y1": 290, "x2": 648, "y2": 311, "cols": 1},
    {"row": 5, "x1": 104, "y1": 370, "x2": 648, "y2": 392, "cols": 1},
]

AGRI_GREEN_ROW_DEFS = [
    {"row": 1, "x1": 68, "y1": 41, "x2": 685, "y2": 101, "cols": 1},
    {"row": 2, "x1": 68, "y1": 130, "x2": 685, "y2": 191, "cols": 1},
    {"row": 3, "x1": 68, "y1": 218, "x2": 685, "y2": 279, "cols": 1},
    {"row": 4, "x1": 68, "y1": 304, "x2": 685, "y2": 365, "cols": 1},
]

LAYOUT_CAPS = {
    # Matrix evidence is mostly UAV/drone thermal/RGB module/cell inspection.
    # The crack/hotspot dataset is close-up 640px imagery, so generated layouts should use fewer,
    # larger modules instead of full-farm miniatures.
    "scenario_1": {"rows": 3, "modules": 5, "module_height": 112},
    "agri_rows": {"rows": 3, "modules": 5, "module_height": 112},
    "desert_farm": {"rows": 3, "modules": 5, "module_height": 108},
    "farm_lake": {"rows": 3, "modules": 4, "module_height": 96},
    # Rooftop needs shorter modules so 3 generated rows fit without merging.
    "rooftop": {"rows": 3, "modules": 5, "module_height": 64},
}

CAP_JUSTIFICATIONS = {
    "scenario_1": (
        "Cap: 3 rows x 5 modules. The default stays close-up, but extra modules are available "
        "when you need a denser row without making the cells unusably small."
    ),
    "agri_rows": (
        "Cap: 3 rows x 5 modules. Agrivoltaic rows keep visible alleys while allowing a longer "
        "module row when needed."
    ),
    "desert_farm": (
        "Cap: 3 rows x 5 modules. Desert imagery is kept at inspection scale while still allowing "
        "more modules per row."
    ),
    "farm_lake": (
        "Cap: 3 rows x 4 modules. The offshore platform has a smaller usable inner water square, so "
        "the module count is lower than the land scenes."
    ),
    "rooftop": (
        "Cap: 3 rows x 5 modules. Rooftop arrays stay close-up by default, with extra modules "
        "available when you want a denser row."
    ),
}

CUSTOM_LAYOUT_X_LIMITS = {
    # Keep max custom layouts inside the usable physical area of each source.
    "farm_lake": (226, 539),  # inner water square, away from the concrete frame
    "rooftop": (199, 609),    # user-marked black-box roof target zone
}

CUSTOM_LAYOUT_Y_LIMITS = {
    # Spread generated rows through a realistic usable zone instead of copying thin OG row strips.
    "farm_lake": (90, 310),
    "rooftop": (80, 310),
}

LOCATION_PANEL_DEFS = {
    "desert_farm": DESERT_USER_ROW_DEFS,
    "agri_rows": AGRI_GREEN_ROW_DEFS,
    "farm_lake": [
        {"row": 1, "x1": 234, "y1": 100, "x2": 506, "y2": 128, "cols": 1},
        {"row": 2, "x1": 234, "y1": 158, "x2": 506, "y2": 186, "cols": 1},
        {"row": 3, "x1": 234, "y1": 216, "x2": 506, "y2": 244, "cols": 1},
        {"row": 4, "x1": 234, "y1": 274, "x2": 506, "y2": 302, "cols": 1},
    ],
    "rooftop": ROOFTOP_ROW_DEFS,
}


def qp_get(name, default=None):
    try:
        value = st.query_params.get(name, default)
        return value[0] if isinstance(value, list) and value else value
    except Exception:
        values = st.experimental_get_query_params().get(name)
        return values[0] if values else default


def qp_set(**kwargs):
    cleaned = {key: value for key, value in kwargs.items() if value not in (None, "")}
    try:
        st.query_params.clear()
        for key, value in cleaned.items():
            st.query_params[key] = str(value)
    except Exception:
        st.experimental_set_query_params(**cleaned)


def qp_int(name, default, min_value, max_value):
    try:
        value = int(qp_get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def custom_layout_enabled():
    return str(qp_get("custom_layout", "1")).lower() in {"1", "true", "yes", "on"}


def row_prefix(location_id):
    return "RT" if location_id == "rooftop" else "R"


def layout_query_values(location_id=None):
    if not custom_layout_enabled():
        return {}
    return {
        "custom_layout": 1,
        "row_count": custom_row_count(location_id),
        "panels_per_row": custom_panels_per_row(location_id),
    }


def navigation_params(location_id, view, selected_pv, selected_cell, **overrides):
    params = {
        "location": location_id,
        "view": view,
        "selected_pv": selected_pv,
        "selected_cell": selected_cell,
    }
    params.update(layout_query_values(location_id))
    params.update(overrides)
    return params


def esc(value):
    return html.escape(str(value), quote=True)


def fit_crop(image, size=BASE_SIZE, crop=None):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if crop:
        w, h = image.size
        left, top, right, bottom = crop
        image = image.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))

    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - dst_w) // 2
    top = (resized.height - dst_h) // 2
    return resized.crop((left, top, left + dst_w, top + dst_h))


def load_rgb(path_name, crop=None):
    return fit_crop(Image.open(ASSET_DIR / path_name), BASE_SIZE, crop)


def annotation_path(location_id):
    return ANNOTATION_DIR / f"{location_id}.json"


@st.cache_data(show_spinner=False)
def load_annotation(location_id):
    path = annotation_path(location_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def annotation_rows_for_location(location_id):
    data = load_annotation(location_id)
    if not data:
        return None
    rows = []
    for idx, item in enumerate(data.get("rows", []), start=1):
        bbox = item.get("bbox") or [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
        if len(bbox) != 4 or any(value is None for value in bbox):
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "row": int(item.get("row", idx)),
                "x1": clamp(x1, 0, BASE_SIZE[0] - 1),
                "y1": clamp(y1, 0, BASE_SIZE[1] - 1),
                "x2": clamp(x2, x1 + 1, BASE_SIZE[0]),
                "y2": clamp(y2, y1 + 1, BASE_SIZE[1]),
                "cols": 1,
                "annotated": True,
            }
        )
    return rows or None


def annotation_row_for_id(location_id, row_id):
    data = load_annotation(location_id)
    if not data:
        return None
    prefix = row_prefix(location_id)
    try:
        row_number = int(str(row_id).replace(prefix, ""))
    except ValueError:
        row_number = None
    rows = data.get("rows", [])
    for idx, item in enumerate(rows, start=1):
        if item.get("id") == row_id or int(item.get("row", idx)) == row_number:
            return item
    return None


def cells_from_annotation(location_id, row_id, row):
    if custom_layout_enabled():
        return None
    annotated_row = annotation_row_for_id(location_id, row_id)
    if not annotated_row:
        return None

    cells = []
    direct_cells = annotated_row.get("cells") or []
    for idx, item in enumerate(direct_cells, start=1):
        bbox = item.get("bbox") or [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
        if len(bbox) != 4 or any(value is None for value in bbox):
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        pv_cell_number = int(item.get("number", idx))
        cell_id = item.get("id", f"{row_id}-C{pv_cell_number:03d}")
        cells.append(
            {
                "id": cell_id,
                "label": item.get("label", f"{row['label']} annotated PV cell {pv_cell_number}"),
                "row": row["row"],
                "cell_row": item.get("cell_row", 1),
                "cell_col": item.get("cell_col", pv_cell_number),
                "panel_number": item.get("module_number", pv_cell_number),
                "panels_in_row": len(direct_cells),
                "module_number": item.get("module_number"),
                "module_cell_row": item.get("module_cell_row"),
                "module_cell_col": item.get("module_cell_col"),
                "module_cell_number": item.get("module_cell_number"),
                "pv_cell_number_in_row": pv_cell_number,
                "pv_cells_in_row": len(direct_cells),
                "module_cell_grid": item.get("module_cell_grid"),
                "x1": clamp(x1, 0, BASE_SIZE[0] - 1),
                "y1": clamp(y1, 0, BASE_SIZE[1] - 1),
                "x2": clamp(x2, x1 + 1, BASE_SIZE[0]),
                "y2": clamp(y2, y1 + 1, BASE_SIZE[1]),
            }
        )
    if cells:
        return cells

    modules = annotated_row.get("modules") or []
    module_cell_cols = int(annotated_row.get("module_cell_cols", CUSTOM_MODULE_CELL_COLS))
    module_cell_rows = int(annotated_row.get("module_cell_rows", CUSTOM_MODULE_CELL_ROWS))
    for module_idx, module in enumerate(modules, start=1):
        bbox = module.get("bbox") or [module.get("x1"), module.get("y1"), module.get("x2"), module.get("y2")]
        if len(bbox) != 4 or any(value is None for value in bbox):
            continue
        mx1, my1, mx2, my2 = [int(round(float(value))) for value in bbox]
        module_w = max(1, mx2 - mx1)
        module_h = max(1, my2 - my1)
        for module_cell_row in range(1, module_cell_rows + 1):
            for module_cell_col in range(1, module_cell_cols + 1):
                pv_cell_number = len(cells) + 1
                module_cell_number = (module_cell_row - 1) * module_cell_cols + module_cell_col
                x1 = int(mx1 + (module_cell_col - 1) * module_w / module_cell_cols)
                x2 = int(mx1 + module_cell_col * module_w / module_cell_cols)
                y1 = int(my1 + (module_cell_row - 1) * module_h / module_cell_rows)
                y2 = int(my1 + module_cell_row * module_h / module_cell_rows)
                cells.append(
                    {
                        "id": f"{row_id}-C{pv_cell_number:03d}",
                        "label": f"{row['label']} annotated PV cell {pv_cell_number} (module {module_idx}, cell {module_cell_number})",
                        "row": row["row"],
                        "cell_row": module_cell_row,
                        "cell_col": (module_idx - 1) * module_cell_cols + module_cell_col,
                        "panel_number": module_idx,
                        "panels_in_row": len(modules),
                        "module_number": module_idx,
                        "module_cell_row": module_cell_row,
                        "module_cell_col": module_cell_col,
                        "module_cell_number": module_cell_number,
                        "pv_cell_number_in_row": pv_cell_number,
                        "pv_cells_in_row": len(modules) * module_cell_cols * module_cell_rows,
                        "module_cell_grid": {
                            "rows": module_cell_rows,
                            "cols": module_cell_cols,
                        },
                        "x1": clamp(x1, 0, BASE_SIZE[0] - 1),
                        "y1": clamp(y1, 0, BASE_SIZE[1] - 1),
                        "x2": clamp(x2, x1 + 1, BASE_SIZE[0]),
                        "y2": clamp(y2, y1 + 1, BASE_SIZE[1]),
                    }
                )
    return cells or None


def annotation_template_for_location(location_id):
    source_rows = LOCATION_PANEL_DEFS.get(location_id, ROW_DEFS)
    return {
        "version": 1,
        "location_id": location_id,
        "image_space": {"width": BASE_SIZE[0], "height": BASE_SIZE[1]},
        "note": "Fill each row bbox and module bbox from a real PV image. The app will split each module into 6x2 numbered cells unless cells are provided directly.",
        "rows": [
            {
                "row": row["row"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "module_cell_cols": CUSTOM_MODULE_CELL_COLS,
                "module_cell_rows": CUSTOM_MODULE_CELL_ROWS,
                "modules": [],
                "cells": [],
            }
            for row in source_rows
        ],
    }


def panel_mask():
    mask = Image.new("L", BASE_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    for row in ROW_DEFS:
        draw.rectangle((row["x1"], row["y1"], row["x2"], row["y2"]), fill=255)

    # Keep visible OG support/service strips with the PV layer.
    draw.rectangle((387, 26, 404, 164), fill=210)
    draw.rectangle((378, 263, 393, 402), fill=210)
    return mask.filter(ImageFilter.GaussianBlur(0.8))


def base_row_defs_for_location(location_id=None):
    annotated_rows = annotation_rows_for_location(location_id)
    if annotated_rows:
        return annotated_rows
    return LOCATION_PANEL_DEFS.get(location_id, ROW_DEFS)


def max_custom_rows(location_id=None):
    return min(MAX_LAYOUT_ROWS, LAYOUT_CAPS.get(location_id, {}).get("rows", MAX_LAYOUT_ROWS))


def max_custom_panels_per_row(location_id=None):
    return min(MAX_PANELS_PER_ROW, LAYOUT_CAPS.get(location_id, {}).get("modules", MAX_PANELS_PER_ROW))


def custom_row_count(location_id=None):
    return qp_int("row_count", len(base_row_defs_for_location(location_id)), 1, max_custom_rows(location_id))


def default_panels_per_row(location_id=None):
    if location_id == "scenario_1":
        return 2
    if location_id == "agri_rows":
        return 2
    if location_id == "desert_farm":
        return 2
    if location_id == "rooftop":
        return 2
    if location_id == "farm_lake":
        return 2
    base_rows = base_row_defs_for_location(location_id)
    avg_width = sum(row["x2"] - row["x1"] for row in base_rows) / max(1, len(base_rows))
    return max(3, min(max_custom_panels_per_row(location_id), round(avg_width / 86)))


def custom_panels_per_row(location_id=None):
    return qp_int("panels_per_row", default_panels_per_row(location_id), 1, max_custom_panels_per_row(location_id))


def preferred_custom_module_height(location_id=None):
    return LAYOUT_CAPS.get(location_id, {}).get("module_height", 28)


def module_row_pixel_width(module_count, module_height):
    """Keep module geometry physical: row length grows with module count."""
    module_gap = max(1, int(round(module_height * CUSTOM_MODULE_GAP_RATIO))) if module_count > 1 else 0
    module_width = max(1, int(round(module_height * CUSTOM_MODULE_ASPECT_RATIO)))
    row_width = module_count * module_width + (module_count - 1) * module_gap
    return row_width, module_width, module_gap


def module_layouts_for_dimensions(width, height, module_count, scale=4, high_res=False):
    """Shared module/cell geometry for RGB render, thermal render, and hit boxes."""
    hi_w = max(64, int(width * scale))
    hi_h = max(32, int(height * scale))
    module_gap = max(scale, int(round(hi_h * CUSTOM_MODULE_GAP_RATIO))) if module_count > 1 else 0
    frame = max(2, int(hi_h * 0.055))
    module_w = max(4, int(round(hi_h * CUSTOM_MODULE_ASPECT_RATIO)))
    actual_row_w = module_count * module_w + max(0, module_count - 1) * module_gap
    start_x = max(0, int(round((hi_w - actual_row_w) / 2)))
    layouts = []
    for module_idx in range(module_count):
        mx1 = int(start_x + module_idx * (module_w + module_gap))
        mx2 = min(hi_w - 1, int(mx1 + module_w))
        if mx2 - mx1 < 6:
            continue
        ox1 = clamp(mx1, 0, hi_w - 2)
        ox2 = clamp(mx2, ox1 + 6, hi_w - 1)
        oy1 = 0
        oy2 = hi_h - 1
        layout = {
            "module_number": module_idx + 1,
            "outer": (ox1, oy1, ox2, oy2),
            "inner": (ox1 + frame, oy1 + frame, ox2 - frame, oy2 - frame),
        }
        layouts.append(layout)

    if high_res:
        return layouts, hi_w, hi_h

    sx = width / max(1, hi_w)
    sy = height / max(1, hi_h)
    converted = []
    for layout in layouts:
        converted.append(
            {
                "module_number": layout["module_number"],
                "outer": tuple(int(round(value * (sx if idx % 2 == 0 else sy))) for idx, value in enumerate(layout["outer"])),
                "inner": tuple(int(round(value * (sx if idx % 2 == 0 else sy))) for idx, value in enumerate(layout["inner"])),
            }
        )
    return converted, width, height


def module_layouts_for_row(row, module_count):
    width = max(1, row["x2"] - row["x1"])
    height = max(1, row["y2"] - row["y1"])
    layouts, _, _ = module_layouts_for_dimensions(width, height, module_count)
    shifted = []
    for layout in layouts:
        shifted.append(
            {
                "module_number": layout["module_number"],
                "outer": tuple(value + (row["x1"] if idx % 2 == 0 else row["y1"]) for idx, value in enumerate(layout["outer"])),
                "inner": tuple(value + (row["x1"] if idx % 2 == 0 else row["y1"]) for idx, value in enumerate(layout["inner"])),
            }
        )
    return shifted


def dynamic_custom_row_bounds(row_x1, row_x2, center_y, step, module_count, location_id=None):
    source_center_x = (row_x1 + row_x2) / 2
    preferred_height = preferred_custom_module_height(location_id)
    max_height = max(16, int(step * 0.80))
    target_height = min(preferred_height, max_height)
    min_height = max(12, int(target_height * 0.5))
    if location_id in CUSTOM_LAYOUT_X_LIMITS:
        available_x1, available_x2 = CUSTOM_LAYOUT_X_LIMITS[location_id]
    else:
        available_x1 = 8
        available_x2 = BASE_SIZE[0] - 8
    available_width = available_x2 - available_x1

    if location_id in CUSTOM_LAYOUT_Y_LIMITS:
        available_y1, available_y2 = CUSTOM_LAYOUT_Y_LIMITS[location_id]
    else:
        available_y1, available_y2 = 0, BASE_SIZE[1]

    row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)
    if row_width > available_width:
        module_gap = max(1, int(round(target_height * CUSTOM_MODULE_GAP_RATIO))) if module_count > 1 else 0
        fit_height = int(
            (available_width - module_gap * (module_count - 1))
            / max(1, module_count * CUSTOM_MODULE_ASPECT_RATIO)
        )
        target_height = clamp(fit_height, min_height, target_height)
        row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)
        if row_width > available_width:
            module_gap = 1 if module_count > 1 else 0
            target_height = clamp(
                int(
                    (available_width - module_gap * (module_count - 1))
                    / max(1, module_count * CUSTOM_MODULE_ASPECT_RATIO)
                ),
                8,
                target_height,
            )
            row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)

    row_width = min(row_width, available_width)
    left = int(round(source_center_x - row_width / 2))
    left = clamp(left, available_x1, available_x2 - row_width)
    right = left + row_width
    top = int(round(center_y - target_height / 2))
    bottom = top + target_height
    if top < available_y1:
        bottom += available_y1 - top
        top = available_y1
    if bottom > available_y2:
        top -= bottom - available_y2
        bottom = available_y2
    top = max(top, 0)
    return int(left), int(top), int(right), int(bottom)


def custom_row_defs_for_location(location_id=None):
    base_rows = base_row_defs_for_location(location_id)
    row_count = custom_row_count(location_id)
    module_count = custom_panels_per_row(location_id)
    x1 = min(row["x1"] for row in base_rows)
    y1 = min(row["y1"] for row in base_rows)
    x2 = max(row["x2"] for row in base_rows)
    y2 = max(row["y2"] for row in base_rows)
    if location_id in CUSTOM_LAYOUT_Y_LIMITS:
        y1, y2 = CUSTOM_LAYOUT_Y_LIMITS[location_id]
    total_height = max(1, y2 - y1)
    step = total_height / row_count
    og_style_locations = {"scenario_1", "agri_rows"}
    guide_rows = sorted(base_rows, key=lambda item: (item["y1"] + item["y2"]) / 2)

    def interpolated_edge(edge_name, normalized_y):
        if location_id not in og_style_locations or len(guide_rows) < 2:
            return x1 if edge_name == "x1" else x2
        guide = [
            (((row["y1"] + row["y2"]) / 2 - y1) / total_height, row[edge_name])
            for row in guide_rows
        ]
        if normalized_y <= guide[0][0]:
            return guide[0][1]
        if normalized_y >= guide[-1][0]:
            return guide[-1][1]
        for left, right in zip(guide, guide[1:]):
            if left[0] <= normalized_y <= right[0]:
                span = max(0.0001, right[0] - left[0])
                ratio = (normalized_y - left[0]) / span
                return int(round(left[1] + (right[1] - left[1]) * ratio))
        return guide[-1][1]

    rows = []
    for idx in range(row_count):
        row_rng = np.random.default_rng(seed_for(location_id, row_count, module_count, idx, "custom-row-drift"))
        center_y = y1 + step * (idx + 0.5)
        center_y += float(row_rng.normal(0, max(0.6, min(4.0, step * 0.035))))
        normalized_y = (center_y - y1) / total_height
        row_x1 = int(interpolated_edge("x1", normalized_y))
        row_x2 = int(interpolated_edge("x2", normalized_y))
        if row_x2 - row_x1 < 80:
            row_x2 = min(BASE_SIZE[0], row_x1 + 80)
        row_x1, top, row_x2, bottom = dynamic_custom_row_bounds(
            row_x1, row_x2, center_y, step, module_count, location_id
        )
        row_w = max(1, row_x2 - row_x1)
        dx = int(round(row_rng.normal(0, max(1.0, min(7.0, row_w * 0.008)))))
        dw = int(round(row_rng.normal(0, max(1.0, min(5.0, row_w * 0.004)))))
        row_x1 = clamp(row_x1 + dx - max(0, dw), 0, BASE_SIZE[0] - 2)
        row_x2 = clamp(row_x2 + dx + max(0, dw), row_x1 + 80, BASE_SIZE[0])
        rows.append(
            {
                "row": idx + 1,
                "x1": row_x1,
                "y1": max(0, top),
                "x2": row_x2,
                "y2": min(BASE_SIZE[1], bottom),
                "cols": 1,
            }
        )
    return rows


def row_defs_for_location(location_id=None):
    if custom_layout_enabled():
        return custom_row_defs_for_location(location_id)
    return base_row_defs_for_location(location_id)


def panels(location_id=None):
    items = []
    prefix = row_prefix(location_id)
    for row in row_defs_for_location(location_id):
        pad = 0 if row.get("annotated") else 2
        items.append(
            {
                "id": f"{prefix}{row['row']}",
                "label": f"Row {row['row']}",
                "row": row["row"],
                "col": 1,
                "x1": row["x1"] + pad,
                "y1": row["y1"] + pad,
                "x2": row["x2"] - pad,
                "y2": row["y2"] - pad,
            }
        )
    return items


def row_by_id(location_id, row_id):
    location_rows = panels(location_id)
    return next((row for row in location_rows if row["id"] == row_id), location_rows[0])


def cells_for_row(location_id, row_id):
    row = row_by_id(location_id, row_id)
    width = row["x2"] - row["x1"]
    height = row["y2"] - row["y1"]
    annotated_cells = cells_from_annotation(location_id, row_id, row)
    if annotated_cells:
        return annotated_cells
    if custom_layout_enabled():
        modules = custom_panels_per_row(location_id)
        module_cell_cols = CUSTOM_MODULE_CELL_COLS
        module_cell_rows = CUSTOM_MODULE_CELL_ROWS
        layouts = module_layouts_for_row(row, modules)
        items = []
        for layout in layouts:
            module_number = layout["module_number"]
            ix1, iy1, ix2, iy2 = layout["inner"]
            module_w = max(1, ix2 - ix1)
            module_h = max(1, iy2 - iy1)
            for module_cell_row in range(1, module_cell_rows + 1):
                for module_cell_col in range(1, module_cell_cols + 1):
                    pv_cell_number = len(items) + 1
                    module_cell_number = (module_cell_row - 1) * module_cell_cols + module_cell_col
                    cell_id = f"{row['id']}-C{pv_cell_number:03d}"
                    label = f"{row['label']} PV cell {pv_cell_number} (module {module_number}, cell {module_cell_number})"
                    x1 = int(ix1 + (module_cell_col - 1) * module_w / module_cell_cols)
                    x2 = int(ix1 + module_cell_col * module_w / module_cell_cols)
                    y1 = int(iy1 + (module_cell_row - 1) * module_h / module_cell_rows)
                    y2 = int(iy1 + module_cell_row * module_h / module_cell_rows)
                    if x2 <= x1:
                        x2 = min(row["x2"], x1 + 1)
                    if y2 <= y1:
                        y2 = min(row["y2"], y1 + 1)
                    items.append(
                        {
                            "id": cell_id,
                            "label": label,
                            "row": row["row"],
                            "cell_row": module_cell_row,
                            "cell_col": (module_number - 1) * module_cell_cols + module_cell_col,
                            "panel_number": module_number,
                            "panels_in_row": modules,
                            "module_number": module_number,
                            "module_cell_row": module_cell_row,
                            "module_cell_col": module_cell_col,
                            "module_cell_number": module_cell_number,
                            "pv_cell_number_in_row": pv_cell_number,
                            "pv_cells_in_row": len(layouts) * module_cell_cols * module_cell_rows,
                            "module_cell_grid": {
                                "rows": module_cell_rows,
                                "cols": module_cell_cols,
                            },
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
        return items
    else:
        modules = None
        module_cell_cols = None
        module_cell_rows = None
        module_gap = 0
        module_width = width
        cols = max(8, min(24, round(width / 34)))
        rows = max(1, min(3, round(height / 34)))
    cell_w = max(1, width / max(1, cols))
    cell_h = max(1, height / max(1, rows))
    gap_x = 0 if custom_layout_enabled() else max(0, min(3, int(cell_w * 0.08)))
    gap_y = 0 if custom_layout_enabled() else max(0, min(3, int(cell_h * 0.08)))
    items = []
    for cy in range(rows):
        for cx in range(cols):
            pv_cell_number = cy * cols + cx + 1
            if custom_layout_enabled():
                module_number = cx // module_cell_cols + 1
                module_cell_col = cx % module_cell_cols + 1
                module_cell_row = cy % module_cell_rows + 1
                module_cell_number = (module_cell_row - 1) * module_cell_cols + module_cell_col
                cell_id = f"{row['id']}-C{pv_cell_number:03d}"
                label = f"{row['label']} PV cell {pv_cell_number} (module {module_number}, cell {module_cell_number})"
                module_left = row["x1"] + (module_number - 1) * (module_width + module_gap)
                x1 = int(module_left + (module_cell_col - 1) * module_width / module_cell_cols)
                x2 = int(module_left + module_cell_col * module_width / module_cell_cols)
            else:
                module_number = None
                module_cell_col = None
                module_cell_row = None
                module_cell_number = None
                cell_id = f"{row['id']}-P{pv_cell_number:03d}"
                label = f"{row['label']} panel {pv_cell_number}"
                x1 = int(row["x1"] + cx * width / cols + gap_x)
                x2 = int(row["x1"] + (cx + 1) * width / cols - gap_x)
            y1 = int(row["y1"] + cy * height / rows + gap_y)
            y2 = int(row["y1"] + (cy + 1) * height / rows - gap_y)
            if x2 <= x1:
                x2 = min(row["x2"], x1 + 1)
            if y2 <= y1:
                y2 = min(row["y2"], y1 + 1)
            items.append(
                {
                    "id": cell_id,
                    "label": label,
                    "row": row["row"],
                    "cell_row": cy + 1,
                    "cell_col": cx + 1,
                    "panel_number": module_number or pv_cell_number,
                    "panels_in_row": modules or rows * cols,
                    "module_number": module_number,
                    "module_cell_row": module_cell_row,
                    "module_cell_col": module_cell_col,
                    "module_cell_number": module_cell_number,
                    "pv_cell_number_in_row": pv_cell_number,
                    "pv_cells_in_row": rows * cols,
                    "module_cell_grid": {
                        "rows": module_cell_rows,
                        "cols": module_cell_cols,
                    }
                    if custom_layout_enabled()
                    else None,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return items


def cell_by_id(location_id, row_id, cell_id):
    cells = cells_for_row(location_id, row_id)
    return next((cell for cell in cells if cell["id"] == cell_id), cells[0])


def target_counts(location_id):
    location_rows = panels(location_id)
    per_row_cells = [len(cells_for_row(location_id, row["id"])) for row in location_rows]
    return {
        "rows": len(location_rows),
        "panel_cells": sum(per_row_cells),
        "pv_cells": sum(per_row_cells),
    }


def module_bbox_for_cell(location_id, row_id, cell):
    module_number = cell.get("module_number")
    if not module_number:
        return {
            "x1": cell["x1"],
            "y1": cell["y1"],
            "x2": cell["x2"],
            "y2": cell["y2"],
        }
    module_cells = [
        item for item in cells_for_row(location_id, row_id)
        if item.get("module_number") == module_number
    ]
    if not module_cells:
        return {
            "x1": cell["x1"],
            "y1": cell["y1"],
            "x2": cell["x2"],
            "y2": cell["y2"],
        }
    return {
        "x1": min(item["x1"] for item in module_cells),
        "y1": min(item["y1"] for item in module_cells),
        "x2": max(item["x2"] for item in module_cells),
        "y2": max(item["y2"] for item in module_cells),
    }


def cell_with_fault_bbox(cell, record):
    fault_bbox = record.get("fault_bbox_px") or record.get("module_bbox_px")
    if not fault_bbox:
        return cell
    item = dict(cell)
    item["fault_bbox_px"] = {
        "x1": int(fault_bbox["x1"]),
        "y1": int(fault_bbox["y1"]),
        "x2": int(fault_bbox["x2"]),
        "y2": int(fault_bbox["y2"]),
    }
    return item


def fault_bounds_for_cell(cell):
    fault_bbox = cell.get("fault_bbox_px")
    if fault_bbox:
        return (
            int(fault_bbox["x1"]),
            int(fault_bbox["y1"]),
            int(fault_bbox["x2"]),
            int(fault_bbox["y2"]),
        )
    return cell["x1"], cell["y1"], cell["x2"], cell["y2"]


def fault_render_bounds_for_cell(cell, fault_type):
    fault_type = canonical_fault_type(fault_type)
    cell_bounds = (cell["x1"], cell["y1"], cell["x2"], cell["y2"])
    module_bounds = fault_bounds_for_cell(cell)
    if fault_type in CRACKING_FAULTS:
        return cell_bounds
    if fault_type in SURFACE_OBSTRUCTION_FAULTS | SOILING_FAULTS:
        cx1, cy1, cx2, cy2 = cell_bounds
        return cx1, cy1, cx2, cy2
    if fault_type in DIODE_FAULTS | BYPASSED_FAULTS | STRING_FAULTS:
        cx1, cy1, cx2, cy2 = cell_bounds
        return cx1, cy1, cx2, cy2
    if fault_type not in HOTSPOT_FAULTS:
        return module_bounds

    cx1, cy1, cx2, cy2 = cell_bounds
    mw1, my1, mw2, my2 = module_bounds
    cw, ch = max(2, cx2 - cx1), max(2, cy2 - cy1)
    # Hotspots should read as local cell damage, not a module-wide cloud.
    # Give them a small halo around the selected cell, clipped to the module.
    halo_x = int(cw * (0.35 if fault_type == "SingleHotSpot" else 0.18))
    halo_y = int(ch * (0.42 if fault_type == "SingleHotSpot" else 0.22))
    if fault_type == "MultiHotSpot":
        return module_bounds
    return (
        clamp(cx1 - halo_x, mw1, mw2),
        clamp(cy1 - halo_y, my1, my2),
        clamp(cx2 + halo_x, mw1, mw2),
        clamp(cy2 + halo_y, my1, my2),
    )


def fault_annotation_bounds_for_cell(cell, fault_type):
    fault_type = canonical_fault_type(fault_type)
    cx1, cy1, cx2, cy2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
    mx1, my1, mx2, my2 = fault_bounds_for_cell(cell)
    cw, ch = max(2, cx2 - cx1), max(2, cy2 - cy1)

    if fault_type == "SingleHotSpot":
        pad_x = int(cw * 0.22)
        pad_y = int(ch * 0.26)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type == "MultiHotSpot":
        pad_x = int(cw * 0.38)
        pad_y = int(ch * 0.36)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in DIODE_FAULTS:
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.08)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in BYPASSED_FAULTS:
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.08)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in STRING_FAULTS | LEGACY_LINE_FAULTS:
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.08)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in THERMAL_BLOCK_FAULTS:
        pad_x = int(cw * 0.30)
        pad_y = int(ch * 0.30)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in SURFACE_OBSTRUCTION_FAULTS | SOILING_FAULTS:
        pad_x = int(cw * 0.04)
        pad_y = int(ch * 0.06)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    if fault_type in CRACKING_FAULTS:
        pad_x = int(cw * 0.04)
        pad_y = int(ch * 0.06)
        return (
            clamp(cx1 - pad_x, mx1, mx2),
            clamp(cy1 - pad_y, my1, my2),
            clamp(cx2 + pad_x, mx1, mx2),
            clamp(cy2 + pad_y, my1, my2),
        )
    return cx1, cy1, cx2, cy2


def fault_scope_note(fault_type):
    fault_type = canonical_fault_type(fault_type)
    if fault_type in DIODE_FAULTS:
        return "Thermal proxy for bypass-diode evidence; confirm with electrical/string context."
    if fault_type in BYPASSED_FAULTS:
        return "Thermal proxy for a bypassed substring/section; generated on the selected cell only."
    if fault_type in STRING_FAULTS:
        return "Thermal proxy for string-level behavior; image-only evidence is not final diagnosis."
    if fault_type in SURFACE_OBSTRUCTION_FAULTS:
        return "Visible/thermal surface obstruction on the selected cell."
    if fault_type in SOILING_FAULTS:
        return "Cell-level dust/soiling evidence on the selected cell."
    if fault_type in CRACKING_FAULTS:
        return "Cell-level cracking evidence on the selected cell."
    return "Selected-cell visual evidence."


def fault_visible_bbox_for_cell(cell, fault_type):
    x1, y1, x2, y2 = fault_annotation_bounds_for_cell(cell, fault_type)
    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def panel_segmentation_preserve_mask(shape, cell):
    height, width = shape[:2]
    preserve = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = fault_bounds_for_cell(cell)
    module_w = max(2, x2 - x1)
    module_h = max(2, y2 - y1)
    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    line_w = max(1, min(3, int(round(min(module_w / cols, module_h / rows) * 0.18))))

    cv2.rectangle(preserve, (x1, y1), (x2 - 1, y2 - 1), 1.0, line_w)
    for idx in range(1, cols):
        x = int(x1 + idx * module_w / cols)
        cv2.line(preserve, (x, y1), (x, y2 - 1), 1.0, line_w, cv2.LINE_AA)
    for idx in range(1, rows):
        y = int(y1 + idx * module_h / rows)
        cv2.line(preserve, (x1, y), (x2 - 1, y), 1.0, line_w, cv2.LINE_AA)
    preserve = cv2.GaussianBlur(preserve, (0, 0), max(0.35, line_w * 0.55))
    return np.clip(preserve, 0, 1)
