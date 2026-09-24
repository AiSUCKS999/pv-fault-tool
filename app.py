import base64
import html
import json
import math
import random
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ANNOTATION_DIR = ROOT / "annotations"
BASE_SIZE = (763, 402)
MAX_LAYOUT_ROWS = 30
MAX_PANELS_PER_ROW = 16
CUSTOM_MODULE_CELL_COLS = 6
CUSTOM_MODULE_CELL_ROWS = 2
CUSTOM_MODULE_ASPECT_RATIO = 1.7
CUSTOM_MODULE_GAP_RATIO = 0.12
DEFAULT_FAULT_SCALE = 5

FAULT_TYPES = {
    "Hotspot": "Irregular high-temperature cell hotspot with a soft thermal halo",
    "Bypass diode": "Hot substring/bypass-band pattern across the selected cell group",
    "Soiling / dust": "Irregular dust/soiling patch with cooler uneven thermal response",
    "Bird droppings": "Dataset-backed bird-drop occlusion fault on one selected PV cell",
    "Snow cover": "Cold occluding snow/ice patch based on solar panel defect datasets",
    "Partial shading": "Cool shaded band with realistic thermal drop across cell regions",
    "Crack / damage": "Jagged crack-like defect with local heat points along the line",
    "Physical damage": "Localized broken-glass/cell damage from visual defect classes",
    "Electrical / open circuit": "Electrical-string style abnormal thermal band on the selected cell",
    "Degradation / resistance": "Resistive degradation fault with mild distributed heating",
}

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
    # Keep this editable view at a readable single-frame inspection scale.
    "scenario_1": {"rows": 4, "modules": 10, "module_height": 39},
    "agri_rows": {"rows": 4, "modules": 12, "module_height": 39},
    "desert_farm": {"rows": 5, "modules": 12, "module_height": 31},
    "farm_lake": {"rows": 4, "modules": 5, "module_height": 36},
    "rooftop": {"rows": 4, "modules": 8, "module_height": 36},
}

CAP_JUSTIFICATIONS = {
    "scenario_1": (
        "Cap: 4 rows x 10 modules. The literature/GitHub review examples are mostly UAV/thermal "
        "inspection crops with a few array rows and tens of module/cell boxes per image; this keeps "
        "a single-frame grass scene readable without shrinking cells into stripes."
    ),
    "agri_rows": (
        "Cap: 4 rows x 12 modules. Agrivoltaic/UAV rows need visible alleys/tram lines between rows; "
        "the cap preserves row spacing while still giving a dense inspection target."
    ),
    "desert_farm": (
        "Cap: 5 rows x 12 modules. Desert UAV imagery tolerates one extra row because the terrain is "
        "visually simpler, but module count is capped so thermal/cell labels remain inspectable."
    ),
    "farm_lake": (
        "Cap: 4 rows x 5 modules. The offshore platform has a small usable inner water square; this "
        "prevents rows from leaving the frame and keeps floating-PV spacing plausible."
    ),
    "rooftop": (
        "Cap: 4 rows x 8 modules. Rooftop studies/datasets use compact roof arrays around obstructions; "
        "this cap avoids HVAC overlap and keeps individual modules large enough for fault targeting."
    ),
}

CUSTOM_LAYOUT_X_LIMITS = {
    # Keep max custom layouts inside the usable physical area of each source.
    "farm_lake": (226, 539),  # inner water square, away from the concrete frame
    "rooftop": (199, 609),    # user-marked black-box roof target zone
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
    return str(qp_get("custom_layout", "0")).lower() in {"1", "true", "yes", "on"}


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
    return []


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
    if location_id == "agri_rows":
        return 12
    if location_id == "desert_farm":
        return 12
    base_rows = base_row_defs_for_location(location_id)
    avg_width = sum(row["x2"] - row["x1"] for row in base_rows) / max(1, len(base_rows))
    return max(8, min(max_custom_panels_per_row(location_id), round(avg_width / 34)))


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
    frame = max(3, int(hi_h * 0.08))
    module_w = max(4, int(round(hi_h * CUSTOM_MODULE_ASPECT_RATIO)))
    actual_row_w = module_count * module_w + max(0, module_count - 1) * module_gap
    start_x = max(0, int(round((hi_w - actual_row_w) / 2)))
    layouts = []
    for module_idx in range(module_count):
        mx1 = int(start_x + module_idx * (module_w + module_gap))
        mx2 = min(hi_w - 1, int(mx1 + module_w))
        if mx2 - mx1 < 6:
            continue
        layout = {
            "module_number": module_idx + 1,
            "outer": (mx1, 0, mx2, hi_h - 1),
            "inner": (mx1 + frame, frame, mx2 - frame, hi_h - frame),
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
    max_height = max(preferred_height, min(58, int(step * 0.82)))
    target_height = min(preferred_height, max_height)
    if location_id in CUSTOM_LAYOUT_X_LIMITS:
        available_x1, available_x2 = CUSTOM_LAYOUT_X_LIMITS[location_id]
    else:
        available_x1 = 8
        available_x2 = BASE_SIZE[0] - 8
    available_width = available_x2 - available_x1

    row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)
    if row_width > available_width:
        module_gap = max(1, int(round(target_height * CUSTOM_MODULE_GAP_RATIO))) if module_count > 1 else 0
        fit_height = int(
            (available_width - module_gap * (module_count - 1))
            / max(1, module_count * CUSTOM_MODULE_ASPECT_RATIO)
        )
        target_height = clamp(fit_height, max(18, int(preferred_height * 0.72)), target_height)
        row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)
        if row_width > available_width:
            module_gap = 1 if module_count > 1 else 0
            target_height = clamp(
                int((available_width - module_gap * (module_count - 1)) / max(1, module_count * CUSTOM_MODULE_ASPECT_RATIO)),
                max(16, int(preferred_height * 0.66)),
                target_height,
            )
            row_width, module_width, module_gap = module_row_pixel_width(module_count, target_height)

    row_width = min(row_width, available_width)
    left = int(round(source_center_x - row_width / 2))
    left = clamp(left, available_x1, available_x2 - row_width)
    right = left + row_width
    top = int(round(center_y - target_height / 2))
    bottom = top + target_height
    if top < 0:
        bottom -= top
        top = 0
    if bottom > BASE_SIZE[1]:
        top -= bottom - BASE_SIZE[1]
        bottom = BASE_SIZE[1]
    return int(left), int(top), int(right), int(bottom)


def custom_row_defs_for_location(location_id=None):
    base_rows = base_row_defs_for_location(location_id)
    row_count = custom_row_count(location_id)
    module_count = custom_panels_per_row(location_id)
    x1 = min(row["x1"] for row in base_rows)
    y1 = min(row["y1"] for row in base_rows)
    x2 = max(row["x2"] for row in base_rows)
    y2 = max(row["y2"] for row in base_rows)
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
        center_y = y1 + step * (idx + 0.5)
        normalized_y = (center_y - y1) / total_height
        row_x1 = int(interpolated_edge("x1", normalized_y))
        row_x2 = int(interpolated_edge("x2", normalized_y))
        if row_x2 - row_x1 < 80:
            row_x2 = min(BASE_SIZE[0], row_x1 + 80)
        row_x1, top, row_x2, bottom = dynamic_custom_row_bounds(row_x1, row_x2, center_y, step, module_count, location_id)
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


def fault_record(location, view, row, cell, fault_type, fault_scale):
    cx = round((cell["x1"] + cell["x2"]) / 2, 2)
    cy = round((cell["y1"] + cell["y2"]) / 2, 2)
    return {
        "location_id": location["id"],
        "location_name": location["name"],
        "view": view.upper(),
        "row_id": row["id"],
        "row_label": row["label"],
        "target_cell_id": cell["id"],
        "pv_cell_number_in_row": cell.get("pv_cell_number_in_row", cell.get("panel_number")),
        "pv_cells_in_selected_row": cell.get("pv_cells_in_row", cell.get("panels_in_row")),
        "module_number_in_row": cell.get("module_number"),
        "module_cell_number": cell.get("module_cell_number"),
        "module_cell_row": cell.get("module_cell_row"),
        "module_cell_col": cell.get("module_cell_col"),
        "target_label": cell["label"],
        # Kept for old CSV/JSON consumers that expected panel_id.
        "panel_id": cell["id"],
        "panel_number_in_row": cell.get("panel_number"),
        "panels_in_selected_row": cell.get("panels_in_row"),
        "panel_label": cell["label"],
        "fault_type": fault_type,
        "fault_scale": fault_scale,
        "layout": {
            "custom": custom_layout_enabled(),
            "row_count": len(panels(location["id"])),
            "panels_per_row": custom_panels_per_row(location["id"]) if custom_layout_enabled() else len(cells_for_row(location["id"], row["id"])),
            "pv_cells_per_row": len(cells_for_row(location["id"], row["id"])),
            "cells_per_module": CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS if custom_layout_enabled() else None,
        },
        "bbox_px": {
            "x1": cell["x1"],
            "y1": cell["y1"],
            "x2": cell["x2"],
            "y2": cell["y2"],
        },
        "center_px": {"x": cx, "y": cy},
        "center_normalized": {
            "x": round(cx / BASE_SIZE[0], 4),
            "y": round(cy / BASE_SIZE[1], 4),
        },
    }


def fault_record_csv(record):
    flat = {
        "location_id": record["location_id"],
        "location_name": record["location_name"],
        "view": record["view"],
        "row_id": record["row_id"],
        "target_cell_id": record["target_cell_id"],
        "pv_cell_number_in_row": record["pv_cell_number_in_row"],
        "pv_cells_in_selected_row": record["pv_cells_in_selected_row"],
        "module_number_in_row": record["module_number_in_row"],
        "module_cell_number": record["module_cell_number"],
        "fault_type": record["fault_type"],
        "fault_scale": record["fault_scale"],
        "x1": record["bbox_px"]["x1"],
        "y1": record["bbox_px"]["y1"],
        "x2": record["bbox_px"]["x2"],
        "y2": record["bbox_px"]["y2"],
        "center_x": record["center_px"]["x"],
        "center_y": record["center_px"]["y"],
        "center_x_norm": record["center_normalized"]["x"],
        "center_y_norm": record["center_normalized"]["y"],
    }
    headers = list(flat)
    values = [str(flat[key]).replace('"', '""') for key in headers]
    return ",".join(headers) + "\n" + ",".join(f'"{value}"' for value in values)


def fault_records_csv(records):
    if not records:
        return "location_id,row_id,target_cell_id,fault_type,fault_scale\n"
    header = fault_record_csv(records[0]).splitlines()[0]
    rows = [fault_record_csv(record).splitlines()[1] for record in records]
    return header + "\n" + "\n".join(rows)


def fault_table_rows(records):
    rows = []
    for idx, record in enumerate(records, start=1):
        rows.append(
            {
                "#": idx,
                "location": record["location_name"],
                "view": record["view"],
                "row": record["row_label"],
                "pv_cell_id": record["target_cell_id"],
                "cell_number": record["pv_cell_number_in_row"],
                "module": record["module_number_in_row"],
                "module_cell": record["module_cell_number"],
                "fault_type": record["fault_type"],
                "x1": record["bbox_px"]["x1"],
                "y1": record["bbox_px"]["y1"],
                "x2": record["bbox_px"]["x2"],
                "y2": record["bbox_px"]["y2"],
            }
        )
    return rows


def fault_type_counts(records):
    counts = {}
    for record in records:
        fault_type = record["fault_type"]
        counts[fault_type] = counts.get(fault_type, 0) + 1
    return counts


def fault_summary_rows(records):
    return [{"fault_type": key, "count": value} for key, value in fault_type_counts(records).items()]


def image_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def cropped_row_image(image, location_id, selected_row_id, selected_cell_id):
    row = row_by_id(location_id, selected_row_id)
    crop = crop_box_for_row(row)
    cropped = image.crop(crop)
    zoom_scale = max(1, min(3, int(980 / max(1, cropped.size[0]))))
    if zoom_scale > 1:
        cropped = cropped.resize((cropped.size[0] * zoom_scale, cropped.size[1] * zoom_scale), Image.Resampling.LANCZOS)
    return draw_visible_cell_grid_on_crop(cropped, location_id, selected_row_id, selected_cell_id, crop, zoom_scale).convert("RGB")


def scene_metadata(location, view, selected_row, selected_cell, records):
    return {
        "location_id": location["id"],
        "location_name": location["name"],
        "view": view.upper(),
        "custom_layout": custom_layout_enabled(),
        "row_count": len(panels(location["id"])),
        "modules_per_row": custom_panels_per_row(location["id"]) if custom_layout_enabled() else default_panels_per_row(location["id"]),
        "selected_row": selected_row["id"],
        "selected_cell": selected_cell["id"],
        "fault_count": len(records),
        "fault_type_counts": fault_type_counts(records),
        "layout_cap": LAYOUT_CAPS.get(location["id"]),
    }


def export_package_bytes(location, view, selected_row, selected_cell, records, terrain, pv_scene):
    metadata = scene_metadata(location, view, selected_row, selected_cell, records)
    cropped = cropped_row_image(pv_scene, location["id"], selected_row["id"], selected_cell["id"])
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("terrain.png", image_png_bytes(terrain))
        package.writestr("pv_scene.png", image_png_bytes(pv_scene))
        package.writestr("cropped_row.png", image_png_bytes(cropped))
        package.writestr("faults.json", json.dumps(records, indent=2))
        package.writestr("faults.csv", fault_records_csv(records))
        package.writestr("metadata.json", json.dumps(metadata, indent=2))
    return buffer.getvalue()


def random_fault_records(location, view, fault_types, fault_count, fault_scale, seed):
    rng = random.Random(seed)
    if isinstance(fault_types, str):
        fault_types = [fault_types]
    fault_types = [fault_type for fault_type in fault_types if fault_type in FAULT_TYPES]
    if not fault_types:
        return []
    shuffled_types = list(fault_types)
    rng.shuffle(shuffled_types)
    candidates = []
    for row in panels(location["id"]):
        for cell in cells_for_row(location["id"], row["id"]):
            candidates.append((row, cell))
    rng.shuffle(candidates)
    selected = candidates[: min(fault_count, len(candidates))]
    return [
        fault_record(location, view, row, cell, shuffled_types[idx % len(shuffled_types)], fault_scale)
        for idx, (row, cell) in enumerate(selected)
    ]


def stored_fault_records():
    return st.session_state.get("random_fault_records", [])


def display_fault_records(manual_record, generated_records):
    """Keep the selected fault target visible, then append generated faults."""
    records = [manual_record]
    seen_targets = {
        (
            manual_record["location_id"],
            manual_record["row_id"],
            manual_record["target_cell_id"],
            manual_record["fault_type"],
        )
    }
    for record in generated_records:
        key = (
            record["location_id"],
            record["row_id"],
            record["target_cell_id"],
            record["fault_type"],
        )
        if key in seen_targets:
            continue
        records.append(record)
        seen_targets.add(key)
    return records


def records_for_view(records, view):
    current_view = view.upper()
    normalized = []
    for record in records:
        item = dict(record)
        item["view"] = current_view
        normalized.append(item)
    return normalized


def location_by_id(location_id):
    return next((item for item in LOCATIONS if item["id"] == location_id), LOCATIONS[0])


def terrain_layer(location_id):
    location = location_by_id(location_id)
    base = load_rgb(location["source"], location["crop"])

    if location.get("prebuilt"):
        terrain = base
    elif location_id == "scenario_1":
        original = load_rgb("scenario_1_og_pv.jpeg")
        blurred = original.filter(ImageFilter.GaussianBlur(location["blur"]))
        terrain = original.copy()
        terrain.paste(blurred, (0, 0), panel_mask())
    else:
        terrain = base.filter(ImageFilter.GaussianBlur(location["blur"]))
        terrain = terrain.filter(ImageFilter.MedianFilter(7))

    contrast, color, brightness = location["color"]
    terrain = ImageEnhance.Contrast(terrain).enhance(contrast)
    terrain = ImageEnhance.Color(terrain).enhance(color)
    terrain = ImageEnhance.Brightness(terrain).enhance(brightness)
    return terrain


def pv_cutout():
    original = load_rgb("scenario_1_og_pv.jpeg")
    cutout = Image.new("RGBA", BASE_SIZE, (0, 0, 0, 0))
    cutout.paste(original.convert("RGBA"), (0, 0), panel_mask())
    return cutout


def local_rect_mask(size, radius=1, feather=0.45):
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    radius = max(0, min(radius, width // 2, height // 2))
    if radius:
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    else:
        draw.rectangle((0, 0, width - 1, height - 1), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask


def source_panel_crop(source, source_row, left_ratio, right_ratio):
    sx1 = source_row["x1"] + 2
    sy1 = source_row["y1"] + 2
    sx2 = source_row["x2"] - 2
    sy2 = source_row["y2"] - 2
    sw = max(1, sx2 - sx1)
    left = int(sx1 + sw * left_ratio)
    right = int(sx1 + sw * right_ratio)
    if right <= left:
        right = min(sx2, left + 1)
    return source.crop((left, sy1, right, sy2))


def source_module_crop(source, source_row, module_number, module_count):
    sx1 = source_row["x1"] + 2
    sy1 = source_row["y1"] + 2
    sx2 = source_row["x2"] - 2
    sy2 = source_row["y2"] - 2
    sw = max(1, sx2 - sx1)
    source_index = (module_number - 1) % max(1, module_count)
    left = int(sx1 + sw * source_index / max(1, module_count))
    right = int(sx1 + sw * (source_index + 1) / max(1, module_count))
    pad = max(1, int((right - left) * 0.04))
    crop = source.crop((max(sx1, left + pad), sy1, min(sx2, right - pad), sy2)).convert("RGB")
    arr = np.array(crop)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] < 118) & (hsv[:, :, 1] > 18)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) > max(8, crop.width * crop.height * 0.05):
        x1 = clamp(int(xs.min()) - 1, 0, crop.width - 1)
        x2 = clamp(int(xs.max()) + 2, x1 + 1, crop.width)
        y1 = clamp(int(ys.min()) - 1, 0, crop.height - 1)
        y2 = clamp(int(ys.max()) + 2, y1 + 1, crop.height)
        crop = crop.crop((x1, y1, x2, y2))
    return crop


def clamp(value, low, high):
    return max(low, min(high, value))


def mean_luma(image):
    return ImageStat.Stat(ImageOps.grayscale(image)).mean[0]


def add_panel_grain(patch, seed, strength=0.035):
    rng = random.Random(seed)
    noise = Image.effect_noise(patch.size, rng.uniform(5.0, 9.0)).convert("L")
    noise = noise.point(lambda value: int(128 + (value - 128) * strength))
    return ImageChops.multiply(patch.convert("RGB"), noise.convert("RGB")).convert("RGBA")


def match_patch_lighting(patch, terrain, box):
    x1, y1, x2, y2 = box
    terrain_crop = terrain.crop((x1, y1, x2, y2)).convert("RGB")
    terrain_level = mean_luma(terrain_crop)
    patch_level = max(1, mean_luma(patch.convert("RGB")))
    # Let the real background lighting influence the pasted PVs without washing
    # out the original panel texture.
    brightness = clamp((terrain_level / patch_level) * 0.72 + 0.34, 0.76, 1.18)
    contrast = clamp(0.92 + (terrain_level / 255) * 0.18, 0.92, 1.08)
    patch = ImageEnhance.Brightness(patch).enhance(brightness)
    patch = ImageEnhance.Contrast(patch).enhance(contrast)
    return patch


def terrain_texture_angle(terrain_crop):
    gray = np.array(ImageOps.grayscale(terrain_crop), dtype=np.uint8)
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = 0.5 * math.atan2(float((2 * gx * gy).mean()), float((gx * gx - gy * gy).mean()) + 1e-6)
    return angle


def harmonize_patch_texture(rgb, terrain_crop, seed, edge_strength=0.11):
    """Borrow local terrain grain/illumination so pasted PVs sit in the scene."""
    width, height = rgb.size
    if width < 3 or height < 3:
        return rgb

    rng = np.random.default_rng(seed)
    panel = np.array(rgb.convert("RGB"), dtype=np.float32)
    terrain_arr = np.array(terrain_crop.resize((width, height), Image.Resampling.BICUBIC).convert("RGB"), dtype=np.float32)
    terrain_gray = cv2.cvtColor(terrain_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    low_sigma = max(2.0, min(width, height) * 0.22)
    low = cv2.GaussianBlur(terrain_gray, (0, 0), low_sigma)
    mid = terrain_gray - cv2.GaussianBlur(terrain_gray, (0, 0), max(0.8, min(width, height) * 0.05))
    low_norm = (low - low.mean()) / (low.std() + 1e-6)
    mid_norm = (mid - mid.mean()) / (mid.std() + 1e-6)

    modulation = 1.0 + low_norm[..., None] * 0.025 + mid_norm[..., None] * 0.018
    sensor_grain = rng.normal(0, 1.0, (height, width, 1)).astype(np.float32)
    sensor_grain = cv2.GaussianBlur(sensor_grain, (0, 0), 0.55)
    if sensor_grain.ndim == 2:
        sensor_grain = sensor_grain[..., None]
    modulation += sensor_grain * 0.006
    panel *= modulation

    # Subtle terrain-color spill on the cut edges prevents sticker-like borders.
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    edge = np.clip(1.0 - dist / max(2.0, min(width, height) * 0.18), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.85)[..., None] * edge_strength
    panel = panel * (1.0 - edge) + terrain_arr * edge

    # Directional low-amplitude streaks follow the local terrain orientation.
    angle = terrain_texture_angle(terrain_crop)
    axis = np.cos(angle) * xx + np.sin(angle) * yy
    streak = np.sin(axis / max(3.0, min(width, height) * 0.42) + seed % 17)
    panel += streak[..., None] * 1.25

    return Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8), "RGB")


def mute_bright_panel_lines(rgb):
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright = gray > np.percentile(gray, 92)
    target = np.array([36, 46, 58], dtype=np.float32)
    arr[bright] = arr[bright] * 0.82 + target * 0.18
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def procedural_panel_module(size, seed):
    width, height = size
    scale = 4
    hi_w = max(32, int(width * scale))
    hi_h = max(24, int(height * scale))
    rng = np.random.default_rng(seed)

    module = np.zeros((hi_h, hi_w, 3), dtype=np.float32)
    module[:] = np.array([22, 35, 55], dtype=np.float32)

    frame = max(3, int(min(hi_w, hi_h) * 0.035))
    cv2.rectangle(module, (0, 0), (hi_w - 1, hi_h - 1), (132, 142, 146), frame)
    cv2.rectangle(module, (frame, frame), (hi_w - frame - 1, hi_h - frame - 1), (17, 30, 50), -1)

    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    gap = max(2, int(min(hi_w, hi_h) * 0.018))
    inner_x1 = frame + gap
    inner_y1 = frame + gap
    inner_x2 = hi_w - frame - gap
    inner_y2 = hi_h - frame - gap
    inner_w = max(1, inner_x2 - inner_x1)
    inner_h = max(1, inner_y2 - inner_y1)

    yy, xx = np.mgrid[0:hi_h, 0:hi_w]
    glass_gradient = 1.08 - 0.16 * (yy / max(1, hi_h - 1)) + 0.04 * (xx / max(1, hi_w - 1))

    for row in range(rows):
        for col in range(cols):
            x1 = int(inner_x1 + col * inner_w / cols) + gap // 2
            y1 = int(inner_y1 + row * inner_h / rows) + gap // 2
            x2 = int(inner_x1 + (col + 1) * inner_w / cols) - gap // 2
            y2 = int(inner_y1 + (row + 1) * inner_h / rows) - gap // 2
            if x2 <= x1 or y2 <= y1:
                continue
            cell_tint = np.array(
                [
                    18 + rng.integers(-4, 5),
                    38 + rng.integers(-5, 7),
                    70 + rng.integers(-7, 9),
                ],
                dtype=np.float32,
            )
            cv2.rectangle(module, (x1, y1), (x2, y2), tuple(int(v) for v in cell_tint), -1)
            bus_alpha = 0.62
            for lx in np.linspace(x1 + (x2 - x1) * 0.22, x2 - (x2 - x1) * 0.22, 3):
                cv2.line(module, (int(lx), y1 + 1), (int(lx), y2 - 1), (156, 172, 184), max(1, hi_w // 360))
            if y2 - y1 > 12:
                cv2.line(module, (x1 + 1, (y1 + y2) // 2), (x2 - 1, (y1 + y2) // 2), (118, 136, 154), 1)
            # soften busbars back into the silicon so they do not become UI lines
            module[y1:y2, x1:x2] = module[y1:y2, x1:x2] * (1 - bus_alpha * 0.08) + cell_tint * (bus_alpha * 0.08)

    # Real glass texture: mild sensor grain plus diagonal reflected light.
    noise = rng.normal(0, 3.2, (hi_h, hi_w, 1)).astype(np.float32)
    diagonal = np.sin((xx + yy * 0.34 + seed % 41) / max(8, hi_w * 0.18))[..., None] * 3.5
    module = module * glass_gradient[..., None] + noise + diagonal

    glare = np.zeros((hi_h, hi_w, 1), dtype=np.float32)
    glare_y = int(hi_h * (0.22 + 0.18 * ((seed % 17) / 16)))
    cv2.line(glare, (0, glare_y), (hi_w - 1, max(0, glare_y - hi_h // 6)), 1.0, max(2, hi_h // 14))
    glare = cv2.GaussianBlur(glare, (0, 0), max(1.0, hi_h * 0.035))
    if glare.ndim == 2:
        glare = glare[..., None]
    module += glare * 18

    module = np.clip(module, 0, 255).astype(np.uint8)
    panel = Image.fromarray(module, "RGB").resize((width, height), Image.Resampling.LANCZOS)
    return ImageEnhance.Sharpness(panel).enhance(1.35)


def color_match_array_to_background(source_rgb, target_rgb):
    if source_rgb.size == 0 or target_rgb.size == 0:
        return source_rgb
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = cv2.meanStdDev(src_lab)
    bg_mean, bg_std = cv2.meanStdDev(bg_lab)
    matched = src_lab.copy()
    src_l_mean = float(src_mean[0][0])
    src_l_std = float(src_std[0][0]) + 1e-5
    bg_l_mean = float(bg_mean[0][0])
    bg_l_std = float(bg_std[0][0])
    target_l_mean = clamp(bg_l_mean * 0.54, 42, 108)
    target_l_std = clamp(bg_l_std * 0.72, 12, 34)
    matched_l = (src_lab[:, :, 0] - src_l_mean) * (target_l_std / src_l_std) + target_l_mean
    matched[:, :, 0] = matched_l * 0.76 + src_lab[:, :, 0] * 0.24
    # Keep silicon blue/black; borrow only a tiny ambient color cast.
    matched[:, :, 1] = src_lab[:, :, 1] * 0.92 + float(bg_mean[1][0]) * 0.08
    matched[:, :, 2] = src_lab[:, :, 2] * 0.92 + float(bg_mean[2][0]) * 0.08
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def procedural_panel_row(size, module_count, seed, location_id=None):
    width, height = size
    scale = 4
    layouts, hi_w, hi_h = module_layouts_for_dimensions(width, height, module_count, scale=scale, high_res=True)
    rng = np.random.default_rng(seed)
    textures = real_panel_textures(texture_group_for_location(location_id)) if location_id else []
    row = np.zeros((hi_h, hi_w, 3), dtype=np.float32)
    row[:] = np.array([16, 28, 48], dtype=np.float32)

    yy, xx = np.mgrid[0:hi_h, 0:hi_w]
    global_glass = 1.05 - 0.13 * (yy / max(1, hi_h - 1)) + 0.035 * (xx / max(1, hi_w - 1))

    for layout in layouts:
        mx1, _, mx2, _ = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        if mx2 - mx1 < 6:
            continue
        frame = max(1, ix1 - mx1)
        cv2.rectangle(row, (mx1, 0), (mx2, hi_h - 1), (82, 91, 94), frame)
        glass_tint = np.array(
            [
                13 + rng.integers(-2, 3),
                29 + rng.integers(-3, 4),
                52 + rng.integers(-4, 5),
            ],
            dtype=np.float32,
        )
        if textures:
            texture = textures[int(rng.integers(0, len(textures)))]
            # Use a slightly different crop from the real patch each time so
            # repeated modules do not read as copy-pasted clones.
            th, tw = texture.shape[:2]
            crop_pad_x = max(0, tw // 12)
            crop_pad_y = max(0, th // 12)
            tx1 = int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0
            ty1 = int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0
            tx2 = tw - (int(rng.integers(0, crop_pad_x + 1)) if crop_pad_x else 0)
            ty2 = th - (int(rng.integers(0, crop_pad_y + 1)) if crop_pad_y else 0)
            texture_crop = texture[ty1:max(ty1 + 1, ty2), tx1:max(tx1 + 1, tx2)]
            textured = cv2.resize(texture_crop, (max(1, ix2 - ix1 + 1), max(1, iy2 - iy1 + 1)), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
            tint = np.full_like(textured, glass_tint, dtype=np.float32)
            textured = textured * 0.82 + tint * 0.18
            texture_noise = rng.normal(0, 1.1, textured.shape).astype(np.float32)
            row[iy1 : iy2 + 1, ix1 : ix2 + 1] = np.clip(textured + texture_noise, 0, 255)
        else:
            cv2.rectangle(row, (ix1, iy1), (ix2, iy2), tuple(int(v) for v in glass_tint), -1)

        # Render the real 6x2 PV cell structure, but as low-contrast glass
        # seams. The bright selectable overlay remains only in the zoom view.
        cell_cols = CUSTOM_MODULE_CELL_COLS
        cell_rows = CUSTOM_MODULE_CELL_ROWS
        seam_width = max(1, hi_w // 1100)
        for cell_row in range(cell_rows):
            for cell_col in range(cell_cols):
                cx1 = int(ix1 + cell_col * (ix2 - ix1) / cell_cols) + 1
                cx2 = int(ix1 + (cell_col + 1) * (ix2 - ix1) / cell_cols) - 1
                cy1 = int(iy1 + cell_row * (iy2 - iy1) / cell_rows) + 1
                cy2 = int(iy1 + (cell_row + 1) * (iy2 - iy1) / cell_rows) - 1
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                if not textures:
                    cell_tint = glass_tint + np.array(
                        [
                            rng.integers(-1, 2),
                            rng.integers(-2, 3),
                            rng.integers(-3, 4),
                        ],
                        dtype=np.float32,
                    )
                    cv2.rectangle(row, (cx1, cy1), (cx2, cy2), tuple(int(v) for v in np.clip(cell_tint, 0, 255)), -1)

        for idx in range(1, cell_cols):
            x = int(ix1 + idx * (ix2 - ix1) / cell_cols)
            cv2.line(row, (x, iy1 + 1), (x, iy2 - 1), (72, 96, 110), seam_width, cv2.LINE_AA)
        for idx in range(1, cell_rows):
            y = int(iy1 + idx * (iy2 - iy1) / cell_rows)
            cv2.line(row, (ix1 + 1, y), (ix2 - 1, y), (66, 88, 102), seam_width, cv2.LINE_AA)

    noise = rng.normal(0, 1.7, (hi_h, hi_w, 1)).astype(np.float32)
    row = row * global_glass[..., None] + noise
    row = np.clip(row, 0, 255).astype(np.uint8)
    row = cv2.resize(row, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return row


def redraw_visible_module_geometry(panel, module_count):
    """Restore the true 6x2 module cell seams after ambient color matching."""
    height, width = panel.shape[:2]
    layouts, _, _ = module_layouts_for_dimensions(width, height, module_count)
    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        ix1, iy1, ix2, iy2 = layout["inner"]
        cv2.rectangle(panel, (ox1, oy1), (ox2, oy2), (100, 108, 108), 1, cv2.LINE_AA)
        cv2.line(panel, (ox1 + 1, oy1 + 1), (ox2 - 1, oy1 + 1), (132, 140, 138), 1, cv2.LINE_AA)
        cv2.line(panel, (ox1 + 1, oy2 - 1), (ox2 - 1, oy2 - 1), (34, 44, 48), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (ix1, iy1), (ix2, iy2), (44, 58, 70), 1, cv2.LINE_AA)
        for idx in range(1, CUSTOM_MODULE_CELL_COLS):
            x = int(ix1 + idx * (ix2 - ix1) / CUSTOM_MODULE_CELL_COLS)
            cv2.line(panel, (x, iy1), (x, iy2), (62, 86, 100), 1, cv2.LINE_AA)
        for idx in range(1, CUSTOM_MODULE_CELL_ROWS):
            y = int(iy1 + idx * (iy2 - iy1) / CUSTOM_MODULE_CELL_ROWS)
            cv2.line(panel, (ix1, y), (ix2, y), (58, 78, 92), 1, cv2.LINE_AA)
    return panel


def apply_mounting_hardware(output_rgb, row, module_count, seed, location_id=None):
    """Draw location-appropriate racking without turning the scene into a diagram."""
    if location_id in {"farm_lake", "rooftop"}:
        return output_rgb
    layouts = module_layouts_for_row(row, module_count)
    if not layouts:
        return output_rgb

    h, w = output_rgb.shape[:2]
    row_h = max(1, row["y2"] - row["y1"])
    rng = np.random.default_rng(seed)
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    metal_mask = np.zeros((h, w), dtype=np.uint8)
    light_mask = np.zeros((h, w), dtype=np.uint8)

    left = min(layout["outer"][0] for layout in layouts)
    right = max(layout["outer"][2] for layout in layouts)
    top = min(layout["outer"][1] for layout in layouts)
    bottom = max(layout["outer"][3] for layout in layouts)
    rail_pad = max(2, int(row_h * 0.18))
    rail_width = 1
    support_w = max(1, int(row_h * 0.045))
    foot_w = max(2, int(row_h * 0.14))
    foot_h = 1

    rail_ys = [
        int(top + row_h * 0.27),
        int(top + row_h * 0.73),
    ]
    x_start = clamp(left - rail_pad, 0, w - 1)
    x_end = clamp(right + rail_pad, x_start + 1, w - 1)

    style = {
        "scenario_1": "ground",
        "agri_rows": "ground",
        "desert_farm": "ground",
        "farm_lake": "floating",
        "rooftop": "roof",
    }.get(location_id, "ground")

    for y in rail_ys:
        y = clamp(y, 0, h - 1)
        shadow_shift = max(1, row_h // 9)
        cv2.line(shadow_mask, (x_start + 1, y + shadow_shift), (x_end + 1, y + shadow_shift), 80, 1, cv2.LINE_AA)
        if style in {"roof", "floating"}:
            cv2.line(metal_mask, (x_start, y), (x_end, y), 95, rail_width, cv2.LINE_AA)
            cv2.line(light_mask, (x_start, max(0, y - 1)), (x_end, max(0, y - 1)), 70, 1, cv2.LINE_AA)

    boundaries = [layouts[0]["outer"][0]] + [layout["outer"][2] for layout in layouts]
    support_xs = []
    for idx, sx in enumerate(boundaries):
        if idx in (0, len(boundaries) - 1) or idx % 3 == 0:
            support_xs.append(sx)

    if style == "roof":
        for sx in support_xs:
            sx += int(rng.integers(-1, 2))
            pad_top = clamp(bottom + max(1, row_h // 16), 0, h - 1)
            pad_bottom = clamp(pad_top + foot_h, pad_top + 1, h - 1)
            fx1 = clamp(sx - foot_w, 0, w - 1)
            fx2 = clamp(sx + foot_w, fx1 + 1, w - 1)
            cv2.rectangle(shadow_mask, (fx1 + 1, pad_top + 1), (fx2 + 1, pad_bottom + 1), 65, -1)
            cv2.rectangle(metal_mask, (fx1, pad_top), (fx2, pad_bottom), 105, -1)
    elif style == "floating":
        for sx in support_xs:
            sx += int(rng.integers(-1, 2))
            marker_y = clamp(bottom + max(1, row_h // 18), 0, h - 1)
            cv2.circle(shadow_mask, (sx + 1, marker_y + 1), max(1, support_w + 1), 55, -1)
            cv2.circle(metal_mask, (sx, marker_y), max(1, support_w), 85, -1)
    else:
        # From nadir, ground-mount racks mostly read as contact shadows and
        # small dark post marks, not large visible legs.
        for sx in support_xs:
            marker_y = clamp(bottom + max(1, row_h // 20), 0, h - 1)
            cv2.circle(shadow_mask, (sx + 1, marker_y + 1), max(1, support_w + 1), 52, -1)

    shadow = cv2.GaussianBlur(shadow_mask, (0, 0), max(0.8, row_h * 0.13)).astype(np.float32) / 255.0
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - shadow[..., None] * 0.16), 0, 255).astype(np.uint8)

    metal = np.zeros((h, w, 3), dtype=np.uint8)
    metal[:, :] = (92, 100, 96)
    highlight = np.zeros_like(metal)
    highlight[:, :] = (150, 156, 150)
    metal_alpha = (metal_mask.astype(np.float32) / 255.0 * (0.28 if style != "roof" else 0.34))[..., None]
    light_alpha = (light_mask.astype(np.float32) / 255.0 * 0.18)[..., None]
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - metal_alpha) + metal.astype(np.float32) * metal_alpha, 0, 255).astype(np.uint8)
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - light_alpha) + highlight.astype(np.float32) * light_alpha, 0, 255).astype(np.uint8)
    return output_rgb


def composite_panel_row(output_rgb, row, module_count, seed, location_id=None):
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    x1, y1 = clamp(x1, 0, BASE_SIZE[0] - 1), clamp(y1, 0, BASE_SIZE[1] - 1)
    x2, y2 = clamp(x2, x1 + 1, BASE_SIZE[0]), clamp(y2, y1 + 1, BASE_SIZE[1])
    row_w, row_h = x2 - x1, y2 - y1
    terrain_crop = output_rgb[y1:y2, x1:x2].copy()
    panel = procedural_panel_row((row_w, row_h), module_count, seed, location_id)
    panel = color_match_array_to_background(panel, terrain_crop)
    panel = cv2.convertScaleAbs(panel, alpha=1.08, beta=1)
    panel = redraw_visible_module_geometry(panel, module_count)
    layouts = module_layouts_for_row(row, module_count)
    output_rgb = apply_mounting_hardware(output_rgb, row, module_count, seed_for(seed, "mounting-hardware"), location_id)

    # Ground contact shadow: offset slightly down/right, blurred, and applied
    # before compositing the glass surface.
    shadow = np.zeros(output_rgb.shape[:2], dtype=np.uint8)
    offset = max(1, int(row_h * 0.13))
    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        sx1 = clamp(ox1 + offset, 0, BASE_SIZE[0] - 1)
        sy1 = clamp(oy1 + max(1, offset // 2), 0, BASE_SIZE[1] - 1)
        sx2 = clamp(ox2 + offset, sx1 + 1, BASE_SIZE[0])
        sy2 = clamp(oy2 + offset, sy1 + 1, BASE_SIZE[1])
        cv2.rectangle(shadow, (sx1, sy1), (sx2, sy2), 180, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.6, row_h * 0.22)).astype(np.float32) / 255.0
    output_rgb[:] = np.clip(output_rgb.astype(np.float32) * (1.0 - shadow[..., None] * 0.18), 0, 255).astype(np.uint8)

    mask = np.zeros(output_rgb.shape[:2], dtype=np.uint8)
    for layout in layouts:
        ox1, oy1, ox2, oy2 = layout["outer"]
        cv2.rectangle(mask, (ox1, oy1), (ox2, oy2), 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    feather = max(2.0, row_h * 0.12)
    alpha = np.clip(dist / feather, 0, 1).astype(np.float32)
    edge_noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(1.0, 0.025, alpha.shape).astype(np.float32), (0, 0), 0.7)
    alpha = np.clip(alpha * edge_noise, 0, 1)
    local_alpha = alpha[y1:y2, x1:x2][..., None]
    blended = terrain_crop.astype(np.float32) * (1.0 - local_alpha) + panel.astype(np.float32) * local_alpha
    output_rgb[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return output_rgb


def prepare_agri_installation_ground(output_rgb, rows, seed):
    """Grade soft mounting corridors into the agri field before panels are drawn."""
    h, w = output_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    source = output_rgb.astype(np.float32)
    smoothed = cv2.GaussianBlur(source, (0, 0), 7.5)
    hsv = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2HSV)
    green_keep = ((hsv[:, :, 0] > 32) & (hsv[:, :, 0] < 96) & (hsv[:, :, 1] > 48)).astype(np.float32)
    green_keep = cv2.GaussianBlur(green_keep, (0, 0), 5.5)

    corridor = np.zeros((h, w), dtype=np.float32)
    service_tracks = np.zeros((h, w), dtype=np.float32)
    tram_tracks = np.zeros((h, w), dtype=np.float32)
    sorted_rows = sorted(rows, key=lambda item: item["y1"])
    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        pad_y = max(11, int(row_h * 1.05))
        pad_x = max(24, int(row_h * 1.6))
        x1 = clamp(row["x1"] - pad_x, 0, w - 1)
        x2 = clamp(row["x2"] + pad_x, x1 + 1, w)
        y1 = clamp(row["y1"] - pad_y, 0, h - 1)
        y2 = clamp(row["y2"] + pad_y, y1 + 1, h)
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(local, (x1, y1), (x2, y2), 255, -1)
        local = cv2.GaussianBlur(local, (0, 0), max(6.5, row_h * 0.42)).astype(np.float32) / 255.0
        rough_edge = cv2.GaussianBlur(rng.normal(1.0, 0.16, (h, w)).astype(np.float32), (0, 0), 5.0)
        corridor = np.maximum(corridor, np.clip(local * rough_edge, 0, 1))
        for ty in (row["y1"] - int(row_h * 0.42), row["y2"] + int(row_h * 0.42)):
            if 0 <= ty < h:
                track = np.zeros((h, w), dtype=np.uint8)
                cv2.line(track, (x1, ty), (x2, ty), 255, max(2, row_h // 5), cv2.LINE_AA)
                track = cv2.GaussianBlur(track, (0, 0), max(2.0, row_h * 0.18)).astype(np.float32) / 255.0
                service_tracks = np.maximum(service_tracks, np.clip(track * rough_edge, 0, 1))

    # Tractor tram lines: paired wheel marks in the working alleys between PV rows.
    x_min = max(10, min(row["x1"] for row in rows) - 18)
    x_max = min(w - 10, max(row["x2"] for row in rows) + 18)
    gaps = []
    for upper, lower in zip(sorted_rows, sorted_rows[1:]):
        gy1 = upper["y2"]
        gy2 = lower["y1"]
        if gy2 - gy1 >= 8:
            gaps.append((gy1, gy2))
    for gy1, gy2 in gaps:
        center_y = (gy1 + gy2) // 2
        lane_half = max(3, min(9, (gy2 - gy1) // 5))
        for offset in (-5, 5):
            y = clamp(center_y + offset, 0, h - 1)
            cv2.line(tram_tracks, (x_min, y), (x_max, y), 1.0, max(2, lane_half // 2), cv2.LINE_AA)
    tram_tracks = cv2.GaussianBlur(tram_tracks, (0, 0), 1.4)

    # Keep the lower green crop identity, while still allowing row beds to read.
    corridor *= 1.0 - np.clip(green_keep * 0.22, 0, 0.22)
    soil_tint = np.array([164, 136, 86], dtype=np.float32)
    soil_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.7)[..., None]
    graded = smoothed * 0.30 + soil_tint * 0.70 + soil_noise * np.array([8, 7, 5], dtype=np.float32)

    # Fine texture from the original prevents the pads from becoming fake flat boxes.
    high = np.clip(source - cv2.GaussianBlur(source, (0, 0), 9), -10, 10)
    graded = graded + high * 0.30
    mixed = source * (1 - corridor[..., None] * 0.92) + graded * (corridor[..., None] * 0.92)
    track_tint = np.array([112, 96, 68], dtype=np.float32)
    track_strength = service_tracks * (1 - np.clip(green_keep * 0.28, 0, 0.28))
    mixed = mixed * (1 - track_strength[..., None] * 0.48) + track_tint * (track_strength[..., None] * 0.48)
    tram_tint = np.array([84, 102, 55], dtype=np.float32)
    mixed = mixed * (1 - tram_tracks[..., None] * 0.34) + tram_tint * (tram_tracks[..., None] * 0.34)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def draw_site_railings(image_rgb, rows, location_id, seed):
    """Add subtle perimeter railings/fence posts around PV fields."""
    # Real nadir/aerial PV imagery rarely shows clean cartoon perimeter rails.
    # Site integration should come from row beds, service tracks, and contact
    # shadows, not an outlined rectangle around the array.
    return image_rgb
    if not rows or location_id in {"farm_lake", "rooftop"}:
        return image_rgb
    h, w = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    out = image_rgb.copy()
    x1 = max(6, min(row["x1"] for row in rows) - (32 if location_id in {"scenario_1", "agri_rows"} else 22))
    x2 = min(w - 7, max(row["x2"] for row in rows) + (32 if location_id in {"scenario_1", "agri_rows"} else 22))
    y1 = max(6, min(row["y1"] for row in rows) - (24 if location_id in {"scenario_1", "agri_rows"} else 18))
    y2 = min(h - 7, max(row["y2"] for row in rows) + (24 if location_id in {"scenario_1", "agri_rows"} else 18))

    rail_mask = np.zeros((h, w), dtype=np.uint8)
    post_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(rail_mask, (x1, y1), (x2, y2), 165, 2 if location_id in {"scenario_1", "desert_farm"} else 1)
    post_step = 26 if location_id in {"scenario_1", "agri_rows"} else 34
    for x in range(x1, x2 + 1, post_step):
        jitter = int(rng.integers(-1, 2))
        cv2.rectangle(post_mask, (clamp(x + jitter - 1, 0, w - 1), y1 - 1), (clamp(x + jitter + 1, 0, w - 1), y1 + 2), 160, -1)
        cv2.rectangle(post_mask, (clamp(x + jitter - 1, 0, w - 1), y2 - 2), (clamp(x + jitter + 1, 0, w - 1), y2 + 1), 145, -1)
    for y in range(y1, y2 + 1, post_step):
        jitter = int(rng.integers(-1, 2))
        cv2.rectangle(post_mask, (x1 - 1, clamp(y + jitter - 1, 0, h - 1)), (x1 + 2, clamp(y + jitter + 1, 0, h - 1)), 150, -1)
        cv2.rectangle(post_mask, (x2 - 2, clamp(y + jitter - 1, 0, h - 1)), (x2 + 1, clamp(y + jitter + 1, 0, h - 1)), 140, -1)

    shadow = cv2.GaussianBlur(np.maximum(rail_mask, post_mask), (0, 0), 0.7).astype(np.float32) / 255.0
    out = np.clip(out.astype(np.float32) * (1.0 - shadow[..., None] * 0.12), 0, 255).astype(np.uint8)
    rail_color = np.array([58, 68, 56] if location_id != "desert_farm" else [68, 62, 50], dtype=np.float32)
    metal_alpha = ((rail_mask.astype(np.float32) / 255.0) * 0.50 + (post_mask.astype(np.float32) / 255.0) * 0.62)[..., None]
    out = np.clip(out.astype(np.float32) * (1 - metal_alpha) + rail_color * metal_alpha, 0, 255).astype(np.uint8)
    return out


def draw_mounting_rails(image_rgb, rows, location_id, seed):
    """Draw mounting rails and small support posts before the PV modules are composited."""
    if not rows:
        return image_rgb
    h, w = image_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    rail_mask = np.zeros((h, w), dtype=np.float32)
    post_mask = np.zeros((h, w), dtype=np.float32)
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    heavy_mounts = location_id in {"farm_lake", "rooftop"}

    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        x_extra = max(8 if heavy_mounts else 5, row_h // (2 if heavy_mounts else 3))
        x1 = clamp(row["x1"] - x_extra, 0, w - 1)
        x2 = clamp(row["x2"] + x_extra, x1 + 1, w - 1)
        rail_width = 2 if heavy_mounts else 1
        upper_y = clamp(row["y1"] - max(3 if heavy_mounts else 2, row_h // 8), 0, h - 1)
        lower_y = clamp(row["y2"] + max(3 if heavy_mounts else 2, row_h // 8), 0, h - 1)
        cv2.line(rail_mask, (x1, upper_y), (x2, upper_y), 1.0, rail_width, cv2.LINE_AA)
        cv2.line(rail_mask, (x1, lower_y), (x2, lower_y), 1.0, rail_width, cv2.LINE_AA)
        if heavy_mounts:
            mid_y = clamp((row["y1"] + row["y2"]) // 2, 0, h - 1)
            cv2.line(rail_mask, (x1, mid_y), (x2, mid_y), 0.72, 1, cv2.LINE_AA)

        module_count = max(1, custom_panels_per_row(location_id))
        step = max(18 if heavy_mounts else 24, int((row["x2"] - row["x1"]) / max(1, module_count)))
        for px in range(row["x1"] + step // 2, row["x2"], step):
            jitter = int(rng.integers(-2, 3))
            px = clamp(px + jitter, 0, w - 1)
            post_w = 2 if heavy_mounts else 1
            post_h = max(5 if heavy_mounts else 4, int(row_h * (0.42 if heavy_mounts else 0.28)))
            cv2.rectangle(post_mask, (px - post_w, lower_y), (px + post_w, clamp(lower_y + post_h, 0, h - 1)), 1.0, -1)
            cv2.rectangle(shadow_mask, (px - post_w + 2, lower_y + 2), (px + post_w + 5, clamp(lower_y + post_h + 4, 0, h - 1)), 1.0, -1)
        cv2.line(shadow_mask, (x1 + 2, lower_y + 2), (x2 + 3, lower_y + 2), 1.0, max(1, rail_width), cv2.LINE_AA)

    shadow = cv2.GaussianBlur(shadow_mask, (0, 0), 1.0 if heavy_mounts else 1.35)[..., None]
    rail = cv2.GaussianBlur(rail_mask, (0, 0), 0.45 if heavy_mounts else 0.55)[..., None]
    posts = cv2.GaussianBlur(post_mask, (0, 0), 0.55 if heavy_mounts else 0.70)[..., None]
    if location_id == "desert_farm":
        rail_color = np.array([116, 103, 78], dtype=np.float32)
        post_color = np.array([100, 92, 72], dtype=np.float32)
    elif location_id == "farm_lake":
        rail_color = np.array([94, 124, 130], dtype=np.float32)
        post_color = np.array([70, 94, 100], dtype=np.float32)
    elif location_id == "rooftop":
        rail_color = np.array([116, 122, 122], dtype=np.float32)
        post_color = np.array([98, 102, 102], dtype=np.float32)
    else:
        rail_color = np.array([78, 88, 76], dtype=np.float32)
        post_color = np.array([68, 78, 70], dtype=np.float32)
    out = image_rgb.astype(np.float32)
    shadow_strength = 0.18 if heavy_mounts else 0.14
    rail_strength = 0.42 if heavy_mounts else 0.32
    post_strength = 0.46 if heavy_mounts else 0.34
    out *= 1.0 - shadow * shadow_strength
    out = out * (1.0 - rail * rail_strength) + rail_color * (rail * rail_strength)
    out = out * (1.0 - posts * post_strength) + post_color * (posts * post_strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare_floating_installation_environment(output_rgb, rows, seed):
    """Add visible pontoons, mooring rails, and service walkways for the offshore scene."""
    return output_rgb


def prepare_rooftop_installation_environment(output_rgb, rows, seed):
    """Add roof ballast pads and stronger metal rails under the rooftop array."""
    return output_rgb


def prepare_installation_environment(output_rgb, rows, location_id, seed):
    """Add site context behind custom PV rows: beds, service roads, and a small control pad."""
    if not rows:
        return output_rgb
    if location_id == "farm_lake":
        return prepare_floating_installation_environment(output_rgb, rows, seed)
    if location_id == "rooftop":
        return prepare_rooftop_installation_environment(output_rgb, rows, seed)
    if location_id == "agri_rows":
        prepared = prepare_agri_installation_ground(output_rgb, rows, seed)
        prepared = draw_mounting_rails(prepared, rows, location_id, seed_for(seed, "mounts"))
        return draw_site_railings(prepared, rows, location_id, seed_for(seed, "railings"))

    h, w = output_rgb.shape[:2]
    rng = np.random.default_rng(seed)
    source = output_rgb.astype(np.float32)
    mixed = source.copy()
    bed_mask = np.zeros((h, w), dtype=np.float32)
    road_mask = np.zeros((h, w), dtype=np.float32)
    asphalt_mask = np.zeros((h, w), dtype=np.float32)
    gravel_mask = np.zeros((h, w), dtype=np.float32)

    x1_all = max(0, min(row["x1"] for row in rows) - 18)
    x2_all = min(w - 1, max(row["x2"] for row in rows) + 18)
    y1_all = max(0, min(row["y1"] for row in rows) - 18)
    y2_all = min(h - 1, max(row["y2"] for row in rows) + 18)
    sorted_rows = sorted(rows, key=lambda item: item["y1"])
    row_gaps = []
    for upper, lower in zip(sorted_rows, sorted_rows[1:]):
        gy1 = upper["y2"]
        gy2 = lower["y1"]
        if gy2 - gy1 >= 8:
            row_gaps.append((gy1, gy2))

    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        pad_y = max(5, int(row_h * 0.85))
        pad_x = max(14, int(row_h * 1.15))
        bx1 = clamp(row["x1"] - pad_x, 0, w - 1)
        bx2 = clamp(row["x2"] + pad_x, bx1 + 1, w)
        by1 = clamp(row["y1"] - pad_y, 0, h - 1)
        by2 = clamp(row["y2"] + pad_y, by1 + 1, h)
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(local, (bx1, by1), (bx2, by2), 255, -1)
        local = cv2.GaussianBlur(local, (0, 0), max(2.2, row_h * 0.20)).astype(np.float32) / 255.0
        bed_mask = np.maximum(bed_mask, local)

    # Service tracks between rows. Avoid full perimeter boxes; in nadir imagery
    # those read as drawn outlines unless they already exist in the source photo.
    road_width = 7 if location_id == "scenario_1" else (6 if location_id == "desert_farm" else 5)
    for row in rows:
        row_h = max(1, row["y2"] - row["y1"])
        y = clamp(row["y2"] + max(4, int(row_h * 0.8)), 0, h - 1)
        cv2.line(road_mask, (x1_all, y), (x2_all, y), 1.0, max(2, road_width // 2), cv2.LINE_AA)
        if location_id == "scenario_1":
            y_top = clamp(row["y1"] - max(4, int(row_h * 0.75)), 0, h - 1)
            cv2.line(road_mask, (x1_all, y_top), (x2_all, y_top), 0.72, max(2, road_width // 3), cv2.LINE_AA)
    if location_id in {"scenario_1", "desert_farm"}:
        access_x = clamp(x1_all - 18, 0, w - 1)
        cv2.line(road_mask, (access_x, y1_all), (access_x, y2_all), 0.55, max(2, road_width // 2), cv2.LINE_AA)
        mid_y = clamp((y1_all + y2_all) // 2, 0, h - 1)
        cv2.line(road_mask, (access_x, mid_y), (x1_all, mid_y), 0.45, max(1, road_width // 3), cv2.LINE_AA)
    if location_id == "scenario_1":
        for gy1, gy2 in row_gaps:
            lane_y = clamp((gy1 + gy2) // 2, 0, h - 1)
            lane_w = max(7, min(15, int((gy2 - gy1) * 0.36)))
            cv2.line(gravel_mask, (x1_all, lane_y), (x2_all, lane_y), 1.0, lane_w, cv2.LINE_AA)
    if location_id == "desert_farm":
        for gy1, gy2 in row_gaps:
            lane_y = clamp((gy1 + gy2) // 2, 0, h - 1)
            lane_w = max(8, min(16, int((gy2 - gy1) * 0.34)))
            cv2.line(asphalt_mask, (x1_all, lane_y), (x2_all, lane_y), 1.0, lane_w, cv2.LINE_AA)
    road_mask = cv2.GaussianBlur(road_mask, (0, 0), 2.0)
    asphalt_mask = cv2.GaussianBlur(asphalt_mask, (0, 0), 1.2)
    gravel_mask = cv2.GaussianBlur(gravel_mask, (0, 0), 1.4)

    if location_id == "scenario_1":
        bed_tint = np.array([112, 116, 80], dtype=np.float32)
        road_tint = np.array([136, 124, 88], dtype=np.float32)
        control_tint = np.array([160, 164, 150], dtype=np.float32)
        bed_alpha = 0.62
        road_alpha = 0.62
        bed_source = 0.56
    elif location_id == "desert_farm":
        bed_tint = np.array([184, 163, 116], dtype=np.float32)
        road_tint = np.array([154, 132, 88], dtype=np.float32)
        control_tint = np.array([178, 170, 148], dtype=np.float32)
        bed_alpha = 0.46
        road_alpha = 0.58
        bed_source = 0.70
    else:
        bed_tint = np.array([117, 132, 82], dtype=np.float32)
        road_tint = np.array([132, 123, 92], dtype=np.float32)
        control_tint = np.array([178, 180, 165], dtype=np.float32)
        bed_alpha = 0.50
        road_alpha = 0.72
        bed_source = 0.58

    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.4)[..., None]
    beds = source * bed_source + bed_tint * (1.0 - bed_source) + noise * np.array([7, 6, 4], dtype=np.float32)
    mixed = mixed * (1 - bed_mask[..., None] * bed_alpha) + beds * (bed_mask[..., None] * bed_alpha)
    roads = source * 0.38 + road_tint * 0.62 + noise * np.array([5, 5, 4], dtype=np.float32)
    mixed = mixed * (1 - road_mask[..., None] * road_alpha) + roads * (road_mask[..., None] * road_alpha)
    if location_id == "scenario_1":
        gravel_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.45)[..., None]
        gravel = source * 0.28 + np.array([158, 150, 126], dtype=np.float32) * 0.72 + gravel_noise * np.array([18, 17, 14], dtype=np.float32)
        mixed = mixed * (1 - gravel_mask[..., None] * 0.78) + gravel * (gravel_mask[..., None] * 0.78)
    if location_id == "desert_farm":
        asphalt_noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.65)[..., None]
        asphalt = source * 0.22 + np.array([76, 74, 68], dtype=np.float32) * 0.78 + asphalt_noise * np.array([10, 10, 9], dtype=np.float32)
        mixed = mixed * (1 - asphalt_mask[..., None] * 0.82) + asphalt * (asphalt_mask[..., None] * 0.82)

    # Control / maintenance pad: small, off to the side, connected by access road.
    pad_w = 34
    pad_h = 23
    if x1_all > 70:
        px1 = max(8, x1_all - pad_w - 18)
    else:
        px1 = min(w - pad_w - 8, x2_all + 18)
    py1 = clamp(y1_all + 5, 8, h - pad_h - 8)
    px2 = px1 + pad_w
    py2 = py1 + pad_h
    pad_mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(pad_mask, (px1, py1), (px2, py2), 1.0, -1)
    cv2.rectangle(pad_mask, (px1 - 4, py1 - 4), (px2 + 4, py2 + 4), 0.45, 1)
    pad_mask = cv2.GaussianBlur(pad_mask, (0, 0), 0.6)
    pad = source * 0.30 + control_tint * 0.70
    mixed = mixed * (1 - pad_mask[..., None] * 0.82) + pad * (pad_mask[..., None] * 0.82)

    # Simple roof/ac unit marks on the control pad.
    rect_color = (72, 78, 78) if location_id == "desert_farm" else (64, 76, 66)
    mixed_u8 = np.clip(mixed, 0, 255).astype(np.uint8)
    cv2.rectangle(mixed_u8, (px1 + 5, py1 + 5), (px2 - 5, py2 - 5), rect_color, 1)
    cv2.rectangle(mixed_u8, (px1 + 9, py1 + 8), (px1 + 17, py1 + 15), rect_color, -1)
    cv2.line(mixed_u8, (px2, py1 + pad_h // 2), (x1_all, py1 + pad_h // 2), tuple(int(v) for v in road_tint), 2, cv2.LINE_AA)
    mixed_u8 = draw_mounting_rails(mixed_u8, rows, location_id, seed_for(seed, "mounts"))
    mixed_u8 = draw_site_railings(mixed_u8, rows, location_id, seed_for(seed, "railings"))
    return mixed_u8


def textured_alpha_mask(size, seed, feather=1.0):
    width, height = size
    mask = local_rect_mask(size, radius=0, feather=feather)
    if width < 4 or height < 4:
        return mask
    rng = random.Random(seed)
    edge_noise = Image.effect_noise(size, rng.uniform(8.0, 14.0)).convert("L")
    edge_noise = edge_noise.point(lambda value: int(232 + (value - 128) * 0.08))
    mask = ImageChops.multiply(mask, edge_noise)
    return mask.filter(ImageFilter.GaussianBlur(0.25))


def draw_realistic_module_patch(size, seed, terrain, box, module_cells, source_patch=None):
    width, height = size
    rng = random.Random(seed)
    width = max(4, int(width))
    height = max(4, int(height))
    terrain_crop = terrain.crop(box).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    terrain_level = mean_luma(terrain_crop)

    base_r = rng.randint(18, 34)
    base_g = rng.randint(42, 62)
    base_b = rng.randint(70, 104)
    if source_patch is not None:
        real_rgb = procedural_panel_module((width, height), seed)
        if source_patch.size[0] >= 12 and source_patch.size[1] >= 6:
            source_texture = source_patch.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
            source_texture = ImageEnhance.Contrast(mute_bright_panel_lines(source_texture)).enhance(1.08)
            real_rgb = Image.blend(real_rgb, source_texture, 0.18)
        real_rgb = match_patch_lighting(real_rgb.convert("RGBA"), terrain, box).convert("RGB")
        tint = Image.new("RGB", (width, height), (base_r, base_g, base_b))
        real_rgb = Image.blend(real_rgb, tint, 0.08)
        module = real_rgb.convert("RGBA")
    else:
        module = Image.new("RGBA", (width, height), (base_r, base_g, base_b, 255))
    draw = ImageDraw.Draw(module)

    frame = max(1, min(width, height) // 13)
    inner = (frame, frame, width - frame - 1, height - frame - 1)
    frame_light = (142, 156, 164, 18)
    frame_dark = (15, 24, 31, 38)
    if source_patch is None:
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=max(1, frame), fill=frame_dark)
        draw.rectangle(inner, fill=(base_r, base_g, base_b, 255))
    else:
        pass

    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    inner_w = max(1, inner[2] - inner[0] + 1)
    inner_h = max(1, inner[3] - inner[1] + 1)
    gap = max(1, min(width, height) // 34)

    for cell in module_cells:
        cx = cell["module_cell_col"] - 1
        cy = cell["module_cell_row"] - 1
        x1 = int(inner[0] + cx * inner_w / cols) + gap
        y1 = int(inner[1] + cy * inner_h / rows) + gap
        x2 = int(inner[0] + (cx + 1) * inner_w / cols) - gap
        y2 = int(inner[1] + (cy + 1) * inner_h / rows) - gap
        if x2 <= x1 or y2 <= y1:
            continue
        radius = max(0, min(x2 - x1, y2 - y1) // 12)
        if source_patch is None:
            tint = rng.randint(-8, 9)
            cell_color = (
                clamp(base_r + tint, 6, 55),
                clamp(base_g + tint + rng.randint(-3, 4), 25, 82),
                clamp(base_b + tint + rng.randint(-5, 8), 52, 126),
                255,
            )
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=cell_color)
        else:
            # Preserve the real panel photograph texture; cell IDs are shown in
            # the zoom overlay, so the scene itself should not look annotated.
            pass
        # Subtle busbar lines inside each selectable cell.
        if source_patch is None and y2 - y1 >= 6:
            by = y1 + (y2 - y1) // 2
            draw.line((x1 + 1, by, x2 - 1, by), fill=(116, 132, 148, 6), width=1)
        if source_patch is None and x2 - x1 >= 8:
            bx = x1 + (x2 - x1) // 2
            draw.line((bx, y1 + 1, bx, y2 - 1), fill=(118, 132, 145, 5), width=1)

    # Real module seam/frame lines, stronger than cell busbars but not UI-white.
    seam = (185, 198, 202, 44 if source_patch is not None else 80)
    frame_seam = (205, 214, 216, 72 if source_patch is not None else 120)
    draw.rectangle((1, 1, width - 2, height - 2), outline=frame_seam, width=1)
    for idx in range(1, cols):
        x = int(inner[0] + idx * inner_w / cols)
        draw.line((x, inner[1], x, inner[3]), fill=seam, width=1)
    for idx in range(1, rows):
        y = int(inner[1] + idx * inner_h / rows)
        draw.line((inner[0], y, inner[2], y), fill=seam, width=1)

    # Glass glare and real imaging grain.
    glare = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glare_draw = ImageDraw.Draw(glare)
    glare_y = rng.randint(max(0, height // 8), max(1, height // 2))
    glare_draw.polygon(
        [
            (0, glare_y),
            (width, max(0, glare_y - height // 5)),
            (width, max(0, glare_y - height // 5 + max(2, height // 7))),
            (0, glare_y + max(2, height // 7)),
        ],
        fill=(255, 255, 255, rng.randint(0, 4) if source_patch is not None else rng.randint(12, 28)),
    )
    module.alpha_composite(glare.filter(ImageFilter.GaussianBlur(max(0.5, height * 0.018))))

    noise = Image.effect_noise((width, height), rng.uniform(6.5, 11.0)).convert("L")
    noise = noise.point(lambda value: int(128 + (value - 128) * (0.018 if source_patch is not None else 0.025)))
    rgb = ImageChops.multiply(module.convert("RGB"), noise.convert("RGB"))
    if source_patch is not None:
        rgb = ImageEnhance.Brightness(rgb).enhance(1.12)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
        rgb = ImageEnhance.Color(rgb).enhance(1.02)
    brightness = clamp((terrain_level / 150) * 0.22 + 1.00, 0.96, 1.26)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(0.98 if source_patch is not None else 1.04)
    if source_patch is not None:
        rgb = harmonize_patch_texture(
            rgb,
            terrain_crop,
            seed_for(seed, "terrain-harmonize"),
            edge_strength=0.04,
        )
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.35)

    alpha = (
        local_rect_mask((width, height), radius=0, feather=0.35)
        if source_patch is not None
        else textured_alpha_mask((width, height), seed_for(seed, "alpha-texture"), feather=0.45)
    )
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def draw_module_detail(draw, box, seed, selected_units, cell=None):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    rng = random.Random(seed)

    minor = (226, 236, 244, 56)
    major = (232, 240, 246, 132)
    shadow = (18, 32, 55, 46)

    # Real PV-cell boundary: faint on every cell so the fault can be checked by
    # counting cells, with a heavier divider at module edges.
    draw.rounded_rectangle((x1, y1, x2, y2), radius=max(1, min(width, height) // 7), outline=minor, width=1)
    if height >= 8:
        bus1 = int(y1 + height * 0.36)
        bus2 = int(y1 + height * 0.67)
        draw.line((x1 + 1, bus1, x2 - 1, bus1), fill=(220, 230, 244, 36), width=1)
        draw.line((x1 + 1, bus2, x2 - 1, bus2), fill=(220, 230, 244, 30), width=1)

    if cell and cell.get("module_cell_col") == 1:
        draw.line((x1, y1, x1, y2), fill=major, width=2)
    if cell and cell.get("module_cell_col") == CUSTOM_MODULE_CELL_COLS:
        draw.line((x2, y1, x2, y2), fill=shadow, width=2)
    if cell and cell.get("module_cell_row") == 1:
        draw.line((x1, y1, x2, y1), fill=major, width=1)
    if cell and cell.get("module_cell_row") == CUSTOM_MODULE_CELL_ROWS:
        draw.line((x1, y2, x2, y2), fill=shadow, width=1)


def custom_panel_scene(location_id):
    """Build real visible PV rows/cells for the current custom layout."""
    location = location_by_id(location_id)
    terrain = terrain_layer(location_id).convert("RGBA")
    if location_id == "desert_farm":
        source = load_rgb("desert_user_reference_with_pvs.png").convert("RGB")
        source_rows = DESERT_USER_ROW_DEFS
    elif location.get("pv_source"):
        source = load_rgb(location["pv_source"]).convert("RGB")
        source_rows = base_row_defs_for_location(location_id)
    else:
        source = load_rgb("scenario_1_og_pv.jpeg").convert("RGB")
        source_rows = ROW_DEFS
    output_rgb = np.array(terrain.convert("RGB"), dtype=np.uint8)
    module_count = custom_panels_per_row(location_id)
    rows = panels(location_id)
    output_rgb = prepare_installation_environment(
        output_rgb,
        rows,
        location_id,
        seed_for(location_id, module_count, "installation-environment"),
    )
    for row in rows:
        output_rgb = composite_panel_row(
            output_rgb,
            row,
            module_count,
            seed_for(location_id, row["id"], module_count, "row-level-composite"),
            location_id,
        )

    return Image.fromarray(output_rgb)


def composite_with_og_pvs(location_id):
    location = location_by_id(location_id)
    if custom_layout_enabled():
        return custom_panel_scene(location_id)
    if location.get("pv_source"):
        return load_rgb(location["pv_source"])

    image = terrain_layer(location_id).convert("RGBA")
    image.alpha_composite(pv_cutout())
    return image.convert("RGB")


def thermal_palette(value):
    stops = [
        (0, (18, 7, 32)),
        (42, (45, 14, 73)),
        (82, (88, 24, 101)),
        (118, (141, 39, 91)),
        (150, (190, 62, 61)),
        (184, (225, 105, 35)),
        (220, (246, 168, 48)),
        (255, (255, 225, 128)),
    ]
    for idx in range(len(stops) - 1):
        left_v, left_c = stops[idx]
        right_v, right_c = stops[idx + 1]
        if value <= right_v:
            ratio = 0 if right_v == left_v else (value - left_v) / (right_v - left_v)
            return tuple(int(left_c[channel] + (right_c[channel] - left_c[channel]) * ratio) for channel in range(3))
    return stops[-1][1]


@st.cache_data(show_spinner=False)
def thermal_lut():
    return [thermal_palette(value) for value in range(256)]


def temperature_to_rgb(temp):
    palette = []
    for color in thermal_lut():
        palette.extend(color)
    pal_img = temp.convert("P")
    pal_img.putpalette(palette)
    return pal_img.convert("RGB")


def thermal_terrain_settings(location_id):
    settings = {
        "scenario_1": (146, 42),
        "agri_rows": (150, 44),
        "desert_farm": (108, 40),
        "farm_lake": (118, 34),
        "rooftop": (150, 38),
    }
    return settings.get(location_id, (146, 40))


def thermal_overlay_alpha(location_id):
    settings = {
        "desert_farm": 0.60,
        "scenario_1": 0.66,
        "agri_rows": 0.62,
        "farm_lake": 0.58,
        "rooftop": 0.66,
    }
    return settings.get(location_id, 0.64)


def thermal_noise(shape, rng, amplitude, sigma):
    noise = rng.normal(0, amplitude, size=shape).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), sigma)


def blend_temp_patch(temp, box, patch, alpha):
    x1, y1, x2, y2 = box
    h, w = temp.shape
    x1, x2 = clamp(x1, 0, w), clamp(x2, 0, w)
    y1, y2 = clamp(y1, 0, h), clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return
    local = patch[: y2 - y1, : x2 - x1]
    temp[y1:y2, x1:x2] = temp[y1:y2, x1:x2] * (1 - alpha) + local * alpha


def apply_thermal_fault(temp, cell, fault_type, scale):
    if cell is None:
        return temp
    x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = 0 if w < 9 else max(1, int(w * 0.06))
    pad_y = 0 if h < 9 else max(1, int(h * 0.08))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    bw, bh = max(3, bx2 - bx1), max(3, by2 - by1)
    severity = clamp(scale, 1, 10) / 10
    rng = np.random.default_rng(seed_for(cell["id"], fault_type, scale, "thermal-field"))

    if fault_type == "Hotspot":
        mask = np.array(irregular_mask((bw, bh), seed_for(cell["id"], "hotspot-mask"), blobs=5), dtype=np.float32) / 255.0
        halo = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(bw, bh) * 0.20))
        core = cv2.GaussianBlur((mask > 0.62).astype(np.float32), (0, 0), max(0.7, min(bw, bh) * 0.06))
        region = temp[by1:by2, bx1:bx2]
        target = 206 + halo[:bh, :bw] * (22 + 34 * severity) + core[:bh, :bw] * (34 + 44 * severity)
        temp[by1:by2, bx1:bx2] = np.maximum(region, target)

    elif fault_type == "Bypass diode":
        band_h = max(2, int(bh * (0.22 + 0.20 * severity)))
        cy = by1 + bh // 2
        band = np.zeros((bh, bw), dtype=np.float32)
        cv2.rectangle(band, (0, max(0, cy - by1 - band_h // 2)), (bw - 1, min(bh - 1, cy - by1 + band_h // 2)), 1.0, -1)
        band = cv2.GaussianBlur(band, (0, 0), max(0.7, band_h * 0.18))
        temp[by1:by2, bx1:bx2] += band[:bh, :bw] * (44 + 54 * severity)

    elif fault_type in {"Soiling / dust", "Bird droppings", "Snow cover"}:
        mask = np.array(irregular_mask((bw, bh), seed_for(cell["id"], "dust-mask"), blobs=9), dtype=np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.8, min(bw, bh) * 0.05))
        if fault_type == "Bird droppings":
            cool_delta = 38 + 30 * severity
            warm_rim = 18 + 18 * severity
        elif fault_type == "Snow cover":
            cool_delta = 54 + 42 * severity
            warm_rim = 6 + 8 * severity
        else:
            cool_delta = 22 + 26 * severity
            warm_rim = 8 + 12 * severity
        temp[by1:by2, bx1:bx2] -= mask[:bh, :bw] * cool_delta
        edge = cv2.Laplacian(mask, cv2.CV_32F)
        temp[by1:by2, bx1:bx2] += np.clip(edge, 0, 1)[:bh, :bw] * warm_rim

    elif fault_type == "Partial shading":
        shade = np.zeros((bh, bw), dtype=np.float32)
        pts = np.array(
            [
                [0, int(bh * 0.16)],
                [bw - 1, int(bh * 0.34 + rng.integers(-1, 2))],
                [bw - 1, min(bh - 1, int(bh * 0.92))],
                [0, min(bh - 1, int(bh * 0.68))],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(shade, [pts], 1.0)
        shade = cv2.GaussianBlur(shade, (0, 0), max(0.8, bh * 0.08))
        temp[by1:by2, bx1:bx2] -= shade[:bh, :bw] * (30 + 32 * severity)

    elif fault_type == "Electrical / open circuit":
        fault = np.zeros((bh, bw), dtype=np.float32)
        cv2.line(fault, (bw // 2, 0), (bw // 2, bh - 1), 1.0, max(1, int(bw * 0.22)))
        cv2.line(fault, (0, bh // 2), (bw - 1, bh // 2), 0.55, max(1, int(bh * 0.16)))
        fault = cv2.GaussianBlur(fault, (0, 0), max(0.65, min(bw, bh) * 0.10))
        temp[by1:by2, bx1:bx2] += fault[:bh, :bw] * (48 + 52 * severity)
        temp[by1:by2, bx1:bx2] -= (1 - fault[:bh, :bw]) * (8 + 10 * severity)

    elif fault_type == "Degradation / resistance":
        gradient = np.linspace(0.15, 1.0, bw, dtype=np.float32)[None, :]
        waviness = rng.normal(0, 0.18, size=(bh, bw)).astype(np.float32)
        waviness = cv2.GaussianBlur(waviness, (0, 0), max(0.7, min(bw, bh) * 0.12))
        field = np.clip(gradient + waviness, 0, 1)
        temp[by1:by2, bx1:bx2] += field[:bh, :bw] * (18 + 28 * severity)

    else:
        crack = np.zeros((bh, bw), dtype=np.float32)
        points = []
        for idx in range(7):
            t = idx / 6
            px = int(bw * t)
            py = int(bh * (0.22 + 0.58 * t) + rng.integers(-max(1, bh // 5), max(2, bh // 5)))
            points.append((px, clamp(py, 0, bh - 1)))
        cv2.polylines(crack, [np.array(points, dtype=np.int32)], False, 1.0, max(1, int(min(bw, bh) * 0.10)))
        crack = cv2.GaussianBlur(crack, (0, 0), max(0.5, min(bw, bh) * 0.035))
        temp[by1:by2, bx1:bx2] -= crack[:bh, :bw] * (34 + 24 * severity)
        for px, py in points[1:-1:2]:
            r = max(1, int(min(bw, bh) * (0.10 + 0.04 * severity)))
            cv2.circle(temp[by1:by2, bx1:bx2], (px, py), r, 245 + 10 * severity, -1)

    return temp


def thermalize(image, location_id, include_pv, fault_cell=None, fault_type=None, fault_scale=DEFAULT_FAULT_SCALE, fault_records=None):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    frame = np.array(rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Thermal-camera style pipeline: build a scalar temperature field first,
    # then colorize it. This avoids the fake "red RGB filter" look.
    soft = cv2.GaussianBlur(gray, (0, 0), 2.4)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(soft)
    terrain_low, terrain_high = {
        "desert_farm": (18, 112),
        "rooftop": (46, 170),
        "agri_rows": (44, 174),
        "farm_lake": (24, 138),
        "scenario_1": (44, 172),
    }.get(location_id, (44, 172))
    temp = cv2.normalize(clahe, None, terrain_low, terrain_high, cv2.NORM_MINMAX).astype(np.float32)

    broad = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 10)
    detail = gray.astype(np.float32) - broad
    temp += detail * 0.12

    rng = np.random.default_rng(seed_for(location_id, "opencv-thermal", include_pv))
    temp += thermal_noise(temp.shape, rng, 2.7, 0.70)
    temp += thermal_noise(temp.shape, rng, 8.5, 5.8)
    temp += thermal_noise(temp.shape, rng, 4.0, 16.0)

    if include_pv:
        shadow_mask = np.zeros(temp.shape, dtype=np.uint8)
        for row in panels(location_id):
            row_h = max(1, row["y2"] - row["y1"])
            offset = max(2, int(row_h * 0.18))
            if custom_layout_enabled():
                module_count = custom_panels_per_row(location_id)
                for layout in module_layouts_for_row(row, module_count):
                    ox1, oy1, ox2, oy2 = layout["outer"]
                    cv2.rectangle(
                        shadow_mask,
                        (ox1 + offset, oy2 - max(1, row_h // 12)),
                        (min(BASE_SIZE[0] - 1, ox2 + offset), min(BASE_SIZE[1] - 1, oy2 + offset)),
                        180,
                        -1,
                    )
            else:
                cv2.rectangle(
                    shadow_mask,
                    (row["x1"] + offset, row["y2"] - max(1, row_h // 12)),
                    (min(BASE_SIZE[0] - 1, row["x2"] + offset), min(BASE_SIZE[1] - 1, row["y2"] + offset)),
                    180,
                    -1,
                )
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), 1.8).astype(np.float32) / 255.0
        shadow_strength = 9 if location_id == "desert_farm" else 34
        temp -= shadow_mask * shadow_strength

        for row in panels(location_id):
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            row_w = max(1, x2 - x1)
            row_h = max(1, y2 - y1)
            row_gray = gray[y1:y2, x1:x2]
            if row_gray.size == 0:
                continue
            panel_local = cv2.GaussianBlur(row_gray, (0, 0), 0.55)
            panel_min, panel_max = {
                "desert_farm": (184, 248),
                "rooftop": (156, 228),
                "agri_rows": (160, 230),
                "farm_lake": (150, 220),
                "scenario_1": (158, 226),
            }.get(location_id, (158, 226))
            panel_local = cv2.normalize(panel_local, None, panel_min, panel_max, cv2.NORM_MINMAX).astype(np.float32)
            panel_local += thermal_noise(panel_local.shape, rng, 1.8, 0.45)
            panel_local += thermal_noise(panel_local.shape, rng, 4.5, 3.2)

            grad_x = np.linspace(-4, 4, row_w, dtype=np.float32)[None, :]
            grad_y = np.linspace(2, -2, row_h, dtype=np.float32)[:, None]
            panel_local += grad_x + grad_y

            row_mask = np.zeros((row_h, row_w), dtype=np.float32)
            for cell in cells_for_row(location_id, row["id"]):
                lx1 = clamp(cell["x1"] - x1, 0, row_w - 1)
                ly1 = clamp(cell["y1"] - y1, 0, row_h - 1)
                lx2 = clamp(cell["x2"] - x1, lx1 + 1, row_w)
                ly2 = clamp(cell["y2"] - y1, ly1 + 1, row_h)
                cv2.rectangle(row_mask, (lx1, ly1), (lx2, ly2), 0.88, -1)
            row_mask = cv2.GaussianBlur(row_mask, (0, 0), 0.42)
            temp[y1:y2, x1:x2] = temp[y1:y2, x1:x2] * (1.0 - row_mask) + panel_local * row_mask

            # Draw separators in temperature space, like real thermal camera
            # output where frames/gaps cool down rather than becoming UI lines.
            for cell in cells_for_row(location_id, row["id"]):
                edge_temp = panel_min - 48
                cv2.rectangle(temp, (cell["x1"], cell["y1"]), (cell["x2"], cell["y2"]), edge_temp, 1)
                if cell.get("module_cell_col") == 1:
                    cv2.line(temp, (cell["x1"], cell["y1"]), (cell["x1"], cell["y2"]), edge_temp - 18, 1)
                if cell.get("module_cell_row") == 1:
                    cv2.line(temp, (cell["x1"], cell["y1"]), (cell["x2"], cell["y1"]), edge_temp - 10, 1)
            if custom_layout_enabled():
                module_count = custom_panels_per_row(location_id)
                for layout in module_layouts_for_row(row, module_count):
                    ox1, oy1, ox2, oy2 = layout["outer"]
                    cv2.rectangle(temp, (ox1, oy1), (ox2, oy2), panel_min - 56, 1)
            else:
                cv2.line(temp, (x1, y1), (x2, y1), panel_min - 54, 1)
                cv2.line(temp, (x1, y2), (x2, y2), panel_min - 62, max(1, row_h // 16))

        if fault_records:
            for record in fault_records:
                row = row_by_id(location_id, record["row_id"])
                cell = cell_by_id(location_id, row["id"], record["target_cell_id"])
                temp = apply_thermal_fault(temp, cell, record["fault_type"], record.get("fault_scale", fault_scale))
        elif fault_cell is not None and fault_type:
            temp = apply_thermal_fault(temp, fault_cell, fault_type, fault_scale)

    temp = np.clip(cv2.GaussianBlur(temp, (0, 0), 0.32), 0, 255).astype(np.uint8)
    thermal_bgr = cv2.applyColorMap(temp, cv2.COLORMAP_PLASMA)
    thermal_rgb = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2RGB)

    # Preserve a little sensor structure without letting RGB leak dominate the
    # thermal palette.
    structure = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2RGB)
    thermal_rgb = cv2.addWeighted(thermal_rgb, 0.975, structure, 0.025, 0)
    return Image.fromarray(thermal_rgb)


def seed_for(*parts):
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def irregular_mask(size, seed, blobs=7):
    width, height = size
    rng = random.Random(seed)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(blobs):
        cx = rng.randint(max(1, width // 5), max(2, width * 4 // 5))
        cy = rng.randint(max(1, height // 5), max(2, height * 4 // 5))
        rx = rng.randint(max(2, width // 8), max(3, width // 3))
        ry = rng.randint(max(2, height // 8), max(3, height // 2))
        fill = rng.randint(110, 230)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)
    return ImageOps.autocontrast(mask.filter(ImageFilter.GaussianBlur(max(1.0, min(width, height) * 0.06))))


def paste_color_with_mask(image, color, box, mask, alpha=255):
    patch = Image.new("RGBA", mask.size, (*color, alpha))
    patch.putalpha(mask.point(lambda value: int(value * alpha / 255)))
    image.alpha_composite(patch, box[:2])


def draw_realistic_fault(image, cell, fault_type, scale, view):
    image = image.convert("RGBA")
    x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    pad_x = max(2, int(w * 0.08))
    pad_y = max(1, int(h * 0.12))
    bx1, by1, bx2, by2 = x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y
    bw, bh = max(4, bx2 - bx1), max(4, by2 - by1)
    rng = random.Random(seed_for(cell["id"], fault_type, scale, view))
    severity = scale / 10
    overlay = Image.new("RGBA", BASE_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if fault_type == "Hotspot":
        mask = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=4 + scale // 2)
        hot_mask = mask.filter(ImageFilter.GaussianBlur(max(1.2, min(bw, bh) * 0.10)))
        core_mask = mask.point(lambda value: 255 if value > 150 else 0).filter(ImageFilter.GaussianBlur(1.0))
        if view == "thermal":
            paste_color_with_mask(overlay, (232, 48, 26), (bx1, by1, bx2, by2), hot_mask, int(150 + 70 * severity))
            paste_color_with_mask(overlay, (255, 244, 150), (bx1, by1, bx2, by2), core_mask, int(130 + 90 * severity))
        else:
            paste_color_with_mask(overlay, (115, 54, 28), (bx1, by1, bx2, by2), hot_mask, int(55 + 45 * severity))

    elif fault_type == "Bypass diode":
        band_count = 2 if scale >= 7 else 1
        for band in range(band_count):
            offset = int((band - (band_count - 1) / 2) * bh * 0.34)
            cy = int((by1 + by2) / 2 + offset)
            band_h = max(3, int(bh * (0.28 + severity * 0.18)))
            jitter = rng.randint(-max(1, bw // 20), max(1, bw // 20))
            box = (bx1 + jitter, cy - band_h // 2, bx2 + jitter, cy + band_h // 2)
            if view == "thermal":
                draw.rounded_rectangle(box, radius=max(1, band_h // 4), fill=(240, 172, 45, int(125 + 70 * severity)))
                draw.line((box[0], cy, box[2], cy), fill=(255, 244, 150, int(110 + 80 * severity)), width=max(1, band_h // 3))
            else:
                draw.rounded_rectangle(box, radius=2, fill=(62, 71, 86, int(65 + 35 * severity)))

    elif fault_type in {"Soiling / dust", "Bird droppings", "Snow cover"}:
        mask = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=9)
        if view == "thermal":
            if fault_type == "Snow cover":
                color = (18, 26, 84)
                alpha = int(135 + 78 * severity)
            elif fault_type == "Bird droppings":
                color = (28, 24, 62)
                alpha = int(122 + 72 * severity)
            else:
                color = (22, 38, 98)
                alpha = int(105 + 70 * severity)
            paste_color_with_mask(overlay, color, (bx1, by1, bx2, by2), mask, alpha)
            edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.8))
            paste_color_with_mask(overlay, (79, 184, 141), (bx1, by1, bx2, by2), edge, 80)
        else:
            if fault_type == "Snow cover":
                paste_color_with_mask(overlay, (224, 232, 232), (bx1, by1, bx2, by2), mask, int(150 + 70 * severity))
                edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.6))
                paste_color_with_mask(overlay, (170, 188, 198), (bx1, by1, bx2, by2), edge, 100)
            elif fault_type == "Bird droppings":
                paste_color_with_mask(overlay, (210, 204, 178), (bx1, by1, bx2, by2), mask, int(145 + 60 * severity))
                speck = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=4).point(lambda value: 255 if value > 168 else 0)
                paste_color_with_mask(overlay, (70, 64, 48), (bx1, by1, bx2, by2), speck, int(75 + 40 * severity))
            else:
                paste_color_with_mask(overlay, (126, 102, 76), (bx1, by1, bx2, by2), mask, int(95 + 55 * severity))

    elif fault_type == "Partial shading":
        points = [
            (bx1 - int(bw * 0.2), by1 + int(bh * 0.15)),
            (bx2 + int(bw * 0.15), by1 + int(bh * 0.45)),
            (bx2 + int(bw * 0.15), by2 + int(bh * 0.15)),
            (bx1 - int(bw * 0.2), by2 - int(bh * 0.15)),
        ]
        fill = (18, 30, 80, int(120 + 65 * severity)) if view == "thermal" else (18, 24, 32, int(80 + 55 * severity))
        draw.polygon(points, fill=fill)
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.6, bh * 0.035)))

    elif fault_type in {"Electrical / open circuit", "Degradation / resistance"}:
        if fault_type == "Electrical / open circuit":
            band_alpha = int(105 + 68 * severity)
            if view == "thermal":
                draw.line((bx1 + bw // 2, by1, bx1 + bw // 2, by2), fill=(255, 238, 132, band_alpha), width=max(1, bw // 4))
                draw.line((bx1, by1 + bh // 2, bx2, by1 + bh // 2), fill=(232, 104, 40, int(band_alpha * 0.75)), width=max(1, bh // 5))
            else:
                draw.line((bx1 + bw // 2, by1, bx1 + bw // 2, by2), fill=(42, 48, 52, int(65 + 40 * severity)), width=max(1, bw // 5))
        else:
            mask = irregular_mask((bw, bh), rng.randint(0, 10_000), blobs=5)
            if view == "thermal":
                paste_color_with_mask(overlay, (248, 162, 48), (bx1, by1, bx2, by2), mask, int(80 + 70 * severity))
            else:
                paste_color_with_mask(overlay, (80, 72, 62), (bx1, by1, bx2, by2), mask, int(42 + 38 * severity))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.35, min(bw, bh) * 0.03)))

    else:
        points = []
        steps = 6
        for idx in range(steps + 1):
            t = idx / steps
            px = int(bx1 + bw * t)
            py = int(by1 + bh * (0.25 + 0.55 * t) + rng.randint(-max(1, bh // 5), max(1, bh // 5)))
            points.append((px, py))
        line_color = (12, 16, 34, int(145 + 70 * severity)) if view == "thermal" else (238, 242, 246, int(120 + 60 * severity))
        draw.line(points, fill=line_color, width=max(1, int(1 + severity * 3)))
        if view == "thermal":
            for px, py in points[1:-1:2]:
                r = max(1, int(min(bw, bh) * (0.07 + 0.04 * severity)))
                draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 230, 118, int(110 + 90 * severity)))

    image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(0.25)))
    return image.convert("RGB")


def scene_image(location_id, view, include_pv, selected_row_id, selected_cell_id, fault_type, fault_scale):
    rgb_image = composite_with_og_pvs(location_id) if include_pv else terrain_layer(location_id)
    row = row_by_id(location_id, selected_row_id)
    cell = cell_by_id(location_id, row["id"], selected_cell_id)
    if view == "thermal":
        return thermalize(rgb_image, location_id, include_pv, cell if include_pv else None, fault_type, fault_scale)
    image = rgb_image
    if include_pv:
        image = draw_realistic_fault(image, cell, fault_type, fault_scale, view)
    return image


def scene_image_with_faults(location_id, view, include_pv, selected_row_id, selected_cell_id, fault_type, fault_scale, fault_records):
    rgb_image = composite_with_og_pvs(location_id) if include_pv else terrain_layer(location_id)
    row = row_by_id(location_id, selected_row_id)
    cell = cell_by_id(location_id, row["id"], selected_cell_id)
    active_records = fault_records if include_pv else []
    if view == "thermal":
        if active_records:
            return thermalize(rgb_image, location_id, include_pv, fault_records=active_records)
        return thermalize(rgb_image, location_id, include_pv, cell if include_pv else None, fault_type, fault_scale)
    image = rgb_image
    if include_pv:
        if active_records:
            for record in active_records:
                record_row = row_by_id(location_id, record["row_id"])
                record_cell = cell_by_id(location_id, record_row["id"], record["target_cell_id"])
                image = draw_realistic_fault(image, record_cell, record["fault_type"], record.get("fault_scale", fault_scale), view)
        else:
            image = draw_realistic_fault(image, cell, fault_type, fault_scale, view)
    return image


def data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def draw_row_bounding_boxes(image, location_id, selected_row_id):
    boxed = image.convert("RGBA")
    overlay = Image.new("RGBA", boxed.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for panel in panels(location_id):
        selected = panel["id"] == selected_row_id
        outline = (250, 204, 21, 255) if selected else (56, 189, 248, 245)
        fill = (250, 204, 21, 42) if selected else (56, 189, 248, 30)
        width = 5 if selected else 4
        draw.rectangle(
            (panel["x1"], panel["y1"], panel["x2"], panel["y2"]),
            outline=outline,
            fill=fill,
            width=width,
        )
    return Image.alpha_composite(boxed, overlay).convert("RGB")


def crop_box_for_row(row):
    margin_x = max(16, int((row["x2"] - row["x1"]) * 0.06))
    if custom_layout_enabled():
        margin_y = max(18, int((row["y2"] - row["y1"]) * 0.55))
    else:
        margin_y = max(24, int((row["y2"] - row["y1"]) * 1.6))
    return (
        max(0, row["x1"] - margin_x),
        max(0, row["y1"] - margin_y),
        min(BASE_SIZE[0], row["x2"] + margin_x),
        min(BASE_SIZE[1], row["y2"] + margin_y),
    )


def grid_font(size):
    for font_name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_visible_cell_grid_on_crop(cropped, location_id, selected_row_id, selected_cell_id, crop, scale=1):
    image = cropped.convert("RGBA")
    draw = ImageDraw.Draw(image)
    crop_w, crop_h = cropped.size
    cells = cells_for_row(location_id, selected_row_id)
    font_size = max(6, min(12, int(crop_h * 0.11), int(crop_w / max(12, len(cells) / 2) * 0.55)))
    font = grid_font(font_size)
    module_boxes = {}
    selected_cell_box = None

    for cell in cells:
        rel_x1 = (cell["x1"] - crop[0]) * scale
        rel_y1 = (cell["y1"] - crop[1]) * scale
        rel_x2 = (cell["x2"] - crop[0]) * scale
        rel_y2 = (cell["y2"] - crop[1]) * scale
        selected = cell["id"] == selected_cell_id
        outline = (224, 242, 254, 92)
        draw.rectangle((rel_x1, rel_y1, rel_x2, rel_y2), outline=outline, width=1)

        if cell.get("module_number"):
            module_number = cell["module_number"]
            if module_number not in module_boxes:
                module_boxes[module_number] = [rel_x1, rel_y1, rel_x2, rel_y2]
            else:
                box = module_boxes[module_number]
                box[0] = min(box[0], rel_x1)
                box[1] = min(box[1], rel_y1)
                box[2] = max(box[2], rel_x2)
                box[3] = max(box[3], rel_y2)
        if selected:
            selected_cell_box = (rel_x1, rel_y1, rel_x2, rel_y2)

        label = str(cell.get("pv_cell_number_in_row", cell.get("panel_number", "")))
        should_label = selected or len(cells) <= 120
        if should_label and rel_x2 - rel_x1 >= 9 and rel_y2 - rel_y1 >= 8:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = rel_x1 + max(1, (rel_x2 - rel_x1 - text_w) / 2)
            ty = rel_y1 + max(1, (rel_y2 - rel_y1 - text_h) / 2)
            draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 210), font=font)
            draw.text((tx, ty), label, fill=(255, 255, 255, 240), font=font)

    if module_boxes:
        label_font = grid_font(max(8, min(13, int(crop_h * 0.12))))
        for module_number, (mx1, my1, mx2, my2) in module_boxes.items():
            draw.rectangle((mx1, my1, mx2, my2), outline=(255, 255, 255, 210), width=3)
            label = f"M{module_number}"
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
            label_x = mx1 + 2
            label_y = my1 - label_h - 7 if my1 - label_h - 7 >= 2 else my1 + 2
            draw.rounded_rectangle(
                (label_x, label_y, label_x + label_w + 6, label_y + label_h + 4),
                radius=2,
                fill=(4, 8, 18, 160),
            )
            draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255, 230), font=label_font)

    if selected_cell_box:
        draw.rectangle(selected_cell_box, outline=(250, 204, 21, 255), width=4)

    return image.convert("RGB")


def render_photo_map(image, location_id, view, selected_row_id, selected_cell_id, include_links, show_bounding_boxes=False):
    if include_links and show_bounding_boxes:
        image = draw_row_bounding_boxes(image, location_id, selected_row_id)
    uri = data_uri(image)
    svg_parts = [
        f'<svg width="{BASE_SIZE[0]}" height="{BASE_SIZE[1]}" viewBox="0 0 {BASE_SIZE[0]} {BASE_SIZE[1]}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{BASE_SIZE[0]}" height="{BASE_SIZE[1]}" preserveAspectRatio="xMidYMid slice"/>',
    ]
    if include_links:
        for panel in panels(location_id):
            selected = panel["id"] == selected_row_id
            stroke = "#facc15" if selected and show_bounding_boxes else "#38bdf8"
            opacity = "1.0" if show_bounding_boxes else "0.0"
            width = "4" if selected and show_bounding_boxes else "3" if show_bounding_boxes else "1"
            fill_opacity = "0.10" if selected and show_bounding_boxes else "0.07" if show_bounding_boxes else "0.001"
            hit_pad_y = max(8, int((panel["y2"] - panel["y1"]) * 0.9)) if custom_layout_enabled() else 0
            hit_y1 = max(0, panel["y1"] - hit_pad_y)
            hit_y2 = min(BASE_SIZE[1], panel["y2"] + hit_pad_y)
            default_cell = cells_for_row(location_id, panel["id"])[0]["id"]
            target = f"?{urlencode(navigation_params(location_id, view, panel['id'], default_cell))}"
            title = esc(panel["label"])
            svg_parts.append(
                f'<a href="{esc(target)}" target="_parent">'
                f'<title>{title}</title>'
                f'<rect x="{panel["x1"]}" y="{panel["y1"]}" width="{panel["x2"] - panel["x1"]}" '
                f'height="{panel["y2"] - panel["y1"]}" fill="#38bdf8" fill-opacity="{fill_opacity}" '
                f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                f'pointer-events="all"/></a>'
                f'<a href="{esc(target)}" target="_parent">'
                f'<title>{title}</title>'
                f'<rect x="{panel["x1"]}" y="{hit_y1}" width="{panel["x2"] - panel["x1"]}" '
                f'height="{hit_y2 - hit_y1}" fill="#000000" fill-opacity="0.001" '
                f'stroke="none" pointer-events="all"/></a>'
            )
    svg_parts.append("</svg>")
    components.html(
        f"""
        <div class="frame">{''.join(svg_parts)}</div>
        <style>
          .frame {{ width:100%; overflow:hidden; border:1px solid #263244; border-radius:10px; background:#050816; }}
          svg {{ width:100%; height:auto; display:block; }}
          a {{ cursor:pointer; }}
          a:hover rect {{ opacity:.55; stroke:#38bdf8; stroke-width:2; }}
        </style>
        """,
        height=430,
        scrolling=False,
    )


def render_zoom_map(image, location_id, view, selected_row_id, selected_cell_id):
    row = row_by_id(location_id, selected_row_id)
    crop = crop_box_for_row(row)
    cropped = image.crop(crop)
    cells = cells_for_row(location_id, selected_row_id)
    base_crop_w, base_crop_h = cropped.size
    zoom_scale = max(1, min(5, math.ceil(len(cells) * 13 / max(1, base_crop_w))))
    if zoom_scale > 1:
        cropped = cropped.resize((base_crop_w * zoom_scale, base_crop_h * zoom_scale), Image.Resampling.LANCZOS)
    cropped = draw_visible_cell_grid_on_crop(cropped, location_id, selected_row_id, selected_cell_id, crop, zoom_scale)
    uri = data_uri(cropped)
    crop_w, crop_h = cropped.size
    svg_parts = [
        f'<svg width="{crop_w}" height="{crop_h}" viewBox="0 0 {crop_w} {crop_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{crop_w}" height="{crop_h}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    for cell in cells:
        selected = cell["id"] == selected_cell_id
        stroke = "#facc15" if selected else "#e0f2fe"
        opacity = "0.98" if selected else "0.28"
        width = "3" if selected else "1"
        target = f"?{urlencode(navigation_params(location_id, view, selected_row_id, cell['id']))}"
        svg_parts.append(
            f'<a href="{esc(target)}" target="_parent">'
            f'<title>{esc(cell["label"])}</title>'
            f'<rect x="{(cell["x1"] - crop[0]) * zoom_scale}" y="{(cell["y1"] - crop[1]) * zoom_scale}" '
            f'width="{(cell["x2"] - cell["x1"]) * zoom_scale}" '
            f'height="{(cell["y2"] - cell["y1"]) * zoom_scale}" fill="#000000" fill-opacity="0.001" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" pointer-events="all"/></a>'
        )
    svg_parts.append("</svg>")
    height = min(560, max(220, crop_h + 36))
    components.html(
        f"""
        <div class="zoom-frame">{''.join(svg_parts)}</div>
        <style>
          .zoom-frame {{ width:100%; overflow:auto; border:1px solid #263244; border-radius:10px; background:#050816; }}
          svg {{ width:{crop_w}px; max-width:none; height:auto; display:block; }}
          a {{ cursor:pointer; }}
          a:hover rect {{ opacity:1; stroke:#38bdf8; stroke-width:3; }}
        </style>
        """,
        height=height,
        scrolling=False,
    )


def selected_location():
    return location_by_id(qp_get("location", LOCATIONS[0]["id"]))


def main():
    st.set_page_config(page_title="Local PV Farm Fault Tool", layout="wide")
    st.title("Local PV Farm Fault Tool")
    st.caption(
        "Real-photo compositor: default mode keeps the prepared PV layouts; custom mode rebuilds visible PV rows and masked cells on the terrain."
    )

    location = selected_location()
    view = qp_get("view", "rgb")
    if view not in ("rgb", "thermal"):
        view = "rgb"

    all_rows = panels(location["id"])
    selected_row_id = qp_get("selected_pv", all_rows[0]["id"])
    if selected_row_id not in {row["id"] for row in all_rows}:
        selected_row_id = all_rows[0]["id"]
    selected_row = row_by_id(location["id"], selected_row_id)
    row_cells = cells_for_row(location["id"], selected_row_id)
    selected_cell_id = qp_get("selected_cell", row_cells[0]["id"])
    valid_cell_ids = {cell["id"] for cell in row_cells}
    if selected_cell_id not in valid_cell_ids:
        legacy_custom_id = selected_cell_id.replace("-P", "-C") if custom_layout_enabled() else selected_cell_id
        selected_cell_id = legacy_custom_id if legacy_custom_id in valid_cell_ids else row_cells[0]["id"]
    selected_cell = cell_by_id(location["id"], selected_row_id, selected_cell_id)

    with st.sidebar:
        st.header("Scene")
        if st.button("Clear all faults", use_container_width=True):
            st.session_state["random_fault_records"] = []
            st.rerun()

        names = [item["name"] for item in LOCATIONS]
        current_index = [item["id"] for item in LOCATIONS].index(location["id"])
        chosen = st.selectbox("Real terrain / location", names, index=current_index)
        chosen_location = LOCATIONS[names.index(chosen)]
        if chosen_location["id"] != location["id"]:
            new_row = panels(chosen_location["id"])[0]["id"]
            qp_set(
                **navigation_params(
                    chosen_location["id"],
                    view,
                    new_row,
                    cells_for_row(chosen_location["id"], new_row)[0]["id"],
                )
            )
            st.rerun()

        chosen_view = st.radio("Camera view", ["RGB", "Thermal"], index=0 if view == "rgb" else 1, horizontal=True)
        chosen_view = chosen_view.lower()
        if chosen_view != view:
            qp_set(**navigation_params(location["id"], chosen_view, selected_row_id, selected_cell_id))
            st.rerun()

        st.caption(location["terrain"])
        st.caption("Use the tabs below for layout, target selection, random faults, and outputs.")

    scene_tab, generation_tab, output_tab, big_screen_tab = st.tabs(
        ["Terrain vs Panel", "Generation", "Output", "Big Screen Test"]
    )

    fault_scale = DEFAULT_FAULT_SCALE
    fault_type = "Hotspot"
    manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
    lock_manual_target = True

    with generation_tab:
        layout_col, target_col = st.columns([1, 1])
        with layout_col:
            st.subheader("Rows and Modules")
            current_custom_layout = custom_layout_enabled()
            custom_choice = st.checkbox("Custom row/module count", value=current_custom_layout)
            if custom_choice != current_custom_layout:
                first_row = f"{row_prefix(location['id'])}1"
                qp_set(
                    location=location["id"],
                    view=view,
                    custom_layout=1 if custom_choice else 0,
                    row_count=custom_row_count(location["id"]) if custom_choice else len(base_row_defs_for_location(location["id"])),
                    panels_per_row=custom_panels_per_row(location["id"]) if custom_choice else default_panels_per_row(location["id"]),
                    selected_pv=first_row,
                    selected_cell=f"{first_row}-C001" if custom_choice else f"{first_row}-P001",
                )
                st.rerun()

            row_control_value = (
                custom_row_count(location["id"])
                if current_custom_layout
                else min(len(base_row_defs_for_location(location["id"])), max_custom_rows(location["id"]))
            )
            panel_control_value = min(
                custom_panels_per_row(location["id"]) if current_custom_layout else default_panels_per_row(location["id"]),
                max_custom_panels_per_row(location["id"]),
            )
            desired_rows = st.number_input(
                "Rows",
                min_value=1,
                max_value=max_custom_rows(location["id"]),
                value=row_control_value,
                step=1,
                disabled=not current_custom_layout,
            )
            desired_panels = st.number_input(
                "Modules per row",
                min_value=1,
                max_value=max_custom_panels_per_row(location["id"]),
                value=panel_control_value,
                step=1,
                disabled=not current_custom_layout,
            )
            if current_custom_layout and (
                int(desired_rows) != custom_row_count(location["id"])
                or int(desired_panels) != custom_panels_per_row(location["id"])
            ):
                first_row = f"{row_prefix(location['id'])}1"
                qp_set(
                    location=location["id"],
                    view=view,
                    custom_layout=1,
                    row_count=int(desired_rows),
                    panels_per_row=int(desired_panels),
                    selected_pv=first_row,
                    selected_cell=f"{first_row}-C001",
                )
                st.rerun()
            cells_per_module = CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS
            st.caption(
                f"{int(desired_panels)} modules per row x {cells_per_module} selectable PV cells/module "
                f"= {int(desired_panels) * cells_per_module} selectable cells per row."
            )
            cap = LAYOUT_CAPS.get(location["id"])
            if cap:
                st.info(CAP_JUSTIFICATIONS[location["id"]])
                if int(desired_rows) >= cap["rows"] or int(desired_panels) >= cap["modules"]:
                    st.warning(
                        "At cap: row/module limits preserve solar-module aspect ratio, readable cell labels, "
                        "and prevent layouts from spilling into unusable image areas."
                    )
            annotation_loaded = load_annotation(location["id"]) is not None
            st.caption(f"Annotation boxes: {'loaded from local JSON' if annotation_loaded else 'generated boxes'}")
            show_bounding_boxes = st.checkbox("Show bounding boxes", value=False)

        with target_col:
            st.subheader("Fault Target")
            current_custom_layout = custom_layout_enabled()
            layout_key = (
                f"{location['id']}_{view}_{int(current_custom_layout)}_"
                f"{custom_row_count(location['id'])}_{custom_panels_per_row(location['id'])}"
            )
            row_options = [row["id"] for row in all_rows]
            row_choice = st.selectbox(
                "PV row",
                row_options,
                index=row_options.index(selected_row_id),
                format_func=lambda row_id: row_by_id(location["id"], row_id)["label"],
                key=f"fault_row_{layout_key}",
            )
            if row_choice != selected_row_id:
                qp_set(**navigation_params(location["id"], view, row_choice, cells_for_row(location["id"], row_choice)[0]["id"]))
                st.rerun()

            cell_options = [cell["id"] for cell in row_cells]
            cell_choice = st.selectbox(
                "PV cell in selected row",
                cell_options,
                index=cell_options.index(selected_cell_id),
                format_func=lambda cell_id: cell_by_id(location["id"], selected_row_id, cell_id)["label"],
                key=f"fault_cell_{layout_key}_{selected_row_id}",
            )
            if cell_choice != selected_cell_id:
                qp_set(**navigation_params(location["id"], view, selected_row_id, cell_choice))
                st.rerun()

            fault_type = st.selectbox("Fault type", list(FAULT_TYPES.keys()))
            manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
            lock_manual_target = st.checkbox("Lock manual target into generated fault set", value=True)
            st.write(FAULT_TYPES[fault_type])
            st.metric("Selected row", selected_row["label"])
            st.metric("Selected PV cell", selected_cell["id"])
            if selected_cell.get("module_number"):
                st.caption(
                    f"Module {selected_cell['module_number']} - cell {selected_cell['module_cell_number']} "
                    f"({CUSTOM_MODULE_CELL_ROWS}x{CUSTOM_MODULE_CELL_COLS} module-cell grid)"
                )

        st.divider()
        rng_col, status_col = st.columns([1, 1])
        with rng_col:
            st.subheader("Random Fault Generator")
            random_types = st.multiselect(
                "Random fault types",
                list(FAULT_TYPES.keys()),
                default=["Hotspot"],
                key="random_fault_types",
            )
            max_faults = max(1, min(50, target_counts(location["id"])["pv_cells"]))
            random_count = st.number_input("Number of random faults", min_value=1, max_value=max_faults, value=min(3, max_faults), step=1)
            random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
            random_locations = st.checkbox("Random fault locations", value=True)
            if st.button("Generate random faults", use_container_width=True):
                if not random_types:
                    st.warning("Choose at least one random fault type.")
                elif random_locations:
                    st.session_state["random_fault_records"] = random_fault_records(
                        location,
                        view,
                        random_types,
                        int(random_count),
                        fault_scale,
                        seed_for(location["id"], view, "|".join(random_types), int(random_count), int(random_seed), custom_row_count(location["id"]), custom_panels_per_row(location["id"])),
                    )
                    st.rerun()
                else:
                    st.session_state["random_fault_records"] = [
                        fault_record(location, view, selected_row, selected_cell, random_type, fault_scale)
                        for random_type in random_types
                    ][: int(random_count)]
                    st.rerun()
            if st.button("Clear generated faults", use_container_width=True):
                st.session_state["random_fault_records"] = []
                st.rerun()
        with status_col:
            counts = target_counts(location["id"])
            active_random = [
                record for record in stored_fault_records()
                if record["location_id"] == location["id"]
            ]
            st.subheader("Current Setting")
            st.metric("Selectable PV cells", counts["pv_cells"])
            st.caption(f"{location['name']} | {view.upper()} | {selected_row['id']} | {selected_cell['id']}")
            st.caption(
                f"Manual target bbox px ({manual_record['bbox_px']['x1']}, {manual_record['bbox_px']['y1']})-"
                f"({manual_record['bbox_px']['x2']}, {manual_record['bbox_px']['y2']})"
            )
            st.metric("Generated faults", len(active_random))

    active_fault_records = [
        record for record in stored_fault_records()
        if record["location_id"] == location["id"]
    ]
    display_records = (
        display_fault_records(manual_record, active_fault_records)
        if lock_manual_target
        else active_fault_records or [manual_record]
    )
    display_records = records_for_view(display_records, view)

    terrain = scene_image_with_faults(location["id"], view, False, selected_row_id, selected_cell_id, fault_type, fault_scale, display_records)
    pv_scene = scene_image_with_faults(location["id"], view, True, selected_row_id, selected_cell_id, fault_type, fault_scale, display_records)
    thermal_records = records_for_view(display_records, "thermal")
    big_screen_thermal = scene_image_with_faults(
        location["id"],
        "thermal",
        True,
        selected_row_id,
        selected_cell_id,
        fault_type,
        fault_scale,
        thermal_records,
    )
    metadata = scene_metadata(location, view, selected_row, selected_cell, display_records)

    with scene_tab:
        st.caption(
            f"{metadata['location_name']} | {metadata['view']} | rows: {metadata['row_count']} | "
            f"modules/row: {metadata['modules_per_row']} | faults: {metadata['fault_count']} | "
            f"target: {metadata['selected_row']} / {metadata['selected_cell']}"
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Terrain only")
            render_photo_map(terrain, location["id"], view, selected_row_id, selected_cell_id, False)
        with right:
            st.subheader("Same location with PV rows")
            render_photo_map(pv_scene, location["id"], view, selected_row_id, selected_cell_id, True, show_bounding_boxes)
            st.caption("Click a row in this image. The selected row opens in the output tab for cell-level inspection.")
        with st.expander("Scene metadata", expanded=False):
            st.json(metadata)

    with output_tab:
        st.subheader("Fault Summary")
        summary_rows = fault_summary_rows(display_records)
        if summary_rows:
            st.table(summary_rows)
        else:
            st.caption("No faults selected.")

        st.subheader("Fault Table")
        st.table(fault_table_rows(display_records))

        st.subheader(f"Cropped row: {selected_row['label']} - selected PV cell {selected_cell['id']}")
        render_zoom_map(pv_scene, location["id"], view, selected_row_id, selected_cell_id)
        st.caption("Click an individual PV cell in the crop, or use the target selector in the generation tab.")

        with st.expander("PV cell IDs and coordinates for selected row", expanded=True):
            st.table(
                [
                    {
                        "pv_cell_number": cell.get("pv_cell_number_in_row", cell.get("panel_number")),
                        "pv_cell_id": cell["id"],
                        "module_number": cell.get("module_number"),
                        "module_cell_number": cell.get("module_cell_number"),
                        "x1": cell["x1"],
                        "y1": cell["y1"],
                        "x2": cell["x2"],
                        "y2": cell["y2"],
                        "center_x": round((cell["x1"] + cell["x2"]) / 2, 2),
                        "center_y": round((cell["y1"] + cell["y2"]) / 2, 2),
                    }
                    for cell in row_cells
                ]
            )

    with big_screen_tab:
        st.caption(
            f"Thermal test view | {location['name']} | rows: {metadata['row_count']} | "
            f"modules/row: {metadata['modules_per_row']} | faults: {metadata['fault_count']}"
        )
        st.image(big_screen_thermal, use_container_width=True)


if __name__ == "__main__":
    main()
