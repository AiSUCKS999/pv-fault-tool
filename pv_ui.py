# Streamlit UI, maps, tabs, controls, and app main().
#
# FIXES in this version (marked "# FIX:" where relevant):
#  - Close diagnostic module rebuilt: square cells on a real 5x8 pitch, chamfered mono cells,
#    busbars + fingers, aluminium frame, and clean glass. No more flat blue rounded boxes.
#  - Close thermal module rebuilt in temperature space: cool cell seams, cooler frame,
#    junction-box warmth, per-cell tone variation, substring-level diode/bypass heating,
#    optics blur + sensor noise + row/column banding. No UI outlines painted on top.
#  - RGB and thermal faults share the same seeded crack / shade / dust shapes so they match.
#  - Segmentation overlays are subtle by default (corner brackets), without debug UI clutter.


def scene_image(location_id, view, include_pv, selected_row_id, selected_cell_id, fault_type, fault_scale):
    rgb_image = composite_with_og_pvs(location_id) if include_pv else terrain_layer(location_id)
    row = row_by_id(location_id, selected_row_id)
    cell = cell_by_id(location_id, row["id"], selected_cell_id)
    if view == "thermal":
        return thermalize(
            rgb_image,
            location_id,
            include_pv,
            cell if include_pv else None,
            fault_type,
            fault_scale,
            scene_distance=qp_get("inspect", "far"),
        )
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
            return thermalize(
                rgb_image,
                location_id,
                include_pv,
                fault_records=active_records,
                scene_distance=qp_get("inspect", "far"),
            )
        return thermalize(
            rgb_image,
            location_id,
            include_pv,
            cell if include_pv else None,
            fault_type,
            fault_scale,
            scene_distance=qp_get("inspect", "far"),
        )
    image = rgb_image
    if include_pv:
        if active_records:
            for record in active_records:
                record_row = row_by_id(location_id, record["row_id"])
                record_cell = cell_with_fault_bbox(
                    cell_by_id(location_id, record_row["id"], record["target_cell_id"]),
                    record,
                )
                image = draw_realistic_fault(image, record_cell, record["fault_type"], record.get("fault_scale", fault_scale), view)
        else:
            image = draw_realistic_fault(image, cell, fault_type, fault_scale, view)
    return image


def data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def draw_fault_bounding_boxes(image, location_id, fault_records):
    boxed = image.convert("RGBA")
    overlay = Image.new("RGBA", boxed.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = grid_font(10)
    occupied_labels = []

    def rectangles_overlap(left, right, pad=0):
        return not (
            left[2] + pad <= right[0]
            or right[2] + pad <= left[0]
            or left[3] + pad <= right[1]
            or right[3] + pad <= left[1]
        )

    def label_position(box, text_w, text_h):
        box_x1, box_y1, box_x2, box_y2 = box
        pad = 8
        label_w = text_w + 10
        label_h = text_h + 6
        box_cx = (box_x1 + box_x2) // 2
        candidates = [
            (box_x1, box_y1 - label_h - pad),
            (box_x1, box_y2 + pad),
            (box_x2 + pad, box_y1),
            (box_x1 - label_w - pad, box_y1),
            (box_cx - label_w // 2, box_y1 - label_h - pad),
            (box_cx - label_w // 2, box_y2 + pad),
            (box_x2 + pad, box_y2 - label_h),
            (box_x1 - label_w - pad, box_y2 - label_h),
        ]
        for dx in (-90, -50, 0, 50, 90):
            candidates.append((box_cx - label_w // 2 + dx, box_y1 - label_h - pad))
            candidates.append((box_cx - label_w // 2 + dx, box_y2 + pad))
        for x, y in candidates:
            if not (0 <= x <= BASE_SIZE[0] - label_w and 0 <= y <= BASE_SIZE[1] - label_h):
                continue
            candidate = (int(x), int(y), int(x + label_w), int(y + label_h))
            if not any(rectangles_overlap(candidate, other, 3) for other in occupied_labels):
                occupied_labels.append(candidate)
                return int(x), int(y)
        x = int(max(0, min(box_x1, BASE_SIZE[0] - label_w)))
        y = int(max(0, min(box_y1 - label_h - pad, BASE_SIZE[1] - label_h)))
        for step_y in range(0, 120, label_h + 4):
            for step_x in range(0, 180, max(24, label_w // 2)):
                for sign_x in (0, -1, 1):
                    candidate_x = int(max(0, min(x + sign_x * step_x, BASE_SIZE[0] - label_w)))
                    for candidate_y in (y - step_y, y + step_y):
                        candidate_y = int(max(0, min(candidate_y, BASE_SIZE[1] - label_h)))
                        candidate = (candidate_x, candidate_y, candidate_x + label_w, candidate_y + label_h)
                        if not any(rectangles_overlap(candidate, other, 3) for other in occupied_labels):
                            occupied_labels.append(candidate)
                            return candidate_x, candidate_y
        candidate = (x, y, x + label_w, y + label_h)
        occupied_labels.append(candidate)
        return x, y

    def draw_corner_box(box, color, width=2):
        x1, y1, x2, y2 = box
        corner = max(7, min(15, int(min(x2 - x1, y2 - y1) * 0.34)))
        segments = [
            ((x1, y1), (x1 + corner, y1)),
            ((x1, y1), (x1, y1 + corner)),
            ((x2, y1), (x2 - corner, y1)),
            ((x2, y1), (x2, y1 + corner)),
            ((x1, y2), (x1 + corner, y2)),
            ((x1, y2), (x1, y2 - corner)),
            ((x2, y2), (x2 - corner, y2)),
            ((x2, y2), (x2, y2 - corner)),
        ]
        for p1, p2 in segments:
            draw.line((p1, p2), fill=color, width=width)

    sorted_records = sorted(
        fault_records or [],
        key=lambda item: (
            item.get("visible_fault_bbox_px", item.get("bbox_px", {})).get("y1", 0),
            item.get("visible_fault_bbox_px", item.get("bbox_px", {})).get("x1", 0),
        ),
    )
    for record in sorted_records:
        if record.get("location_id") != location_id:
            continue
        row = row_by_id(location_id, record["row_id"])
        cell = cell_with_fault_bbox(cell_by_id(location_id, row["id"], record["target_cell_id"]), record)
        label = fault_display_name(record["fault_type"])
        x1, y1, x2, y2 = fault_annotation_bounds_for_cell(cell, record["fault_type"])
        box_x1 = max(0, min(x1 - 2, BASE_SIZE[0] - 1))
        box_y1 = max(0, min(y1 - 2, BASE_SIZE[1] - 1))
        box_x2 = min(BASE_SIZE[0] - 1, max(x2 + 2, box_x1 + 10))
        box_y2 = min(BASE_SIZE[1] - 1, max(y2 + 2, box_y1 + 10))
        draw_corner_box((box_x1, box_y1, box_x2, box_y2), (56, 189, 248, 205), 1)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_x, label_y = label_position((box_x1, box_y1, box_x2, box_y2), text_w, text_h)
        label_w = text_w + 10
        label_h = text_h + 6
        draw.rounded_rectangle(
            (label_x, label_y, label_x + label_w, label_y + label_h),
            radius=2,
            fill=(15, 23, 42, 160),
            outline=(125, 211, 252, 205),
            width=1,
        )
        anchor_x = max(box_x1, min(label_x + label_w // 2, box_x2))
        anchor_y = box_y1 if label_y < box_y1 else box_y2
        draw.line(
            ((label_x + label_w // 2, label_y + label_h // 2), (anchor_x, anchor_y)),
            fill=(125, 211, 252, 180),
            width=1,
        )
        draw.text((label_x + 5, label_y + 3), label, fill=(255, 255, 255, 255), font=font)
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


# FIX: shared helper for subtle selection / fault markers (replaces thick full-rectangle outlines).
def draw_corner_marks(draw, box, color, width=2, ratio=0.28, grow=2):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 - grow, y1 - grow, x2 + grow, y2 + grow
    c = max(5, int(min(x2 - x1, y2 - y1) * ratio))
    for p1, p2 in (
        ((x1, y1), (x1 + c, y1)), ((x1, y1), (x1, y1 + c)),
        ((x2, y1), (x2 - c, y1)), ((x2, y1), (x2, y1 + c)),
        ((x1, y2), (x1 + c, y2)), ((x1, y2), (x1, y2 - c)),
        ((x2, y2), (x2 - c, y2)), ((x2, y2), (x2, y2 - c)),
    ):
        draw.line((p1, p2), fill=color, width=width)


def short_fault_label(fault_type):
    name = fault_display_name(fault_type)
    return {
        "Surface Obstruction / Partial Shading": "Partial Shading",
        "Thermal Hotspot Symptom": "Hotspot",
        "Bypass Diode Fault": "Diode Fault",
        "Open Circuit String": "Open String",
        "Reversed Polarity String": "Reversed String",
    }.get(name, name)


def draw_fault_label(draw, box, label, font, canvas_size, fill=(56, 189, 248, 230)):
    x1, y1, x2, y2 = box
    canvas_w, canvas_h = canvas_size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad_x, pad_y = 5, 3
    label_w = text_w + pad_x * 2
    label_h = text_h + pad_y * 2
    label_x = max(2, min(x1, canvas_w - label_w - 2))
    label_y = y1 - label_h - 4
    if label_y < 2:
        label_y = min(y2 + 4, canvas_h - label_h - 2)
    draw.rounded_rectangle(
        (label_x, label_y, label_x + label_w, label_y + label_h),
        radius=2,
        fill=(8, 13, 24, 190),
        outline=fill,
        width=1,
    )
    draw.text((label_x + pad_x, label_y + pad_y), label, fill=(255, 255, 255, 245), font=font)


# FIX: row-crop overlay is subtle by default. Full grid/numbers only when show_grid=True.
def draw_visible_cell_grid_on_crop(cropped, location_id, selected_row_id, selected_cell_id, crop, scale=1, show_grid=False):
    image = cropped.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    crop_w, crop_h = cropped.size
    cells = cells_for_row(location_id, selected_row_id)
    font_size = max(6, min(11, int(crop_h * 0.10), int(crop_w / max(12, len(cells) / 2) * 0.55)))
    font = grid_font(font_size)
    module_boxes = {}
    selected_cell_box = None

    for cell in cells:
        rel_x1 = (cell["x1"] - crop[0]) * scale
        rel_y1 = (cell["y1"] - crop[1]) * scale
        rel_x2 = (cell["x2"] - crop[0]) * scale
        rel_y2 = (cell["y2"] - crop[1]) * scale
        selected = cell["id"] == selected_cell_id
        if show_grid:
            draw.rectangle((rel_x1, rel_y1, rel_x2, rel_y2), outline=(224, 242, 254, 55), width=1)

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

        if show_grid:
            label = str(cell.get("pv_cell_number_in_row", cell.get("panel_number", "")))
            if len(cells) <= 120 and rel_x2 - rel_x1 >= 12 and rel_y2 - rel_y1 >= 9:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                tx = rel_x1 + max(1, (rel_x2 - rel_x1 - text_w) / 2)
                ty = rel_y1 + max(1, (rel_y2 - rel_y1 - text_h) / 2)
                draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 150), font=font)
                draw.text((tx, ty), label, fill=(255, 255, 255, 200), font=font)

    if show_grid and module_boxes:
        label_font = grid_font(max(8, min(12, int(crop_h * 0.11))))
        for module_number, (mx1, my1, mx2, my2) in module_boxes.items():
            draw.rectangle((mx1, my1, mx2, my2), outline=(255, 255, 255, 120), width=1)
            label = f"M{module_number}"
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
            label_x = mx1 + 2
            label_y = my1 - label_h - 7 if my1 - label_h - 7 >= 2 else my1 + 2
            draw.rounded_rectangle(
                (label_x, label_y, label_x + label_w + 6, label_y + label_h + 4),
                radius=2,
                fill=(4, 8, 18, 140),
            )
            draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255, 220), font=label_font)

    if selected_cell_box:
        draw_corner_marks(draw, selected_cell_box, (250, 204, 21, 255), width=2)

    return image.convert("RGB")


def render_photo_map(image, location_id, view, selected_row_id, selected_cell_id, include_links, show_bounding_boxes=False, fault_records=None):
    if include_links and show_bounding_boxes:
        image = draw_fault_bounding_boxes(image, location_id, fault_records)
    uri = data_uri(image)
    svg_parts = [
        f'<svg width="{BASE_SIZE[0]}" height="{BASE_SIZE[1]}" viewBox="0 0 {BASE_SIZE[0]} {BASE_SIZE[1]}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{BASE_SIZE[0]}" height="{BASE_SIZE[1]}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    if include_links:
        for panel in panels(location_id):
            selected = panel["id"] == selected_row_id
            stroke = "#38bdf8"
            opacity = "0.0"
            width = "1"
            fill_opacity = "0.001"
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
          .frame {{ width:100%; overflow:auto; border:1px solid #263244; border-radius:10px; background:#050816; }}
          svg {{ width:100%; height:auto; display:block; }}
          a {{ cursor:pointer; }}
          a:hover rect {{ opacity:.55; stroke:#38bdf8; stroke-width:2; }}
        </style>
        """,
        height=430,
        scrolling=True,
    )


def render_zoom_map(image, location_id, view, selected_row_id, selected_cell_id, show_grid=False):
    row = row_by_id(location_id, selected_row_id)
    crop = crop_box_for_row(row)
    cropped = image.crop(crop)
    cells = cells_for_row(location_id, selected_row_id)
    base_crop_w, base_crop_h = cropped.size
    zoom_scale = max(1, min(5, math.ceil(len(cells) * 13 / max(1, base_crop_w))))
    if zoom_scale > 1:
        cropped = cropped.resize((base_crop_w * zoom_scale, base_crop_h * zoom_scale), Image.Resampling.LANCZOS)
    cropped = draw_visible_cell_grid_on_crop(cropped, location_id, selected_row_id, selected_cell_id, crop, zoom_scale, show_grid)
    uri = data_uri(cropped)
    crop_w, crop_h = cropped.size
    svg_parts = [
        f'<svg width="{crop_w}" height="{crop_h}" viewBox="0 0 {crop_w} {crop_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{crop_w}" height="{crop_h}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    for cell in cells:
        target = f"?{urlencode(navigation_params(location_id, view, selected_row_id, cell['id']))}"
        svg_parts.append(
            f'<a href="{esc(target)}" target="_parent">'
            f'<title>{esc(cell["label"])}</title>'
            f'<rect x="{(cell["x1"] - crop[0]) * zoom_scale}" y="{(cell["y1"] - crop[1]) * zoom_scale}" '
            f'width="{(cell["x2"] - cell["x1"]) * zoom_scale}" '
            f'height="{(cell["y2"] - cell["y1"]) * zoom_scale}" fill="#000000" fill-opacity="0.001" '
            f'stroke="#e0f2fe" stroke-width="1" opacity="0" pointer-events="all"/></a>'
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
          a:hover rect {{ opacity:1; stroke:#38bdf8; stroke-width:2; }}
        </style>
        """,
        height=height,
        scrolling=False,
    )


def module_crop_box(location_id, selected_row_id, selected_cell_id, pad_ratio=0.28):
    cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
    bbox = module_bbox_for_cell(location_id, selected_row_id, cell)
    module_w = max(1, bbox["x2"] - bbox["x1"])
    module_h = max(1, bbox["y2"] - bbox["y1"])
    pad_x = max(6, int(module_w * pad_ratio))
    pad_y = max(6, int(module_h * pad_ratio))
    return (
        clamp(bbox["x1"] - pad_x, 0, BASE_SIZE[0] - 1),
        clamp(bbox["y1"] - pad_y, 0, BASE_SIZE[1] - 1),
        clamp(bbox["x2"] + pad_x, 1, BASE_SIZE[0]),
        clamp(bbox["y2"] + pad_y, 1, BASE_SIZE[1]),
    )


def module_cells_for_target(location_id, selected_row_id, selected_cell_id):
    selected = cell_by_id(location_id, selected_row_id, selected_cell_id)
    module_number = selected.get("module_number")
    if not module_number:
        return [selected]
    return [
        cell for cell in cells_for_row(location_id, selected_row_id)
        if cell.get("module_number") == module_number
    ]


# ---------------------------------------------------------------------------
# FIX: realistic close-module geometry, textures, faults (RGB + thermal)
# ---------------------------------------------------------------------------

def _noise_field(shape, seed, sigma, octaves=2):
    """Self-contained smooth 0..1 noise (no dependency on other helpers)."""
    h, w = shape
    rng = np.random.default_rng(int(seed) % (2 ** 32))
    out = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    for octave in range(octaves):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), max(0.5, sigma / (2 ** octave)))
        n -= float(n.mean())
        n /= max(float(n.std()), 1e-6)
        out += n * amp
        total += amp
        amp *= 0.5
    out /= total
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(1e-6, float(hi - lo)), 0, 1).astype(np.float32)


def close_module_geometry(size):
    """Square cells on a real pitch. Frame and backsheet border are sized to hug the cell grid."""
    width, height = size
    cols = CUSTOM_MODULE_CELL_ROWS   # display columns (10 for a 40-cell module laid landscape)
    rows = CUSTOM_MODULE_CELL_COLS   # display rows (4)
    outer_margin = max(16, int(min(width, height) * 0.03))
    frame_t = max(12, int(min(width, height) * 0.026))
    pad = max(6, int(frame_t * 0.55))
    border = frame_t + pad
    avail_w = width - 2 * outer_margin - 2 * border
    avail_h = height - 2 * outer_margin - 2 * border
    pitch = min(avail_w / cols, avail_h / rows)
    grid_w, grid_h = pitch * cols, pitch * rows
    mod_w, mod_h = grid_w + 2 * border, grid_h + 2 * border
    mx1 = (width - mod_w) / 2
    my1 = (height - mod_h) / 2
    frame = (int(round(mx1)), int(round(my1)), int(round(mx1 + mod_w)), int(round(my1 + mod_h)))
    inner = (frame[0] + frame_t, frame[1] + frame_t, frame[2] - frame_t, frame[3] - frame_t)
    origin = (mx1 + border, my1 + border)
    return {
        "cols": cols,
        "rows": rows,
        "pitch": pitch,
        "gap": max(2.0, pitch * 0.045),
        "frame": frame,
        "inner": inner,
        "origin": origin,
    }


def close_module_cell_rects(module_cells, size):
    """High-detail diagnostic view: one 40-cell module, drawn landscape, square cells."""
    geo = close_module_geometry(size)
    pitch, gap = geo["pitch"], geo["gap"]
    ox, oy = geo["origin"]
    rects = {}
    for cell in module_cells:
        col = clamp(int(cell.get("module_cell_row") or 1) - 1, 0, geo["cols"] - 1)
        row = clamp(int(cell.get("module_cell_col") or 1) - 1, 0, geo["rows"] - 1)
        x1 = int(round(ox + col * pitch + gap / 2))
        x2 = int(round(ox + (col + 1) * pitch - gap / 2))
        y1 = int(round(oy + row * pitch + gap / 2))
        y2 = int(round(oy + (row + 1) * pitch - gap / 2))
        rects[cell["id"]] = (x1, y1, max(x1 + 4, x2), max(y1 + 4, y2))
    return rects


def pv_cell_chamfer_mask(w, h):
    """Pseudo-square mono cell: clipped corners, anti-aliased."""
    c = max(2, int(round(min(w, h) * 0.085)))
    ss = 3
    m = np.zeros((h * ss, w * ss), dtype=np.uint8)
    pts = np.array(
        [(c, 0), (w - c, 0), (w, c), (w, h - c), (w - c, h), (c, h), (0, h - c), (0, c)],
        dtype=np.float32,
    ) * ss
    cv2.fillConvexPoly(m, pts.round().astype(np.int32), 255, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def pv_cell_tile(w, h, rng):
    """RGB texture of one monocrystalline cell: blue-black base, fingers, 5 busbars, grain."""
    tone = float(rng.normal(0, 3.2))
    base = np.array([34, 48, 82], np.float32) + tone * np.array([0.55, 0.78, 1.08], np.float32)
    tile = np.empty((h, w, 3), np.float32)
    tile[:] = base
    grain = cv2.GaussianBlur(rng.normal(0, 2.6, (h, w)).astype(np.float32), (0, 0), 0.9)
    tile += grain[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
    spacing = max(4, int(round(h / 22)))
    tile[::spacing] += np.array([2.2, 3.2, 5.0], np.float32)
    bw = max(1, int(round(w / 70)))
    for k in range(5):
        cx = int(round((k + 0.5) / 5 * w))
        x0 = max(0, cx - bw // 2)
        x1 = min(w, x0 + bw)
        tile[:, x0:x1] = tile[:, x0:x1] * 0.58 + np.array([132, 148, 170], np.float32) * 0.42
    tile *= (1 + np.linspace(0.03, -0.03, h, dtype=np.float32))[:, None, None]
    return tile, pv_cell_chamfer_mask(w, h)


def crack_polylines(w, h, seed):
    rnd = random.Random(int(seed))
    x0 = rnd.uniform(0.0, 0.25) * w
    y0 = rnd.uniform(0.15, 0.85) * h
    x1 = rnd.uniform(0.75, 1.0) * w
    y1 = rnd.uniform(0.15, 0.85) * h

    def wander(p, q, steps, jitter):
        pts = [p]
        for i in range(1, steps):
            t = i / steps
            pts.append((
                p[0] + (q[0] - p[0]) * t + rnd.gauss(0, jitter * w),
                p[1] + (q[1] - p[1]) * t + rnd.gauss(0, jitter * h),
            ))
        pts.append(q)
        return pts

    main = wander((x0, y0), (x1, y1), 9, 0.035)
    lines = [main]
    for _ in range(rnd.randint(1, 3)):
        base = main[rnd.randint(2, len(main) - 3)]
        ang = rnd.uniform(-0.9, 0.9) + (math.pi / 2 if rnd.random() < 0.5 else -math.pi / 2)
        length = rnd.uniform(0.18, 0.42) * min(w, h)
        end = (base[0] + math.cos(ang) * length, base[1] + math.sin(ang) * length)
        lines.append(wander(base, end, 5, 0.03))
    return lines


def crack_mask(w, h, seed):
    ss = 3
    big = np.zeros((h * ss, w * ss), dtype=np.uint8)
    thick = max(1, int(round(max(1.0, min(w, h) * 0.018) * ss)))
    for line in crack_polylines(w, h, seed):
        pts = (np.array(line, dtype=np.float32) * ss).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(big, [pts], False, 255, thick, cv2.LINE_AA)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def soiling_mask(w, h, seed):
    blob = _noise_field((h, w), seed_for(seed, "blob"), max(2.0, min(w, h) * 0.20), 3)
    speck = _noise_field((h, w), seed_for(seed, "speck"), max(0.8, min(w, h) * 0.04), 2)
    dust = np.clip((blob - 0.42) / 0.38, 0, 1) * 0.65 + np.clip((speck - 0.68) / 0.30, 0, 1) * 0.35
    dust *= 0.55 + 0.9 * np.linspace(0, 1, h, dtype=np.float32)[:, None]   # dirt collects at the lower edge
    return np.clip(dust, 0, 1).astype(np.float32)


def shade_mask(w, h, seed):
    rnd = random.Random(int(seed))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    slope = rnd.uniform(-0.45, 0.45)
    boundary = h * rnd.uniform(0.25, 0.55) + slope * (xx - w * 0.5)
    wob = (_noise_field((h, w), seed_for(seed, "wob"), max(2.0, min(w, h) * 0.18), 3) - 0.5) * h * 0.30
    mask = (yy > boundary + wob).astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), max(0.8, min(w, h) * 0.025)), 0, 1)


def _fault_seed(record, label):
    return seed_for(record["target_cell_id"], record["fault_type"], record.get("fault_scale", 8), label)


def draw_close_rgb_module(module_cells, selected_cell_id, fault_records, size, location_id):
    width, height = size
    geo = close_module_geometry(size)
    rects = close_module_cell_rects(module_cells, size)
    rng = np.random.default_rng(seed_for(location_id, selected_cell_id, "close-rgb-module"))

    canvas = np.empty((height, width, 3), np.float32)
    canvas[:] = (20, 25, 34)
    canvas += np.linspace(0, 1, width, dtype=np.float32)[None, :, None] * 4
    canvas += np.linspace(0, 1, height, dtype=np.float32)[:, None, None] * 5

    # Aluminium frame
    fx1, fy1, fx2, fy2 = geo["frame"]
    fh, fw = fy2 - fy1, fx2 - fx1
    frame = np.empty((fh, fw, 3), np.float32)
    frame[:] = (118, 128, 138)
    frame += rng.normal(0, 1.8, (1, fw, 1)).astype(np.float32) + rng.normal(0, 0.9, (fh, fw, 1)).astype(np.float32)
    frame[:3] += 14
    frame[-3:] -= 14
    canvas[fy1:fy2, fx1:fx2] = frame

    # Backsheet visible between cells
    ix1, iy1, ix2, iy2 = geo["inner"]
    canvas[iy1:iy2, ix1:ix2] = (25, 32, 45)
    canvas[iy1:iy1 + 2, ix1:ix2] *= 0.72
    canvas[iy1:iy2, ix1:ix1 + 2] *= 0.78

    # Cells
    for cell in module_cells:
        x1, y1, x2, y2 = rects[cell["id"]]
        w, h = x2 - x1, y2 - y1
        tile, m = pv_cell_tile(w, h, rng)
        region = canvas[y1:y2, x1:x2]
        region[:] = region * (1 - m[..., None]) + tile * m[..., None]

    # Keep RGB glass clean. Strong synthetic glare made the panels look pasted on.
    iw, ih = ix2 - ix1, iy2 - iy1
    clean_glass = (_noise_field((ih, iw), seed_for(location_id, "close-clean-glass"), iw * 0.35, 2) - 0.5) * 1.4
    canvas[iy1:iy2, ix1:ix2] += clean_glass[..., None] * np.array([0.45, 0.55, 0.70], np.float32)

    # Faults visible in daylight
    for record in fault_records or []:
        tid = record.get("target_cell_id")
        if tid not in rects:
            continue
        x1, y1, x2, y2 = rects[tid]
        w, h = x2 - x1, y2 - y1
        ft = canonical_fault_type(record["fault_type"])
        sev = clamp(record.get("fault_scale", 8), 1, 10) / 10
        region = canvas[y1:y2, x1:x2]
        if ft in CRACKING_FAULTS:
            core = crack_mask(w, h, _fault_seed(record, "close-crack"))
            shadow = cv2.GaussianBlur(core, (0, 0), 1.3)
            region[:] = region * (1 - shadow[..., None] * 0.28) + np.array([205, 214, 222], np.float32) * (core[..., None] * 0.70)
        elif ft in SOILING_FAULTS:
            dust = soiling_mask(w, h, _fault_seed(record, "close-dust")) * (0.55 + 0.35 * sev)
            region[:] = region * (1 - dust[..., None] * 0.6) + np.array([118, 102, 78], np.float32) * (dust[..., None] * 0.6)
        elif ft in SURFACE_OBSTRUCTION_FAULTS:
            mask = shade_mask(w, h, _fault_seed(record, "close-shade")) * (0.55 + 0.25 * sev)
            region[:] = region * (1 - mask[..., None]) + np.array([4, 8, 16], np.float32) * mask[..., None]
        elif ft in HOTSPOT_FAULTS:
            rnd = random.Random(_fault_seed(record, "close-burn"))
            cx, cy = w * rnd.uniform(0.38, 0.62), h * rnd.uniform(0.38, 0.62)
            yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
            d2 = ((xx2 - cx) / (w * 0.20)) ** 2 + ((yy2 - cy) / (h * 0.20)) ** 2
            brown = np.exp(-d2 * 0.8) * (0.30 + 0.30 * sev)
            core_dark = np.exp(-d2 * 3.0) * (0.35 + 0.25 * sev)
            region[:] = region * (1 - brown[..., None]) + np.array([96, 70, 34], np.float32) * brown[..., None]
            region[:] = region * (1 - core_dark[..., None]) + np.array([36, 26, 20], np.float32) * core_dark[..., None]
        # Diode / bypassed / string / legacy-line faults have no daylight signature on purpose.

    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.55)
    canvas += rng.normal(0, 1.1, (height, width, 1)).astype(np.float32)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")


def _add_gaussian(temp, cx, cy, sx, sy, amp):
    h, w = temp.shape
    x1, x2 = int(max(0, cx - 4 * sx)), int(min(w, cx + 4 * sx + 1))
    y1, y2 = int(max(0, cy - 4 * sy)), int(min(h, cy + 4 * sy + 1))
    if x2 <= x1 or y2 <= y1:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    temp[y1:y2, x1:x2] += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)


def _substring_band(shape, geo, sub_index, rows_per_sub):
    ox, oy = geo["origin"]
    pitch = geo["pitch"]
    y1 = int(oy + sub_index * rows_per_sub * pitch)
    y2 = int(oy + (sub_index + 1) * rows_per_sub * pitch)
    x1 = int(ox)
    x2 = int(ox + geo["cols"] * pitch)
    m = np.zeros(shape, np.float32)
    m[max(0, y1):y2, max(0, x1):x2] = 1.0
    return cv2.GaussianBlur(m, (0, 0), max(1.5, pitch * 0.10))


def draw_close_thermal_module(module_cells, selected_cell_id, fault_records, size, location_id, scene_distance="close"):
    width, height = size
    geo = close_module_geometry(size)
    rects = close_module_cell_rects(module_cells, size)
    cell_map = {c["id"]: c for c in module_cells}
    pitch = geo["pitch"]
    rng = np.random.default_rng(seed_for(location_id, module_cells[0]["id"] if module_cells else "empty", "close-thermal-module"))

    # Background (sky / ground behind the module)
    temp = np.full((height, width), 70, np.float32)
    temp = thermal_background_realism(temp, terrain_layer(location_id), location_id)

    # Frame (cooler, metal) and backsheet gaps
    fx1, fy1, fx2, fy2 = geo["frame"]
    ix1, iy1, ix2, iy2 = geo["inner"]
    temp[fy1:fy2, fx1:fx2] = 104
    temp[iy1:iy2, ix1:ix2] = 126
    iw, ih = ix2 - ix1, iy2 - iy1

    # The fallback close renderer uses the same healthy module plane as scene
    # masters. Preserve physical cell gaps without stamping a warm spot beneath
    # every diagnosis. Target selection never seeds the healthy texture.
    module_seed = seed_for(location_id, module_cells[0]["id"] if module_cells else "empty", "healthy-close-module")
    module_field = healthy_thermal_module_field((iw, ih), module_seed, *thermal_pv_operating_range(location_id))
    module_field = thermal_module_frame_continuity(
        module_field, np.ones(module_field.shape, np.uint8),
        [np.float32([[0, 0], [iw-1, 0], [iw-1, ih-1], [0, ih-1]])],
        seed=module_seed,
    )
    temp[iy1:iy2, ix1:ix2] = module_field

    rows_per_sub = max(1, geo["rows"] // 3)

    for record in thermal_records_for_distance(fault_records, scene_distance):
        tid = record.get("target_cell_id")
        if str(scene_distance).lower() == "close" and tid != selected_cell_id:
            continue
        if tid not in rects:
            continue
        x1, y1, x2, y2 = rects[tid]
        cw, ch = x2 - x1, y2 - y1
        ft = canonical_fault_type(record["fault_type"])
        sev = clamp(record.get("fault_scale", 8), 1, 10) / 10
        row_idx = int(cell_map[tid].get("module_cell_col") or 1) - 1
        sub_idx = clamp(row_idx // rows_per_sub, 0, 2)

        if ft in {"SingleDiode", "SingleByPassed"}:
            target = dict(cell_map[tid])
            target.update({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "module_bbox_px": {"x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2},
                "active_module_bbox_px": {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2},
            })
            temp = apply_thermal_fault(
                temp, target, ft, record.get("fault_scale", DEFAULT_FAULT_SCALE),
                production_reference_id=record.get("production_reference_id"), require_production=True,
            )
            continue

        if ft in CRACKING_FAULTS:
            core = crack_mask(cw, ch, _fault_seed(record, "close-crack"))   # same shape as RGB
            glow = cv2.GaussianBlur(core, (0, 0), max(1.5, min(cw, ch) * 0.05))
            patch = _noise_field((ch, cw), _fault_seed(record, "crack-patch"), min(cw, ch) * 0.22, 2)
            delta = glow * (26 + 34 * sev) + core * 10 + np.clip((patch - 0.4) / 0.5, 0, 1) * (16 + 22 * sev) + 5 * sev
            temp[y1:y2, x1:x2] += delta
        elif ft in SURFACE_OBSTRUCTION_FAULTS:
            mask = shade_mask(cw, ch, _fault_seed(record, "close-shade"))   # same shape as RGB
            temp[y1:y2, x1:x2] -= mask * (28 + 30 * sev)
        elif ft in SOILING_FAULTS:
            dust = soiling_mask(cw, ch, _fault_seed(record, "close-dust"))
            temp[y1:y2, x1:x2] -= dust * (10 + 14 * sev)
        elif ft in DIODE_FAULTS:
            temp += _substring_band(temp.shape, geo, sub_idx, rows_per_sub) * (20 + 22 * sev)
            _add_gaussian(temp, (ix1 + ix2) / 2, iy1 + ih * 0.12, iw * 0.06, ih * 0.08, 8 + 12 * sev)
        elif ft in BYPASSED_FAULTS:
            temp += _substring_band(temp.shape, geo, sub_idx, rows_per_sub) * (14 + 16 * sev)
        elif ft in STRING_FAULTS | LEGACY_LINE_FAULTS:
            m = np.zeros(temp.shape, np.float32)
            m[iy1:iy2, ix1:ix2] = 1.0
            temp += cv2.GaussianBlur(m, (0, 0), 3.0) * (12 + 14 * sev)
        else:
            rnd = random.Random(_fault_seed(record, "close-hot"))
            spots = 2 + rnd.randint(0, 1) if ft == "MultiHotSpot" else 1
            for spot_idx in range(spots):
                cx = x1 + cw * rnd.uniform(0.34, 0.66)
                cy = y1 + ch * rnd.uniform(0.34, 0.66)
                sx = cw * (0.18 if spots == 1 else 0.12) * rnd.uniform(0.78, 1.18)
                sy = ch * (0.22 if spots == 1 else 0.15) * rnd.uniform(0.78, 1.18)
                local_h, local_w = max(1, y2 - y1), max(1, x2 - x1)
                yy_l, xx_l = np.mgrid[0:local_h, 0:local_w].astype(np.float32)
                dx = xx_l + x1 - cx
                dy = yy_l + y1 - cy
                angle = rnd.uniform(-0.8, 0.8)
                xr = math.cos(angle) * dx + math.sin(angle) * dy
                yr = -math.sin(angle) * dx + math.cos(angle) * dy
                core = np.exp(-((xr / max(1.0, sx)) ** 2 + (yr / max(1.0, sy)) ** 2) * 2.2)
                halo = np.exp(-((xr / max(1.0, sx * 1.9)) ** 2 + (yr / max(1.0, sy * 1.7)) ** 2) * 1.1)
                rough = _noise_field((local_h, local_w), _fault_seed(record, f"close-hot-rough-{spot_idx}"), max(4.0, min(cw, ch) * 0.22), 3)
                anomaly = np.clip(halo * (0.62 + rough * 0.48) + core * 0.62, 0, 1)
                temp[y1:y2, x1:x2] += anomaly * (34 + 46 * sev) + core * (28 + 34 * sev)
            temp[y1:y2, x1:x2] += 5 + 7 * sev

    return thermal_display_rgb_from_scalar(
        apply_sensor_model(temp, location_id, scene_distance), location_id, scene_distance
    )


def close_module_diagnostic_image(location_id, view, selected_row_id, selected_cell_id, fault_records=None):
    module_cells = module_cells_for_target(location_id, selected_row_id, selected_cell_id)
    size = (1100, 650)
    records = [
        record for record in (fault_records or [])
        if record.get("target_cell_id") in {cell["id"] for cell in module_cells}
    ]
    if view == "thermal":
        return draw_close_thermal_module(module_cells, selected_cell_id, records, size, location_id)
    return draw_close_rgb_module(module_cells, selected_cell_id, records, size, location_id)


def render_module_zoom_map(image, location_id, view, selected_row_id, selected_cell_id, fault_records=None, show_grid=False):
    module_image = close_module_diagnostic_image(location_id, view, selected_row_id, selected_cell_id, fault_records)
    crop_w, crop_h = module_image.size
    module_cells = module_cells_for_target(location_id, selected_row_id, selected_cell_id)
    selected_module = cell_by_id(location_id, selected_row_id, selected_cell_id).get("module_number")
    base_rects = close_module_cell_rects(module_cells, module_image.size)

    if image is not None:
        background = ImageOps.fit(image.convert("RGB"), module_image.size, method=Image.Resampling.BICUBIC)
        background = background.filter(ImageFilter.GaussianBlur(4.0))
        background = ImageEnhance.Brightness(background).enhance(0.62)
        background = ImageEnhance.Contrast(background).enhance(0.88).convert("RGBA")
        scale = 0.9
        panel_w = int(crop_w * scale)
        panel_h = int(crop_h * scale)
        panel_x = (crop_w - panel_w) // 2
        panel_y = (crop_h - panel_h) // 2
        shadow = Image.new("RGBA", (panel_w + 42, panel_h + 42), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow, "RGBA")
        shadow_draw.rounded_rectangle((21, 21, panel_w + 21, panel_h + 21), radius=6, fill=(0, 0, 0, 118))
        shadow = shadow.filter(ImageFilter.GaussianBlur(9.0))
        background.alpha_composite(shadow, (panel_x - 21, panel_y - 14))
        background.paste(module_image.resize((panel_w, panel_h), Image.Resampling.BICUBIC), (panel_x, panel_y))
        cropped = background.convert("RGB")
        rects = {
            cell_id: (
                int(panel_x + rect[0] * scale),
                int(panel_y + rect[1] * scale),
                int(panel_x + rect[2] * scale),
                int(panel_y + rect[3] * scale),
            )
            for cell_id, rect in base_rects.items()
        }
    else:
        cropped = module_image
        rects = base_rects

    draw = ImageDraw.Draw(cropped, "RGBA")
    faults_by_cell = {}
    for record in fault_records or []:
        faults_by_cell.setdefault(record.get("target_cell_id"), []).append(record)
    fault_cell_ids = set(faults_by_cell)
    label_font = grid_font(12)

    # FIX: no thick boxes on every cell. Corner brackets mark selection / faults; grid is opt-in.
    for cell in module_cells:
        rect = rects[cell["id"]]
        selected = cell["id"] == selected_cell_id
        faulted = cell["id"] in fault_cell_ids
        if show_grid:
            outline = (56, 189, 248, 225) if faulted else (224, 242, 254, 45)
            draw.rectangle(rect, outline=outline, width=2 if faulted else 1)
        if faulted and not selected:
            draw_corner_marks(draw, rect, (56, 189, 248, 235), width=2)
        if selected:
            draw_corner_marks(draw, rect, (250, 204, 21, 255), width=3)
        if show_grid and faulted:
            names = []
            for record in faults_by_cell[cell["id"]]:
                name = short_fault_label(record.get("fault_type"))
                if name not in names:
                    names.append(name)
            label = ", ".join(names[:2])
            if len(names) > 2:
                label += f" +{len(names) - 2}"
            draw_fault_label(draw, rect, label, label_font, cropped.size)

    uri = data_uri(cropped)
    svg_parts = [
        f'<svg width="{crop_w}" height="{crop_h}" viewBox="0 0 {crop_w} {crop_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{crop_w}" height="{crop_h}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    for cell in module_cells:
        target = f"?{urlencode(navigation_params(location_id, view, selected_row_id, cell['id'], inspect='close'))}"
        rel_x1, rel_y1, rel_x2, rel_y2 = rects[cell["id"]]
        svg_parts.append(
            f'<a href="{esc(target)}" target="_parent">'
            f'<title>{esc(cell["label"])}</title>'
            f'<rect x="{rel_x1}" y="{rel_y1}" '
            f'width="{rel_x2 - rel_x1}" '
            f'height="{rel_y2 - rel_y1}" fill="#000000" fill-opacity="0.001" '
            f'stroke="#38bdf8" stroke-width="1" opacity="0" pointer-events="all"/></a>'
        )
    svg_parts.append("</svg>")
    components.html(
        f"""
        <div class="module-frame">{''.join(svg_parts)}</div>
        <style>
          .module-frame {{ width:100%; overflow:auto; border:1px solid #263244; border-radius:10px; background:#050816; }}
          svg {{ width:{crop_w}px; max-width:none; height:auto; display:block; margin:auto; }}
          a {{ cursor:pointer; }}
          a:hover rect {{ opacity:1; stroke:#facc15; stroke-width:2; }}
        </style>
        """,
        height=min(820, max(420, crop_h + 44)),
        scrolling=False,
    )
    return selected_module, len(module_cells)


def render_diagnostic_module_map(location, view, selected_row_id, selected_cell_id, selected_region, selected_module, fault_records=None, show_boxes=True, show_cell_map=False):
    module_image = close_module_diagnostic_image(location["id"], view, selected_row_id, selected_cell_id, fault_records)
    crop_w, crop_h = module_image.size
    module_cells = module_cells_for_target(location["id"], selected_row_id, selected_cell_id)
    rects = close_module_cell_rects(module_cells, module_image.size)
    draw = ImageDraw.Draw(module_image, "RGBA")
    faults_by_cell = {}
    for record in fault_records or []:
        if record.get("location_id") != location["id"]:
            continue
        faults_by_cell.setdefault(record.get("target_cell_id"), []).append(record)
    label_font = grid_font(12)

    for cell in module_cells:
        rect = rects[cell["id"]]
        selected = cell["id"] == selected_cell_id
        faulted = cell["id"] in faults_by_cell
        if show_cell_map:
            draw.rectangle(rect, outline=(186, 230, 253, 38), width=1)
        if show_boxes and faulted:
            draw_corner_marks(draw, rect, (251, 146, 60, 245), width=2)
            names = []
            for record in faults_by_cell[cell["id"]]:
                name = short_fault_label(record.get("fault_type"))
                if name not in names:
                    names.append(name)
            label = ", ".join(names[:2])
            if len(names) > 2:
                label += f" +{len(names) - 2}"
            draw_fault_label(draw, rect, label, label_font, module_image.size)
        if selected:
            draw_corner_marks(draw, rect, (56, 189, 248, 255), width=3)

    uri = data_uri(module_image)
    svg_parts = [
        f'<svg width="{crop_w}" height="{crop_h}" viewBox="0 0 {crop_w} {crop_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{crop_w}" height="{crop_h}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    row_number = selected_region["row"] if selected_region else row_number_from_id(location["id"], selected_row_id)
    array_col = selected_region["col"] if selected_region else int(cell_by_id(location["id"], selected_row_id, selected_cell_id).get("panel_number") or 1)
    array_key = selected_region["key"] if selected_region else f"R{row_number}-A{array_col:02d}"
    for cell in module_cells:
        rel_x1, rel_y1, rel_x2, rel_y2 = rects[cell["id"]]
        params = navigation_params(
            location["id"],
            view,
            selected_row_id,
            cell["id"],
            section="Faults",
            selected_array=array_key,
            selected_module=selected_module,
            cell_active=1,
        )
        target = f"?{urlencode(params)}"
        svg_parts.append(
            f'<a href="{esc(target)}" target="_parent">'
            f'<title>{esc(cell["label"])}</title>'
            f'<rect x="{rel_x1}" y="{rel_y1}" width="{rel_x2 - rel_x1}" height="{rel_y2 - rel_y1}" '
            f'fill="#000000" fill-opacity="0.001" stroke="#38bdf8" stroke-width="1" opacity="0" pointer-events="all"/></a>'
        )
    svg_parts.append("</svg>")
    components.html(
        f"""
        <div class="diagnostic-module-frame">{''.join(svg_parts)}</div>
        <style>
          .diagnostic-module-frame {{
            width:100%;
            overflow:hidden;
            border:1px solid #20324a;
            border-radius:14px;
            background:#020617;
          }}
          .diagnostic-module-frame svg {{
            width:100%;
            max-width:1100px;
            height:auto;
            display:block;
            margin:auto;
          }}
          .diagnostic-module-frame a {{ cursor:pointer; }}
          .diagnostic-module-frame a:hover rect {{
            opacity:1;
            stroke:#38bdf8;
            stroke-width:2;
            filter:drop-shadow(0 0 7px rgba(56,189,248,.72));
          }}
        </style>
        """,
        height=690,
        scrolling=False,
    )
    return module_image


def show_diagnostic_compare(location, selected_row_id, selected_cell_id, fault_records=None, show_boxes=True, show_cell_map=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("RGB diagnostic")
        image = close_module_diagnostic_image(location["id"], "rgb", selected_row_id, selected_cell_id, fault_records)
        image = draw_module_fault_boxes(image, location["id"], selected_row_id, selected_cell_id, fault_records, show_boxes, show_cell_map)
        st.image(image, use_container_width=True)
    with col_b:
        st.caption("Thermal diagnostic")
        image = close_module_diagnostic_image(location["id"], "thermal", selected_row_id, selected_cell_id, fault_records)
        image = draw_module_fault_boxes(image, location["id"], selected_row_id, selected_cell_id, fault_records, show_boxes, show_cell_map)
        st.image(image, use_container_width=True)


def draw_module_fault_boxes(image, location_id, selected_row_id, selected_cell_id, fault_records=None, show_boxes=True, show_cell_map=False):
    module_cells = module_cells_for_target(location_id, selected_row_id, selected_cell_id)
    rects = close_module_cell_rects(module_cells, image.size)
    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output, "RGBA")
    label_font = grid_font(12)
    if show_cell_map:
        for rect in rects.values():
            draw.rectangle(rect, outline=(186, 230, 253, 38), width=1)
    if show_boxes:
        for record in fault_records or []:
            cell_id = record.get("target_cell_id")
            if cell_id not in rects:
                continue
            rect = rects[cell_id]
            draw_corner_marks(draw, rect, (251, 146, 60, 245), width=2)
            draw_fault_label(draw, rect, short_fault_label(record.get("fault_type")), label_font, output.size)
    selected_rect = rects.get(selected_cell_id)
    if selected_rect:
        draw_corner_marks(draw, selected_rect, (56, 189, 248, 255), width=3)
    return output.convert("RGB")


def selected_cell_detail_image(module_image, location_id, selected_row_id, selected_cell_id, fault_records=None, show_boxes=True, show_cell_map=False):
    module_cells = module_cells_for_target(location_id, selected_row_id, selected_cell_id)
    selected_cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
    rects = close_module_cell_rects(module_cells, module_image.size)
    row = int(selected_cell.get("module_cell_row") or 1)
    col = int(selected_cell.get("module_cell_col") or 1)
    neighbor_ids = [
        cell["id"]
        for cell in module_cells
        if abs(int(cell.get("module_cell_row") or 1) - row) <= 1
        and abs(int(cell.get("module_cell_col") or 1) - col) <= 1
    ]
    boxes = [rects[cell_id] for cell_id in neighbor_ids if cell_id in rects] or [rects[selected_cell_id]]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad_x = max(12, int((x2 - x1) * 0.10))
    pad_y = max(12, int((y2 - y1) * 0.10))
    crop = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(module_image.width, x2 + pad_x),
        min(module_image.height, y2 + pad_y),
    )
    detail = module_image.crop(crop).convert("RGBA")
    draw = ImageDraw.Draw(detail, "RGBA")
    label_font = grid_font(11)
    for cell in module_cells:
        rect = rects[cell["id"]]
        if rect[2] < crop[0] or rect[0] > crop[2] or rect[3] < crop[1] or rect[1] > crop[3]:
            continue
        local = (rect[0] - crop[0], rect[1] - crop[1], rect[2] - crop[0], rect[3] - crop[1])
        if show_cell_map:
            draw.rectangle(local, outline=(186, 230, 253, 70), width=1)
        if cell["id"] == selected_cell_id:
            draw_corner_marks(draw, local, (56, 189, 248, 255), width=3)
    if show_boxes:
        for record in fault_records or []:
            cell_id = record.get("target_cell_id")
            if cell_id not in rects:
                continue
            rect = rects[cell_id]
            if rect[2] < crop[0] or rect[0] > crop[2] or rect[3] < crop[1] or rect[1] > crop[3]:
                continue
            local = (rect[0] - crop[0], rect[1] - crop[1], rect[2] - crop[0], rect[3] - crop[1])
            draw_corner_marks(draw, local, (251, 146, 60, 245), width=2)
            draw_fault_label(draw, local, short_fault_label(record.get("fault_type")), label_font, detail.size)
    if detail.width < 460:
        scale = min(3, max(1, int(math.ceil(460 / max(1, detail.width)))))
        detail = detail.resize((detail.width * scale, detail.height * scale), Image.Resampling.LANCZOS)
    return detail.convert("RGB")


def array_context_thumbnail(scene_image, region, selected_module):
    crop, origin, modules, _ = crop_array_from_module_union(scene_image, region, 0.12)
    thumb = crop.convert("RGBA")
    draw = ImageDraw.Draw(thumb, "RGBA")
    scale = 1
    for module_number, module in modules.items():
        display_polygon = polygon_to_crop(module_outer_polygon(module), origin, scale)
        color = (56, 189, 248, 235) if module_number == selected_module else (226, 232, 240, 92)
        width = 3 if module_number == selected_module else 1
        draw.line([tuple(point) for point in display_polygon] + [tuple(display_polygon[0])], fill=color, width=width)
    return thumb.convert("RGB")


def scale_polygon(points, scale_x, scale_y):
    return [(float(x) * scale_x, float(y) * scale_y) for x, y in points]


def scale_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def scale_region_to_image(region, from_size, to_size):
    scale_x = to_size[0] / max(1, from_size[0])
    scale_y = to_size[1] / max(1, from_size[1])
    scaled = dict(region)
    scaled["box"] = scale_box(region["box"], scale_x, scale_y)
    scaled["polygon"] = scale_polygon(region.get("polygon") or box_polygon(region["box"]), scale_x, scale_y)
    modules = {}
    for module_number, module in (region.get("modules") or array_module_polygons(region)).items():
        module_polygon = scale_polygon(module["polygon"], scale_x, scale_y)
        modules[int(module_number)] = {
            "module_number": int(module_number),
            "polygon": module_polygon,
            "box": polygon_bounds(module_polygon),
        }
    scaled["modules"] = modules
    return scaled


def close_source_scene(location_id, view, selected_row_id, selected_cell_id, fault_records=None, fallback=None):
    if view == "thermal" and reference_demo_selected():
        return reference_demo_generated(location_id)
    if view == "thermal":
        fault_records = thermal_records_for_distance(fault_records, "close", selected_row_id, selected_cell_id)
    source = cached_close_source_scene(
        location_id,
        view,
        source_fault_key(fault_records),
        solar_render_cache_signature(location_id),
    )
    if source is not None:
        return source.convert("RGB")
    return fallback.convert("RGB") if fallback is not None else None


def crop_natural_module_image(source_image, preview_size, region, selected_module, pad_ratio=0.62):
    source_region = scale_region_to_image(region, preview_size, source_image.size)
    module_geometry = (source_region.get("modules") or array_module_polygons(source_region))[int(selected_module)]
    crop, origin = crop_from_polygon(source_image, module_geometry["polygon"], pad_ratio, min_pad=32, max_pad_ratio=0.92)
    return crop, origin, source_region, module_geometry


def cell_no_for_selected(location_id, selected_row_id, selected_cell_id):
    try:
        selected_cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
        return int(selected_cell.get("module_cell_number") or 1)
    except Exception:
        return 1


def fault_cell_numbers_for_module(location_id, region, selected_module, fault_records):
    row_id = f"{row_prefix(location_id)}{region['row']}"
    faulted = {}
    for record in fault_records or []:
        if record.get("location_id") != location_id or record.get("row_id") != row_id:
            continue
        try:
            cell = cell_by_id(location_id, row_id, record.get("target_cell_id"))
        except Exception:
            continue
        if int(cell.get("panel_number") or 1) != int(region["col"]):
            continue
        if int(cell.get("module_number_in_panel") or 1) != int(selected_module):
            continue
        faulted.setdefault(int(cell.get("module_cell_number") or 1), []).append(record)
    return faulted


def render_natural_module_map(source_image, preview_size, location, view, region, selected_module, selected_row_id, selected_cell_id, fault_records=None, show_boxes=True, show_cell_map=False):
    crop, origin, source_region, module_geometry = crop_natural_module_image(source_image, preview_size, region, selected_module, 0.62)
    crop_w, crop_h = crop.size
    module_polygon = module_geometry["polygon"]
    display_module_polygon = polygon_to_crop(module_polygon, origin, 1)
    cells = module_cell_polygons(module_polygon)
    selected_cell_no = cell_no_for_selected(location["id"], selected_row_id, selected_cell_id)
    faulted = fault_cell_numbers_for_module(location["id"], region, selected_module, fault_records)
    image = crop.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = grid_font(12)

    if show_cell_map:
        for cell_geometry in cells.values():
            draw.line(
                [tuple(point) for point in polygon_to_crop(cell_geometry["polygon"], origin, 1)] + [tuple(polygon_to_crop(cell_geometry["polygon"], origin, 1)[0])],
                fill=(186, 230, 253, 46),
                width=1,
            )
    for cell_no, records in faulted.items():
        if not show_boxes:
            continue
        fault_polygon = polygon_to_crop(cells[cell_no]["polygon"], origin, 1)
        draw.line([tuple(point) for point in fault_polygon] + [tuple(fault_polygon[0])], fill=(251, 146, 60, 245), width=2)
    selected_polygon = polygon_to_crop(cells[selected_cell_no]["polygon"], origin, 1)
    draw.line([tuple(point) for point in selected_polygon] + [tuple(selected_polygon[0])], fill=(56, 189, 248, 170), width=2)

    uri = data_uri(image.convert("RGB"))
    row_number = region["row"]
    array_col = region["col"]
    array_key = region["key"]
    svg_parts = [
        f'<svg width="{crop_w}" height="{crop_h}" viewBox="0 0 {crop_w} {crop_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{crop_w}" height="{crop_h}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    for cell_no, cell_geometry in cells.items():
        display_cell_polygon = polygon_to_crop(cell_geometry["polygon"], origin, 1)
        row_id, target_cell = default_cell_for_array(location["id"], row_number, array_col, selected_module, cell_no)
        params = navigation_params(
            location["id"],
            view,
            row_id,
            target_cell["id"],
            section="Faults",
            selected_array=array_key,
            selected_module=selected_module,
            cell_active=1,
        )
        target = f"?{urlencode(params)}"
        svg_parts.append(
            f'<a href="{esc(target)}" target="_parent"><title>{esc(target_cell["label"])}</title>'
            f'<polygon points="{svg_points(display_cell_polygon)}" fill="#38bdf8" fill-opacity=".001" '
            f'stroke="#38bdf8" stroke-width="1" stroke-opacity="0" pointer-events="all"/></a>'
        )
    svg_parts.append("</svg>")
    components.html(
        f"""
        <div class="natural-module-frame">{''.join(svg_parts)}</div>
        <style>
          .natural-module-frame {{
            width:100%;
            overflow:hidden;
            border:1px solid #20324a;
            border-radius:14px;
            background:#020617;
          }}
          .natural-module-frame svg {{
            width:100%;
            max-height:72vh;
            height:auto;
            display:block;
            margin:auto;
          }}
          .natural-module-frame a {{ cursor:pointer; }}
          .natural-module-frame a:hover polygon {{
            stroke-opacity:1;
            stroke-width:2;
            filter:drop-shadow(0 0 7px rgba(56,189,248,.72));
          }}
        </style>
        """,
        height=640,
        scrolling=False,
    )
    return image.convert("RGB"), origin, source_region, module_geometry, cells


def natural_cell_detail_image(source_image, preview_size, location, region, selected_module, selected_row_id, selected_cell_id, fault_records=None, show_boxes=True, show_cell_map=False):
    source_region = scale_region_to_image(region, preview_size, source_image.size)
    module_geometry = (source_region.get("modules") or array_module_polygons(source_region))[int(selected_module)]
    cells = module_cell_polygons(module_geometry["polygon"])
    selected_no = cell_no_for_selected(location["id"], selected_row_id, selected_cell_id)
    selected_cell = cell_by_id(location["id"], selected_row_id, selected_cell_id)
    selected_row = int(selected_cell.get("module_cell_row") or 1)
    selected_col = int(selected_cell.get("module_cell_col") or 1)
    module_cells = module_cells_for_target(location["id"], selected_row_id, selected_cell_id)
    neighbor_numbers = [
        int(cell.get("module_cell_number") or 1)
        for cell in module_cells
        if abs(int(cell.get("module_cell_row") or 1) - selected_row) <= 1
        and abs(int(cell.get("module_cell_col") or 1) - selected_col) <= 1
    ]
    points = [point for no in neighbor_numbers for point in cells.get(no, cells[selected_no])["polygon"]]
    crop, origin = crop_from_polygon(source_image, points, 0.14, min_pad=14, max_pad_ratio=0.24)
    image = crop.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    faulted = fault_cell_numbers_for_module(location["id"], region, selected_module, fault_records)
    for no in neighbor_numbers:
        polygon = polygon_to_crop(cells[no]["polygon"], origin, 1)
        if show_cell_map:
            draw.line([tuple(point) for point in polygon] + [tuple(polygon[0])], fill=(186, 230, 253, 70), width=1)
        if no == selected_no:
            draw.line([tuple(point) for point in polygon] + [tuple(polygon[0])], fill=(56, 189, 248, 255), width=3)
        if show_boxes and no in faulted:
            draw.line([tuple(point) for point in polygon] + [tuple(polygon[0])], fill=(251, 146, 60, 245), width=2)
    if image.width < 460:
        scale = min(3, max(1, int(math.ceil(460 / max(1, image.width)))))
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def show_natural_compare(location, selected_row_id, selected_cell_id, preview_size, region, selected_module, fault_records=None, show_boxes=True, show_cell_map=False, rgb_fallback=None, thermal_fallback=None):
    rgb_source = rgb_fallback.convert("RGB") if rgb_fallback is not None else close_source_scene(location["id"], "rgb", selected_row_id, selected_cell_id, [], rgb_fallback)
    thermal_source = thermal_fallback.convert("RGB") if thermal_fallback is not None else close_source_scene(location["id"], "thermal", selected_row_id, selected_cell_id, fault_records, rgb_fallback)
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("RGB natural crop")
        crop, origin, source_region, module_geometry = crop_natural_module_image(rgb_source, preview_size, region, selected_module, 0.62)
        st.image(crop, use_container_width=True)
    with col_b:
        st.caption("Thermal natural crop")
        crop, origin, source_region, module_geometry = crop_natural_module_image(thermal_source, preview_size, region, selected_module, 0.62)
        st.image(crop, use_container_width=True)


def module_cells_by_number(location_id, selected_row_id):
    grouped = {}
    for cell in cells_for_row(location_id, selected_row_id):
        module_number = cell.get("module_number")
        if module_number is not None:
            grouped.setdefault(module_number, []).append(cell)
    return grouped


def panel_number_for_module(module_number):
    return (int(module_number) - 1) // SOLAR_PANEL_MODULES + 1


def module_numbers_for_panel(panel_number):
    start = (int(panel_number) - 1) * SOLAR_PANEL_MODULES + 1
    return list(range(start, start + SOLAR_PANEL_MODULES))


def cells_for_panel(location_id, selected_row_id, panel_number):
    wanted = set(module_numbers_for_panel(panel_number))
    return [
        cell for cell in cells_for_row(location_id, selected_row_id)
        if cell.get("module_number") in wanted
    ]


def crop_module_face(tile, margin=10):
    fx1, fy1, fx2, fy2 = close_module_geometry(tile.size)["frame"]
    return tile.crop((
        max(0, fx1 - margin),
        max(0, fy1 - margin),
        min(tile.width, fx2 + margin),
        min(tile.height, fy2 + margin),
    ))


def compact_fault_key(fault_records):
    return tuple(
        sorted(
            (
                str(record.get("target_cell_id", "")),
                str(canonical_fault_type(record.get("fault_type", ""))),
                int(record.get("fault_scale", DEFAULT_FAULT_SCALE)),
            )
            for record in (fault_records or [])
        )
    )


def source_fault_key(fault_records):
    return tuple(
        sorted(
            (
                str(record.get("row_id", "")),
                str(record.get("target_cell_id", "")),
                str(canonical_fault_type(record.get("fault_type", ""))),
                int(record.get("fault_scale", DEFAULT_FAULT_SCALE)),
            )
            for record in (fault_records or [])
        )
    )


def solar_render_cache_signature(location_id, size=None):
    signature = [(THERMAL_REALISM_VERSION + str(st.session_state.get("thermal_renderer_mode", "Current Thermal")), 0, 0)]
    for path in (solar_render_asset_path(location_id), solar_render_metadata_path(location_id),
                 ROOT / "pv_ui.py", ROOT / "pv_render_thermal.py", ROOT / "pv_thermal_realism.py", ROOT / "pv_real_farm.py"):
        if not path:
            signature.append(("", 0, 0))
            continue
        try:
            stat = path.stat()
            signature.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            signature.append((str(path), 0, 0))
    return (tuple(signature), tuple(size or ()))


def records_from_source_fault_key(fault_key):
    return [
        {
            "row_id": row_id,
            "target_cell_id": target_cell_id,
            "fault_type": fault_type,
            "fault_scale": fault_scale,
        }
        for row_id, target_cell_id, fault_type, fault_scale in (fault_key or ())
    ]


def stable_fault_render_target(location_id, fault_key):
    for row_id, target_cell_id, _fault_type, _fault_scale in (fault_key or ()):
        try:
            row_by_id(location_id, row_id)
            cell_by_id(location_id, row_id, target_cell_id)
            return row_id, target_cell_id
        except Exception:
            continue
    try:
        row_id = panels(location_id)[0]["id"]
        cell_id = cells_for_row(location_id, row_id)[0]["id"]
        return row_id, cell_id
    except Exception:
        return "", ""


@st.cache_data(show_spinner=False, max_entries=2)
def cached_close_source_scene(location_id, view, fault_key, render_signature):
    master = solar_render_image(location_id, None)
    if master is None:
        return None
    master = master.convert("RGB")
    if view != "thermal":
        return master
    records = records_from_source_fault_key(fault_key)
    render_row_id, render_cell_id = stable_fault_render_target(location_id, fault_key)
    base, geometries = cached_native_thermal_master(location_id, solar_render_cache_signature(location_id))
    return finalize_solar_render_thermal_image(
        location_id, base, geometries, render_row_id, render_cell_id, records, scene_distance="far"
    ).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=512)
def cached_module_preview_face(location_id, view, selected_row_id, selected_cell_id, fault_key, scene_distance="far"):
    module_cells = module_cells_for_target(location_id, selected_row_id, selected_cell_id)
    records = [
        {
            "target_cell_id": target_cell_id,
            "fault_type": fault_type,
            "fault_scale": fault_scale,
        }
        for target_cell_id, fault_type, fault_scale in fault_key
    ]
    size = (460, 275)
    if view == "thermal":
        module_img = draw_close_thermal_module(module_cells, selected_cell_id, records, size, location_id, scene_distance)
    else:
        module_img = draw_close_rgb_module(module_cells, selected_cell_id, records, size, location_id)
    return crop_module_face(module_img, margin=5)


def solar_panel_tile_image(location_id, view, selected_row_id, panel_number, selected_cell_id, fault_records=None, scene_distance="far"):
    grouped = module_cells_by_number(location_id, selected_row_id)
    selected_cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
    selected_module = selected_cell.get("module_number")
    module_numbers = module_numbers_for_panel(panel_number)
    sub_w, sub_h = (260, 154)
    gap = 10 if view == "rgb" else 7
    tile_w = sub_w * SOLAR_PANEL_MODULE_COLS + gap
    tile_h = sub_h * SOLAR_PANEL_MODULE_ROWS + gap
    fill = (9, 13, 22) if view == "rgb" else (31, 8, 45)
    panel = Image.new("RGB", (tile_w, tile_h), fill)
    draw = ImageDraw.Draw(panel, "RGBA")
    for idx, module_number in enumerate(module_numbers):
        cells = grouped.get(module_number) or []
        if not cells:
            continue
        module_records = [
            record for record in (fault_records or [])
            if cell_by_id(location_id, selected_row_id, record.get("target_cell_id")).get("module_number") == module_number
        ]
        tile_cell = selected_cell if selected_module == module_number else cells[0]
        module_img = cached_module_preview_face(
            location_id,
            view,
            selected_row_id,
            tile_cell["id"],
            compact_fault_key(module_records),
            scene_distance,
        )
        module_img = module_img.resize((sub_w, sub_h), Image.Resampling.BICUBIC)
        col = idx % SOLAR_PANEL_MODULE_COLS
        row = idx // SOLAR_PANEL_MODULE_COLS
        x = col * (sub_w + gap)
        y = row * (sub_h + gap)
        panel.paste(module_img, (x, y))
        draw.rectangle((x, y, x + sub_w - 1, y + sub_h - 1), outline=(22, 26, 34, 180), width=2)
    return panel


def module_field_tile_numbers(module_numbers, selected_module, cols=5, rows=2):
    capacity = cols * rows
    if not module_numbers:
        return []
    if selected_module not in module_numbers:
        selected_module = module_numbers[0]
    selected_index = module_numbers.index(selected_module)
    start = max(0, min(selected_index - capacity // 2, len(module_numbers) - capacity))
    return module_numbers[start:start + capacity]


@st.cache_data(show_spinner=False, max_entries=64)
def module_field_background(location_id, view, size, selected_row_id=None):
    terrain = terrain_layer(location_id).convert("RGB")
    if view == "thermal":
        terrain = thermalize(terrain, location_id, include_pv=False, scene_distance="far").convert("RGB")
    else:
        terrain = ImageEnhance.Color(terrain).enhance(0.72)
        terrain = ImageEnhance.Contrast(terrain).enhance(0.92)

    src_w, src_h = terrain.size
    dst_w, dst_h = size
    scale = max(dst_w / max(1, src_w), dst_h / max(1, src_h))
    terrain = terrain.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.BICUBIC)
    row_center_y = 0.5
    if selected_row_id is not None:
        try:
            row = row_by_id(location_id, selected_row_id)
            row_center_y = ((row["y1"] + row["y2"]) / 2) / max(1, BASE_SIZE[1])
        except Exception:
            row_center_y = 0.5
    if location_id == "agri_field_new":
        # The agri reference has a road/pad along the top; focus the synthetic array on the crop field.
        row_center_y = 0.68
    elif location_id == "grass_open":
        row_center_y = max(row_center_y, 0.56)
    elif location_id in {"rooftop_warehouse", "floating_water", "desert_track"}:
        row_center_y = 0.5
    left = max(0, (terrain.width - dst_w) // 2)
    top = int(clamp(int(terrain.height * row_center_y - dst_h / 2), 0, max(0, terrain.height - dst_h)))
    terrain = terrain.crop((left, top, left + dst_w, top + dst_h))
    return ImageEnhance.Brightness(terrain).enhance(0.78)


def module_field_static_canvas_size(location_id, view):
    """Keep the visible background stable when row/module counts change."""
    if location_id == "rooftop_warehouse":
        return (1250, 550)
    if location_id == "agri_field_new":
        return (1280, 720)
    if location_id == "floating_water":
        return (1280, 720)
    return (1280, 720)


@st.cache_data(show_spinner=False)
def load_solar_render_geometry_index(path_text, modified_ns):
    try:
        return json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def solar_render_geometry_items(location_id):
    path = ASSET_DIR / "solar-render-geometry.json"
    if not path.exists():
        return []
    try:
        index = load_solar_render_geometry_index(str(path), path.stat().st_mtime_ns)
    except OSError:
        return []
    layout_key = f"{custom_row_count(location_id)}x{custom_panels_per_row(location_id)}"
    items = (((index or {}).get("scenes") or {}).get(location_id) or {}).get(layout_key, [])
    asset = solar_render_asset_path(location_id)
    return aligned_generated_array_geometry(str(asset), tuple(json.dumps(item, sort_keys=True) for item in items)) if asset else items


@st.cache_data(show_spinner=False, max_entries=8)
def aligned_generated_array_geometry(asset_path, item_keys):
    """Remove cast shadows from the fixed Generated asset's four-module envelopes."""
    rgb = np.asarray(Image.open(asset_path).convert("RGB"))
    height, width = rgb.shape[:2]
    items = [json.loads(key) for key in item_keys]
    for item in items:
        points = np.asarray(item["polygon"], np.float32) * [width, height]
        x1, y1 = np.maximum(0, np.floor(points.min(0)).astype(int))
        x2, y2 = np.minimum([width, height], np.ceil(points.max(0)).astype(int))
        local = rgb[y1:y2, x1:x2].astype(np.float32)
        r, g, b = local[:, :, 0], local[:, :, 1], local[:, :, 2]
        faces = ((b > r * 1.10) & (b > g * .95) & (b > 22) & (local.mean(2) < 145)).astype(np.uint8)
        faces = cv2.morphologyEx(faces, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(faces, 8)
        candidates = [i for i in range(1, count) if stats[i, 4] > faces.size * .025]
        if len(candidates) != 4:
            continue  # Retain the supplied geometry if source faces are ambiguous.
        candidates = sorted(candidates, key=lambda i: centers[i, 1])
        candidates = sorted(candidates[:2], key=lambda i: centers[i, 0]) + sorted(candidates[2:], key=lambda i: centers[i, 0])
        polygons = []
        for i in candidates:
            x, y, w, h, area = stats[i]
            polygons.append([[x1+x-4,y1+y-4],[x1+x+w+3,y1+y-4],
                             [x1+x+w+3,y1+y+h+3],[x1+x-4,y1+y+h+3]])
        all_points = np.asarray(polygons).reshape(-1, 2)
        left, top = all_points.min(0); right, bottom = all_points.max(0)
        item["polygon"] = (np.asarray([[left,top],[right,top],[right,bottom],[left,bottom]]) / [width,height]).tolist()
        item["module_polygons"] = (np.asarray(polygons) / [width,height]).tolist()
    return items


def solar_render_panel_mask(rendered, location_id=None):
    geometry_items = solar_render_geometry_items(location_id) if location_id else []
    if geometry_items:
        width, height = rendered.size
        mask = np.zeros((height, width), dtype=np.uint8)
        for item in geometry_items:
            for quad in item.get("module_polygons", [item.get("polygon", [])]):
                polygon = np.rint(np.asarray(quad) * [width, height]).astype(np.int32)
                if polygon.shape == (4, 2):
                    cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
        return cv2.GaussianBlur(mask, (0, 0), 0.75)

    # Compatibility fallback for older asset bundles without geometry metadata.
    rgb = np.asarray(rendered.convert("RGB")).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = rgb.mean(axis=2)
    vegetation_green = (2.0 * g) - r - b
    dark_hardware = (gray < 118) & (vegetation_green < 38) & (b > r * 0.88) & (g < 126)
    mask = dark_hardware.astype(np.uint8) * 255
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_mid = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mid)
    return cv2.GaussianBlur(mask, (0, 0), 1.2)


def solar_render_array_boxes(panel_mask, rows, cols):
    expected = max(1, rows * cols)
    binary = (panel_mask > 72).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(60, int(binary.size * 0.00035))
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h), int(area), centroids[label]))

    if len(boxes) >= expected:
        boxes = sorted(boxes, key=lambda item: item[4], reverse=True)[:expected]
        boxes = sorted(boxes, key=lambda item: item[5][1])
        grouped = []
        for row_idx in range(rows):
            group = boxes[row_idx * cols:(row_idx + 1) * cols]
            grouped.extend(sorted(group, key=lambda item: item[5][0]))
        return {
            (idx // cols + 1, idx % cols + 1): box[:4]
            for idx, box in enumerate(grouped)
        }

    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        return {}
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    box_w = max(1, (right - left) / max(1, cols))
    box_h = max(1, (bottom - top) / max(1, rows))
    fallback = {}
    for row_idx in range(rows):
        for col_idx in range(cols):
            x1 = int(left + col_idx * box_w)
            x2 = int(left + (col_idx + 1) * box_w)
            y1 = int(top + row_idx * box_h)
            y2 = int(top + (row_idx + 1) * box_h)
            fallback[(row_idx + 1, col_idx + 1)] = (x1, y1, x2, y2)
    return fallback


def solar_render_metadata_array_boxes(panel_mask, metadata):
    transforms = ((metadata or {}).get("topology") or {}).get("array_transforms") or []
    row_strings = ((metadata or {}).get("topology") or {}).get("row_strings") or []
    if not transforms or not row_strings:
        return {}
    binary = (panel_mask > 72).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(60, int(binary.size * 0.00035))
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h), int(area), centroids[label]))
    if len(boxes) < len(transforms):
        return {}
    boxes = sorted(boxes, key=lambda item: item[4], reverse=True)[:len(transforms)]
    boxes = sorted(boxes, key=lambda item: item[5][1])
    result = {}
    cursor = 0
    for row in sorted(row_strings, key=lambda item: item.get("row_id", 0)):
        row_id = int(row.get("row_id", 0))
        row_count = int(row.get("count", 0))
        group = sorted(boxes[cursor:cursor + row_count], key=lambda item: item[5][0])
        cursor += row_count
        row_transforms = sorted(
            [item for item in transforms if int(item.get("row", 0)) == row_id],
            key=lambda item: int(item.get("col", 0)),
        )
        for transform, box in zip(row_transforms, group):
            result[(int(transform.get("row", 0)), int(transform.get("col", 0)))] = box[:4]
    return result


def polygon_bounds(points):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def box_polygon(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]
    return [
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    ]


def quad_point(quad, u, v):
    tl, tr, br, bl = [np.asarray(point, dtype=np.float32) for point in quad]
    point = (1 - u) * (1 - v) * tl + u * (1 - v) * tr + u * v * br + (1 - u) * v * bl
    return (float(point[0]), float(point[1]))


def quad_subpolygon(quad, u1, v1, u2, v2):
    return [
        quad_point(quad, u1, v1),
        quad_point(quad, u2, v1),
        quad_point(quad, u2, v2),
        quad_point(quad, u1, v2),
    ]


MODULE_ACTIVE_INSET_U = 0.060
MODULE_ACTIVE_INSET_V = 0.085
CELL_SAFE_INSET_U = 0.16
CELL_SAFE_INSET_V = 0.18


def polygon_centroid(points):
    pts = list(points or [])
    if not pts:
        return (0.0, 0.0)
    return (
        float(sum(float(point[0]) for point in pts) / len(pts)),
        float(sum(float(point[1]) for point in pts) / len(pts)),
    )


def point_in_polygon(point, polygon):
    x, y = float(point[0]), float(point[1])
    inside = False
    pts = list(polygon or [])
    if len(pts) < 3:
        return False
    j = len(pts) - 1
    for i, pi in enumerate(pts):
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(pts[j][0]), float(pts[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def active_module_polygon(module_polygon):
    return quad_subpolygon(
        module_polygon,
        MODULE_ACTIVE_INSET_U,
        MODULE_ACTIVE_INSET_V,
        1.0 - MODULE_ACTIVE_INSET_U,
        1.0 - MODULE_ACTIVE_INSET_V,
    )


def module_outer_polygon(module):
    return module.get("outer_polygon") or module.get("polygon") or box_polygon(module["box"])


def module_active_polygon(module):
    return module.get("active_polygon") or active_module_polygon(module_outer_polygon(module))


def region_modules(region):
    return region.get("modules") or array_module_polygons(region)


def array_module_union_points(region):
    points = []
    for module in region_modules(region).values():
        points.extend(module_outer_polygon(module))
    return points or list(region.get("polygon") or box_polygon(region["box"]))


def crop_array_from_module_union(image, region, pad_ratio=0.12, target_width=None, max_scale=3):
    union_points = array_module_union_points(region)
    crop, origin = crop_from_polygon(image, union_points, pad_ratio, min_pad=10, max_pad_ratio=0.24)
    scale = 1
    if target_width:
        scale = max(1, min(max_scale, math.ceil(target_width / max(1, crop.width))))
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.BICUBIC)
    return crop, origin, region_modules(region), scale


def polygon_to_crop(points, crop_origin, scale=1):
    ox, oy = crop_origin[0], crop_origin[1]
    return [((float(x) - ox) * scale, (float(y) - oy) * scale) for x, y in points]


def svg_points(points):
    return " ".join(f"{float(x):.1f},{float(y):.1f}" for x, y in points)


def component_geometries_from_mask(panel_mask):
    binary = (panel_mask > 72).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(60, int(binary.size * 0.00035))
    components = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        component = ((labels == label).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            polygon = order_quad_points(cv2.boxPoints(rect))
        else:
            polygon = box_polygon((x, y, x + w, y + h))
        components.append(
            {
                "box": (int(x), int(y), int(x + w), int(y + h)),
                "polygon": polygon,
                "area": int(area),
                "centroid": (float(centroids[label][0]), float(centroids[label][1])),
            }
        )
    return components


def solar_render_metadata_array_geometries(panel_mask, metadata):
    location_id = ((metadata or {}).get("composite") or {}).get("location_id", "grass_open")
    items = solar_render_geometry_items(location_id)
    if items:
        height, width = panel_mask.shape
        result = {}
        for item in items:
            polygon = (np.asarray(item["polygon"]) * [width, height]).tolist()
            modules = {}
            quads = item.get("module_polygons")
            if not quads:
                fallback = {"box": polygon_bounds(polygon), "polygon": polygon}
                quads = [(np.asarray(module_outer_polygon(m)) / [width, height]).tolist()
                         for m in array_module_polygons(fallback).values()]
            for number, quad in enumerate(quads, 1):
                outer = (np.asarray(quad) * [width, height]).tolist()
                active = active_module_polygon(outer)
                modules[number] = {"module_number": number, "outer_polygon": outer, "polygon": outer,
                                   "active_polygon": active, "box": polygon_bounds(outer), "active_box": polygon_bounds(active)}
            result[(int(item["row"]), int(item["col"]))] = {"box": polygon_bounds(polygon), "polygon": polygon,
                                                              "modules": modules, "orientation": "landscape"}
        return result
    transforms = ((metadata or {}).get("topology") or {}).get("array_transforms") or []
    row_strings = ((metadata or {}).get("topology") or {}).get("row_strings") or []
    components = component_geometries_from_mask(panel_mask)
    if not transforms or not row_strings or len(components) < len(transforms):
        return {}

    components = sorted(components, key=lambda item: item["area"], reverse=True)[:len(transforms)]
    components = sorted(components, key=lambda item: item["centroid"][1])
    result = {}
    cursor = 0
    for row in sorted(row_strings, key=lambda item: item.get("row_id", 0)):
        row_id = int(row.get("row_id", 0))
        row_count = int(row.get("count", 0))
        group = sorted(components[cursor:cursor + row_count], key=lambda item: item["centroid"][0])
        cursor += row_count
        row_transforms = sorted(
            [item for item in transforms if int(item.get("row", 0)) == row_id],
            key=lambda item: int(item.get("col", 0)),
        )
        for transform, component in zip(row_transforms, group):
            key = (int(transform.get("row", 0)), int(transform.get("col", 0)))
            result[key] = {
                **component,
                "orientation": transform.get("orientation", "landscape"),
                "array_id": int(transform.get("array_id", 0)),
                "string_id": transform.get("string_id"),
                "transform": transform,
            }
    return result


def array_module_polygons(region):
    if region.get("modules"):
        return region["modules"]
    array_polygon = region.get("polygon") or box_polygon(region["box"])
    inset = 0.065
    module_gap = 0.014
    modules = {}
    for module_number, (u1, v1, u2, v2) in {
        1: (inset, inset, 0.5 - module_gap, 0.5 - module_gap),
        2: (0.5 + module_gap, inset, 1.0 - inset, 0.5 - module_gap),
        3: (inset, 0.5 + module_gap, 0.5 - module_gap, 1.0 - inset),
        4: (0.5 + module_gap, 0.5 + module_gap, 1.0 - inset, 1.0 - inset),
    }.items():
        polygon = quad_subpolygon(array_polygon, u1, v1, u2, v2)
        active_polygon = active_module_polygon(polygon)
        modules[module_number] = {
            "module_number": module_number,
            "orientation": region.get("orientation", "unknown"),
            "outer_polygon": polygon,
            "polygon": polygon,
            "active_polygon": active_polygon,
            "box": polygon_bounds(polygon),
            "active_box": polygon_bounds(active_polygon),
        }
    return modules


def module_cell_polygons(module_geometry):
    cells = {}
    cols = CUSTOM_MODULE_CELL_COLS
    rows = CUSTOM_MODULE_CELL_ROWS
    if isinstance(module_geometry, dict):
        active_polygon = module_active_polygon(module_geometry)
    else:
        active_polygon = active_module_polygon(module_geometry)
    for r in range(rows):
        for c in range(cols):
            cell_no = r * cols + c + 1
            u1, u2 = c / cols, (c + 1) / cols
            v1, v2 = r / rows, (r + 1) / rows
            polygon = quad_subpolygon(active_polygon, u1, v1, u2, v2)
            safe_polygon = quad_subpolygon(
                active_polygon,
                u1 + (u2 - u1) * CELL_SAFE_INSET_U,
                v1 + (v2 - v1) * CELL_SAFE_INSET_V,
                u2 - (u2 - u1) * CELL_SAFE_INSET_U,
                v2 - (v2 - v1) * CELL_SAFE_INSET_V,
            )
            centroid = polygon_centroid(polygon)
            safe_centroid = polygon_centroid(safe_polygon)
            assert point_in_polygon(centroid, active_polygon)
            assert point_in_polygon(safe_centroid, active_polygon)
            cells[cell_no] = {
                "cell_number": cell_no,
                "polygon": polygon,
                "safe_fault_polygon": safe_polygon,
                "active_polygon": active_polygon,
                "box": polygon_bounds(polygon),
                "safe_box": polygon_bounds(safe_polygon),
                "centroid": centroid,
                "safe_centroid": safe_centroid,
            }
    return cells


def crop_from_polygon(image, polygon, pad_ratio=0.08, min_pad=6, max_pad_ratio=0.18):
    box = polygon_bounds(polygon)
    crop, origin = crop_with_padding(image, box, pad_ratio, min_pad=min_pad, max_pad_ratio=max_pad_ratio)
    ox1, oy1, ox2, oy2 = origin
    tolerance = 1.0
    image_w, image_h = image.size
    visible_points = [
        (x, y)
        for x, y in polygon
        if -tolerance <= x <= image_w + tolerance and -tolerance <= y <= image_h + tolerance
    ]
    assert all(
        ox1 - tolerance <= x <= ox2 + tolerance and oy1 - tolerance <= y <= oy2 + tolerance
        for x, y in visible_points
    )
    return crop, origin


def row_number_from_id(location_id, row_id):
    try:
        return int(str(row_id).replace(row_prefix(location_id), ""))
    except (TypeError, ValueError):
        return 1


def rendered_fault_cell_from_record(location_id, record, selected_row_id, selected_cell_id, array_geometries):
    record_row_id = record.get("row_id", selected_row_id)
    record_row = row_number_from_id(location_id, record_row_id)
    record_cell = cell_by_id(location_id, record_row_id, record.get("target_cell_id", selected_cell_id))
    record_col = int(record_cell.get("panel_number") or 1)
    array_geometry = array_geometries.get((record_row, record_col))
    if not array_geometry:
        return None

    module_number = int(record_cell.get("module_number_in_panel") or 1)
    module_number = max(1, min(SOLAR_PANEL_MODULES, module_number))
    region = array_geometry
    module_geometry = array_module_polygons(region)[module_number]
    cell_row = int(record_cell.get("module_cell_row") or 1)
    cell_col = int(record_cell.get("module_cell_col") or 1)
    cell_row = max(1, min(CUSTOM_MODULE_CELL_ROWS, cell_row))
    cell_col = max(1, min(CUSTOM_MODULE_CELL_COLS, cell_col))
    cell_no = (cell_row - 1) * CUSTOM_MODULE_CELL_COLS + cell_col
    cell_geometry = module_cell_polygons(module_geometry)[cell_no]
    cx1, cy1, cx2, cy2 = cell_geometry["safe_box"]
    mx1, my1, mx2, my2 = module_geometry["box"]
    ax1, ay1, ax2, ay2 = polygon_bounds(cell_geometry["active_polygon"])

    return {
        **record_cell,
        "id": record.get("target_cell_id", record_cell["id"]),
        "x1": cx1,
        "y1": cy1,
        "x2": cx2,
        "y2": cy2,
        "source_polygon": cell_geometry["polygon"],
        "active_module_polygon": cell_geometry["active_polygon"],
        "safe_fault_polygon": cell_geometry["safe_fault_polygon"],
        "fault_origin_px": {
            "x": float(cell_geometry["safe_centroid"][0]),
            "y": float(cell_geometry["safe_centroid"][1]),
        },
        "active_module_bbox_px": {
            "x1": int(ax1),
            "y1": int(ay1),
            "x2": int(ax2),
            "y2": int(ay2),
        },
        "module_bbox_px": {
            "x1": int(mx1),
            "y1": int(my1),
            "x2": int(mx2),
            "y2": int(my2),
        },
        "fault_bbox_px": {
            "x1": int(cx1),
            "y1": int(cy1),
            "x2": int(cx2),
            "y2": int(cy2),
        },
    }


def smooth_thermal_array_field(size, seed, panel_min, panel_max):
    """Compatibility entry point for the shared healthy thermal model."""
    return healthy_thermal_module_field(size, seed, panel_min, panel_max)


def solar_render_thermal_base_field(location_id, rendered):
    temp = thermal_background_realism(
        np.empty((rendered.height, rendered.width), np.float32), rendered, location_id
    )
    background_scalar = temp.copy() if st.session_state.get("thermal_renderer_mode") != "Legacy Thermal (Fallback)" else None
    scene_drift = thermal_scene_drift(temp.shape, seed_for(location_id, "healthy-thermal-scene"))

    panel_mask = solar_render_panel_mask(rendered, location_id).astype(np.float32) / 255.0
    panel_mask = np.clip(panel_mask * 1.18, 0.0, 0.98)
    metadata = load_solar_render_metadata(location_id)
    array_geometries = solar_render_metadata_array_geometries((panel_mask * 255).astype(np.uint8), metadata)
    if not array_geometries:
        fallback_boxes = solar_render_array_boxes(
            (panel_mask * 255).astype(np.uint8),
            custom_row_count(location_id),
            custom_panels_per_row(location_id),
        )
        array_geometries = {
            key: {"box": box, "polygon": box_polygon(box)}
            for key, box in fallback_boxes.items()
        }

    panel_min, panel_max = thermal_pv_operating_range(location_id)

    healthy_module_support = np.zeros(temp.shape, np.uint8)
    healthy_module_geometry = []
    source_gray = cv2.cvtColor(np.asarray(rendered.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    for key, geometry in array_geometries.items():
        # Geometry supplies physical modules only. There is no filled table
        # rectangle and no thermal subdivision painted inside any module.
        region = geometry
        for number, module in array_module_polygons(region).items():
            cv2.fillConvexPoly(healthy_module_support, np.rint(module_outer_polygon(module)).astype(np.int32), 255)
            healthy_module_geometry.append(module_outer_polygon(module))
            composite_healthy_thermal_module(
                temp, module_outer_polygon(module), scene_drift,
                seed_for(location_id, key, number, "healthy-module"),
                panel_min, panel_max, panel_mask, source_gray=source_gray,
            )

    temp = thermal_boundary_response(
        temp, healthy_module_support, seed_for(location_id, "healthy-boundary-response"),
        location_id=location_id, source_image=rendered,
        geometries=[geometry["polygon"] for geometry in array_geometries.values()],
        module_geometries=healthy_module_geometry,
    )
    temp = thermal_refine_background_materials(temp, rendered, location_id, healthy_module_geometry)
    if background_scalar is not None:
        for start in range(0, temp.shape[0], 128):
            rows = slice(start, start + 128)
            outside = healthy_module_support[rows] == 0
            temp[rows][outside] = background_scalar[rows][outside]
    return temp.astype(np.float32), array_geometries


def thermal_display_rgb_from_scalar(temp, location_id, scene_distance="far", display_range=None):
    """Map one scalar thermal master to the app palette with a stable robust range."""
    temp = np.asarray(temp, dtype=np.float32)
    if temp.size == 0:
        return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    # Same relative scalar calibration for healthy/faulted and far/close. A
    # provided measured-scene range remains authoritative. Crops never autoscale.
    calibration = THERMAL_SCALAR_DISPLAY_RANGE if display_range is None else display_range
    low, high = (float(v) for v in calibration)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Invalid thermal display calibration")
    rng = np.random.default_rng(seed_for(location_id, "thermal-dither", scene_distance))
    lut = np.array(thermal_lut(), dtype=np.uint8)
    rgb = np.empty((*temp.shape, 3), dtype=np.uint8)
    for start in range(0, temp.shape[0], 128):
        rows = slice(start, start + 128)
        # Sequential row draws retain the original scene-space RNG stream.
        dither = rng.uniform(-0.45, 0.45, temp[rows].shape).astype(np.float32)
        temp_u8 = np.clip((temp[rows] + dither - low) / max(1.0, high - low) * 255.0, 0, 255).astype(np.uint8)
        rgb[rows] = lut[temp_u8]
    return Image.fromarray(rgb)


def scalar_debug_image(field, mode="thermal", location_id="debug"):
    field = np.asarray(field, dtype=np.float32)
    if field.size == 0:
        return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    if mode == "thermal":
        return thermal_display_rgb_from_scalar(field, location_id, "close")
    high = float(np.percentile(np.abs(field), 99.2))
    if high <= 1e-6:
        high = 1.0
    normalized = np.clip(np.abs(field) / high * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, "L").convert("RGB")


def live_fault_scalar_debug(location_id, selected_row_id, selected_cell_id, fault_records=None, scene_distance="close"):
    render_size = module_field_static_canvas_size(location_id, "thermal")
    rendered = solar_render_image(location_id, render_size)
    if rendered is None:
        return None
    base_temp, array_geometries = solar_render_thermal_base_field(location_id, rendered.convert("RGB"))
    selected_record = None
    for record in fault_records or []:
        if (
            record.get("location_id") == location_id
            and record.get("row_id") == selected_row_id
            and record.get("target_cell_id") == selected_cell_id
        ):
            selected_record = record
            break
    if selected_record is None:
        return None
    rendered_cell = rendered_fault_cell_from_record(
        location_id,
        selected_record,
        selected_row_id,
        selected_cell_id,
        array_geometries,
    )
    if rendered_cell is None:
        return None
    healthy = np.array(base_temp, dtype=np.float32, copy=True)
    faulted = np.array(base_temp, dtype=np.float32, copy=True)
    before = np.array(faulted, dtype=np.float32, copy=True)
    apply_thermal_fault(
        faulted,
        rendered_cell,
        selected_record["fault_type"],
        selected_record.get("fault_scale", DEFAULT_FAULT_SCALE),
    )
    delta = faulted - before
    applied = bool(float(np.max(np.abs(delta))) > 0.01)
    mx1 = int(rendered_cell.get("module_bbox_px", {}).get("x1", rendered_cell.get("x1", 0)))
    my1 = int(rendered_cell.get("module_bbox_px", {}).get("y1", rendered_cell.get("y1", 0)))
    mx2 = int(rendered_cell.get("module_bbox_px", {}).get("x2", rendered_cell.get("x2", 0)))
    my2 = int(rendered_cell.get("module_bbox_px", {}).get("y2", rendered_cell.get("y2", 0)))
    h, w = healthy.shape
    pad = int(max(12, max(mx2 - mx1, my2 - my1) * 0.18))
    x1, y1 = max(0, mx1 - pad), max(0, my1 - pad)
    x2, y2 = min(w, mx2 + pad), min(h, my2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = (x1, y1, x2, y2)
    support = (np.abs(delta) > 0.75).astype(np.float32) * 255.0
    support_img = Image.fromarray(np.clip(support[y1:y2, x1:x2], 0, 255).astype(np.uint8), "L").convert("RGB")
    support_bounds = None
    ys, xs = np.where(support > 0)
    if len(xs):
        support_bounds = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max() + 1),
            "y2": int(ys.max() + 1),
        }
    return {
        "record": dict(selected_record),
        "applied": bool(applied),
        "rendered_cell": rendered_cell,
        "support_bounds": support_bounds,
        "healthy_scalar": scalar_debug_image(healthy[y1:y2, x1:x2], "thermal", location_id),
        "faulted_scalar": scalar_debug_image(faulted[y1:y2, x1:x2], "thermal", location_id),
        "delta": scalar_debug_image(delta[y1:y2, x1:x2], "delta", location_id),
        "support": support_img,
        "final_palette": thermal_display_rgb_from_scalar(
            apply_sensor_model(faulted, location_id, scene_distance),
            location_id,
            scene_distance,
        ).crop(crop),
        "metrics": scalar_fault_delta_metrics(healthy, faulted, rendered_cell, support),
    }


def scalar_fault_delta_metrics(healthy, faulted, rendered_cell, support_mask=None):
    healthy = np.asarray(healthy, dtype=np.float32)
    faulted = np.asarray(faulted, dtype=np.float32)
    delta = faulted - healthy
    module_box = rendered_cell.get("module_bbox_px", {}) if isinstance(rendered_cell, dict) else {}
    x1 = int(module_box.get("x1", rendered_cell.get("x1", 0)))
    y1 = int(module_box.get("y1", rendered_cell.get("y1", 0)))
    x2 = int(module_box.get("x2", rendered_cell.get("x2", healthy.shape[1])))
    y2 = int(module_box.get("y2", rendered_cell.get("y2", healthy.shape[0])))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(healthy.shape[1], x2), min(healthy.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, healthy.shape[1], healthy.shape[0]
    healthy_roi = healthy[y1:y2, x1:x2]
    faulted_roi = faulted[y1:y2, x1:x2]
    delta_roi = delta[y1:y2, x1:x2]
    support_roi = None
    if support_mask is not None:
        support_roi = np.asarray(support_mask[y1:y2, x1:x2]) > 0
    if support_roi is None or not np.any(support_roi):
        support_roi = np.abs(delta_roi) > 0.75
    outside = ~support_roi
    return {
        "healthy_min": float(np.min(healthy_roi)),
        "healthy_max": float(np.max(healthy_roi)),
        "healthy_mean": float(np.mean(healthy_roi)),
        "faulted_min": float(np.min(faulted_roi)),
        "faulted_max": float(np.max(faulted_roi)),
        "faulted_mean": float(np.mean(faulted_roi)),
        "max_abs_delta": float(np.max(np.abs(delta_roi))) if delta_roi.size else 0.0,
        "mean_delta_inside_support": float(np.mean(delta_roi[support_roi])) if np.any(support_roi) else 0.0,
        "mean_delta_outside_support": float(np.mean(delta_roi[outside])) if np.any(outside) else 0.0,
        "support_pixel_count": int(np.count_nonzero(support_roi)),
    }


def close_fault_visibility_cell(cell, fault_type):
    """Expand only the close-view display footprint while keeping the stored target cell unchanged."""
    fault_type = canonical_fault_type(fault_type)
    item = dict(cell)
    module_box = item.get("active_module_bbox_px") or item.get("module_bbox_px") or item.get("fault_bbox_px") or {}
    mx1 = int(module_box.get("x1", item["x1"]))
    my1 = int(module_box.get("y1", item["y1"]))
    mx2 = int(module_box.get("x2", item["x2"]))
    my2 = int(module_box.get("y2", item["y2"]))
    cx1, cy1, cx2, cy2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
    cell_w, cell_h = max(2, cx2 - cx1), max(2, cy2 - cy1)

    if fault_type in SOILING_FAULTS:
        bounds = {"x1": mx1, "y1": my1, "x2": mx2, "y2": my2}
        item.update(bounds)
        item["fault_bbox_px"] = dict(bounds)
        return item
    elif fault_type in SURFACE_OBSTRUCTION_FAULTS:
        pad_x, pad_y = 1.35 * cell_w, 1.65 * cell_h
    elif fault_type in CRACKING_FAULTS:
        pad_x, pad_y = 0.35 * cell_w, 0.45 * cell_h
    else:
        return item

    bounds = {
        "x1": int(clamp(round(cx1 - pad_x), mx1, mx2)),
        "y1": int(clamp(round(cy1 - pad_y), my1, my2)),
        "x2": int(clamp(round(cx2 + pad_x), mx1, mx2)),
        "y2": int(clamp(round(cy2 + pad_y), my1, my2)),
    }
    item.update(bounds)
    item["fault_bbox_px"] = dict(bounds)
    return item


def finalize_solar_render_thermal_image(location_id, base_temp, array_geometries, selected_row_id, selected_cell_id, fault_records=None, scene_distance="far", acquisition_size=None):
    temp = np.array(base_temp, dtype=np.float32, copy=True)
    records = thermal_records_for_distance(fault_records, scene_distance)
    fault_support = np.zeros(temp.shape, np.uint8) if records else None
    for record in records:
        cell = rendered_fault_cell_from_record(location_id, record, selected_row_id, selected_cell_id, array_geometries)
        if cell is None:
            continue
        fault_type = canonical_fault_type(record["fault_type"])
        # Resolve bank morphology in the physical module plane, then project its
        # scalar residual through the same active polygon used by selection.
        # Axis-aligned cell boxes distort diagonal modules and spill onto roofs.
        polygon = cell["active_module_polygon"]
        cv2.fillConvexPoly(fault_support, np.rint(polygon).astype(np.int32), 255, lineType=cv2.LINE_8)
        module = {"module_id": f"{location_id}:{record.get('row_id')}:{cell.get('module_number')}",
                  "active_polygon_rgb": polygon, "outer_polygon_rgb": polygon,
                  "canonical_size": real_farm_canonical_size(polygon),
                  "mask_source": "MANUAL_FINAL_POLYGON", "refinement_status": "accepted"}
        observation = {"observation_id": record.get("target_cell_id"), "module_id": module["module_id"],
                       "severity": int(record.get("fault_scale", DEFAULT_FAULT_SCALE)),
                       "local_uv": [(int(cell.get("module_cell_col") or 1)-0.5)/CUSTOM_MODULE_CELL_COLS,
                                    (int(cell.get("module_cell_row") or 1)-0.5)/CUSTOM_MODULE_CELL_ROWS]}
        temp = real_farm_apply_production_fault(temp, module, observation, fault_type, in_place=True)

    pv_support = np.zeros(temp.shape, np.uint8)
    for geometry in array_geometries.values():
        for module in array_module_polygons(geometry).values():
            cv2.fillConvexPoly(pv_support, np.rint(module_outer_polygon(module)).astype(np.int32), 255)
    temp = apply_sensor_model(temp, location_id, "far", pv_support)
    if records:
        healthy_acquisition = apply_sensor_model(base_temp, location_id, "far", pv_support)
        for start in range(0, temp.shape[0], 128):
            rows = slice(start, start + 128)
            outside = fault_support[rows] == 0
            temp[rows][outside] = healthy_acquisition[rows][outside]
        del healthy_acquisition, fault_support
    if acquisition_size:
        height,width=temp.shape
        ratio=min(1.,acquisition_size[0]/width,acquisition_size[1]/height)
        if ratio<1.:
            temp=cv2.resize(temp,(max(1,round(width*ratio)),max(1,round(height*ratio))),interpolation=cv2.INTER_AREA)
    # Spatial detector integration precedes the fixed shared level/span and LUT.
    thermal_image = thermal_display_rgb_from_scalar(temp, location_id, "far")
    return thermal_image


def solar_render_thermal_image(location_id, rendered, selected_row_id, selected_cell_id, fault_records=None, scene_distance="far"):
    base_temp, array_geometries = solar_render_thermal_base_field(location_id, rendered)
    return finalize_solar_render_thermal_image(
        location_id,
        base_temp,
        array_geometries,
        selected_row_id,
        selected_cell_id,
        fault_records,
        scene_distance,
    )


def draw_solar_render_overlays(image, location_id, selected_row_id, selected_cell_id, fault_records=None, show_grid=False):
    if not show_grid:
        return image
    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output, "RGBA")
    mask = solar_render_panel_mask(image, location_id)
    metadata = load_solar_render_metadata(location_id)
    geometries = solar_render_metadata_array_geometries(mask, metadata)
    if not geometries:
        fallback_boxes = solar_render_array_boxes(mask, custom_row_count(location_id), custom_panels_per_row(location_id))
        geometries = {key: {"box": box, "polygon": box_polygon(box)} for key, box in fallback_boxes.items()}

    if show_grid:
        selected_row_num = row_number_from_id(location_id, selected_row_id)
        selected_cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
        selected_panel = int(selected_cell.get("panel_number") or 1)
        for key, geometry in geometries.items():
            color = (70, 180, 255, 68)
            width = 1
            if key == (selected_row_num, selected_panel):
                color = (250, 204, 21, 210)
                width = 3
            draw.line(
                [tuple(point) for point in geometry["polygon"]] + [tuple(geometry["polygon"][0])],
                fill=color,
                width=width,
            )

    return output.convert("RGB")


@st.cache_data(show_spinner=False, max_entries=2)
def cached_native_thermal_master(location_id, render_signature):
    """Authoritative Generated healthy scalar scene at source resolution."""
    rendered = solar_render_image(location_id, None)
    if rendered is None:
        return None
    return solar_render_thermal_base_field(location_id, rendered)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_solar_render_thermal_base(location_id, render_size, render_signature):
    master = cached_native_thermal_master(location_id, solar_render_cache_signature(location_id))
    if master is None:
        return None
    scalar, geometries = master
    if not render_size:
        return scalar, geometries
    height, width = scalar.shape
    ratio = min(float(render_size[0]) / width, float(render_size[1]) / height)
    size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    sx, sy = size[0] / width, size[1] / height
    scaled = {
        key: {**geometry, "box": scale_box(geometry["box"], sx, sy),
              "polygon": scale_polygon(geometry["polygon"], sx, sy)}
        for key, geometry in geometries.items()
    }
    return cv2.resize(scalar, size, interpolation=cv2.INTER_AREA), scaled


@st.cache_data(show_spinner=False, max_entries=48)
def cached_module_field_thermal_scene(location_id, render_size, fault_key, render_signature, scene_distance="far"):
    cached_base = cached_native_thermal_master(location_id, solar_render_cache_signature(location_id))
    if cached_base is None:return None
    base_temp,array_geometries=cached_base
    records=records_from_source_fault_key(fault_key)
    render_row_id,render_cell_id=stable_fault_render_target(location_id,fault_key)
    return finalize_solar_render_thermal_image(location_id,base_temp,array_geometries,
        render_row_id,render_cell_id,records,scene_distance,acquisition_size=render_size).convert("RGB")


def module_field_image(location_id, view, selected_row_id, selected_cell_id, fault_records=None, show_grid=True, scene_distance="far"):
    if view == "thermal" and reference_demo_selected():
        thermal = reference_demo_generated(location_id, module_field_static_canvas_size(location_id, view))
        return draw_solar_render_overlays(thermal, location_id, selected_row_id, selected_cell_id, [], show_grid)
    if view == "thermal":
        fault_records = thermal_records_for_distance(fault_records, scene_distance, selected_row_id, selected_cell_id)
    render_size = module_field_static_canvas_size(location_id, view)
    rendered = None if view == "thermal" else solar_render_image(location_id, render_size)
    if rendered is not None and view == "rgb":
        return draw_solar_render_overlays(rendered, location_id, selected_row_id, selected_cell_id, fault_records, show_grid)
    if view == "thermal":
        # Inject the same native scalar fault as Close, integrate the Far
        # detector plane, then palette-map. Never filter a colorized hotspot.
        thermal=cached_module_field_thermal_scene(location_id,render_size,source_fault_key(fault_records),
            solar_render_cache_signature(location_id),"far")
        if thermal is not None:
            return draw_solar_render_overlays(thermal,location_id,selected_row_id,selected_cell_id,fault_records,show_grid)
        rendered = solar_render_image(location_id, render_size)
    if rendered is not None and view == "thermal":
        thermal = solar_render_thermal_image(location_id, rendered, selected_row_id, selected_cell_id, fault_records, scene_distance)
        return draw_solar_render_overlays(thermal, location_id, selected_row_id, selected_cell_id, fault_records, show_grid)

    all_rows = panels(location_id)
    selected_cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
    selected_module = selected_cell.get("module_number")
    selected_panel = selected_cell.get("panel_number") or panel_number_for_module(selected_module or 1)
    panel_count = custom_panels_per_row(location_id) if custom_layout_enabled() else default_panels_per_row(location_id)
    row_count = len(all_rows)
    max_visible_panels = max(1, panel_count)
    max_visible_rows = max(1, row_count)

    selected_row_index = next((idx for idx, row in enumerate(all_rows) if row["id"] == selected_row_id), 0)
    visible_rows = all_rows[:max_visible_rows]
    if row_count > max_visible_rows:
        start_row = max(0, min(selected_row_index - max_visible_rows // 2, row_count - max_visible_rows))
        visible_rows = all_rows[start_row:start_row + max_visible_rows]

    selected_panel_index = int(selected_panel or 1) - 1
    start_panel = 0
    if panel_count > max_visible_panels:
        start_panel = max(0, min(selected_panel_index - max_visible_panels // 2, panel_count - max_visible_panels))
    visible_panels = list(range(start_panel + 1, start_panel + max_visible_panels + 1))

    if not visible_rows or not visible_panels:
        return close_module_diagnostic_image(location_id, view, selected_row_id, selected_cell_id, fault_records)

    tile_w, tile_h = 150, 104
    if location_id == "rooftop_warehouse":
        tile_w, tile_h = 140, 98
    if max_visible_panels >= 7:
        tile_w, tile_h = 132, 92
    if max_visible_panels >= 10:
        tile_w, tile_h = 112, 78
    gap_x, gap_y = 22, 34
    pad_x, pad_y = 62, 36
    if location_id == "rooftop_warehouse":
        pad_x = 86
    if location_id == "agri_field_new":
        tile_w, tile_h = 128, 90
        gap_x, gap_y = 38, 48
        pad_x, pad_y = 164, 270
    if location_id == "desert_track":
        # Desert reference has a large prepared pad on the right side; keep arrays there,
        # with enough right-side margin when the user selects the maximum panel count.
        tile_w, tile_h = 124, 88
        gap_x, gap_y = 22, 54
        pad_x, pad_y = 390, 190
    if location_id == "floating_water":
        # Keep offshore PV panels inside the inner raft boundary.
        tile_w, tile_h = 132, 92
        gap_x, gap_y = 30, 46
        pad_x, pad_y = 235, 150
    if location_id == "grass_open":
        # Open grass has lots of free field; spread the array so it does not look packed.
        tile_w, tile_h = 144, 100
        gap_x, gap_y = 42, 58
        pad_x, pad_y = 120, 96
    if max_visible_panels >= 6 or max_visible_rows >= 4:
        # Dense mode for the 4 x 6 cap. Keep the full array visible in the
        # fixed scene canvas instead of letting it spill out of the background.
        if location_id == "rooftop_warehouse":
            tile_w, tile_h = 88, 50
            gap_x, gap_y = 20, 24
            pad_x, pad_y = 310, 86
        elif location_id == "desert_track":
            tile_w, tile_h = 78, 54
            gap_x, gap_y = 15, 30
            pad_x, pad_y = 365, 180
        elif location_id == "floating_water":
            tile_w, tile_h = 92, 64
            gap_x, gap_y = 22, 26
            pad_x, pad_y = 300, 165
        elif location_id == "agri_field_new":
            tile_w, tile_h = 86, 60
            gap_x, gap_y = 22, 30
            pad_x, pad_y = 215, 180
        elif location_id == "grass_open":
            tile_w, tile_h = 104, 72
            gap_x, gap_y = 34, 40
            pad_x, pad_y = 155, 118
    dynamic_field_w = pad_x * 2 + tile_w * max_visible_panels + gap_x * max(0, max_visible_panels - 1)
    dynamic_field_h = pad_y * 2 + tile_h * max_visible_rows + gap_y * max(0, max_visible_rows - 1)
    if location_id == "agri_field_new":
        dynamic_field_w += 132
        dynamic_field_h += 24
    static_field_w, static_field_h = module_field_static_canvas_size(location_id, view)
    field_w, field_h = static_field_w, static_field_h
    # Missing/pre-render-free layouts still build a single scalar scene. They
    # must not assemble already palette-mapped thermal preview tiles.
    background_view = "rgb" if view == "thermal" else view
    background_target = None if view == "thermal" else selected_row_id
    field = module_field_background(location_id, background_view, (field_w, field_h), background_target).convert("RGBA")
    thermal_scalar = None
    thermal_geometries = {}
    healthy_module_support = np.zeros((field_h, field_w), np.uint8)
    healthy_module_geometry = []
    if view == "thermal":
        thermal_scalar = thermal_background_realism(np.empty((field_h, field_w), np.float32), field, location_id)
        thermal_drift = thermal_scene_drift(thermal_scalar.shape, seed_for(location_id, "healthy-thermal-scene"))
    draw = ImageDraw.Draw(field, "RGBA")
    label_font = grid_font(12 if max_visible_panels < 6 else 10)

    selected_rect = None
    for row_idx, row in enumerate(visible_rows):
        for col_idx, panel_number in enumerate(visible_panels):
            cells = cells_for_panel(location_id, row["id"], panel_number)
            if not cells:
                continue
            is_selected = row["id"] == selected_row_id and panel_number == selected_panel
            panel_modules = set(module_numbers_for_panel(panel_number))
            records = [
                record for record in (fault_records or [])
                if record.get("row_id") == row["id"]
                and cell_by_id(location_id, row["id"], record.get("target_cell_id")).get("module_number") in panel_modules
            ]
            tile_selected_cell_id = selected_cell_id if is_selected else cells[0]["id"]
            if view != "thermal":
                tile = solar_panel_tile_image(location_id, view, row["id"], panel_number, tile_selected_cell_id, records, scene_distance)
                tile = tile.resize((tile_w, tile_h), Image.Resampling.BICUBIC)
            row_offset_x = 0
            if location_id == "agri_field_new":
                row_offset_x = [70, 70, 70, 70, 70][row_idx % 5]
            x = pad_x + row_offset_x + col_idx * (tile_w + gap_x)
            y = pad_y + row_idx * (tile_h + gap_y)
            if view == "thermal":
                key = (row_number_from_id(location_id, row["id"]), panel_number)
                geometry = {"box": (x, y, x + tile_w, y + tile_h),
                            "polygon": box_polygon((x, y, x + tile_w, y + tile_h))}
                thermal_geometries[key] = geometry
                for number, module in array_module_polygons(geometry).items():
                    cv2.fillConvexPoly(healthy_module_support, np.rint(module_outer_polygon(module)).astype(np.int32), 255)
                    healthy_module_geometry.append(module_outer_polygon(module))
                    composite_healthy_thermal_module(
                        thermal_scalar, module_outer_polygon(module), thermal_drift,
                        seed_for(location_id, key, number, "healthy-module"), *thermal_pv_operating_range(location_id),
                    )
                continue
            if view == "rgb":
                if location_id in {"agri_field_new", "grass_open"}:
                    rack_shadow = Image.new("RGBA", (tile_w + 30, tile_h + 34), (0, 0, 0, 0))
                    rack_draw = ImageDraw.Draw(rack_shadow, "RGBA")
                    rack_draw.polygon(
                        [
                            (14, 18),
                            (tile_w + 10, 18),
                            (tile_w + 24, tile_h + 18),
                            (22, tile_h + 25),
                        ],
                        fill=(0, 0, 0, 58),
                    )
                    rack_shadow = rack_shadow.filter(ImageFilter.GaussianBlur(6.0))
                    field.alpha_composite(rack_shadow, (x - 10, y - 8))
                    support = Image.new("RGBA", (tile_w + 10, 12), (0, 0, 0, 0))
                    support_draw = ImageDraw.Draw(support, "RGBA")
                    support_draw.line((5, 3, tile_w + 5, 3), fill=(16, 18, 18, 92), width=2)
                    support = support.filter(ImageFilter.GaussianBlur(1.2))
                    field.alpha_composite(support, (x - 5, y + tile_h - 2))
                shadow = Image.new("RGBA", (tile_w + 16, tile_h + 16), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow, "RGBA")
                shadow_draw.rounded_rectangle((8, 8, tile_w + 8, tile_h + 8), radius=2, fill=(0, 0, 0, 54))
                shadow = shadow.filter(ImageFilter.GaussianBlur(2.6))
                field.alpha_composite(shadow, (x - 8, y - 6))
            field.paste(tile, (x, y))
            if show_grid:
                outline = (56, 189, 248, 180) if records else (224, 242, 254, 80)
                draw.rectangle((x, y, x + tile_w, y + tile_h), outline=outline, width=2)
                if records:
                    names = []
                    for record in records:
                        name = short_fault_label(record.get("fault_type"))
                        if name not in names:
                            names.append(name)
                    label = ", ".join(names[:2])
                    if len(names) > 2:
                        label += f" +{len(names) - 2}"
                    draw_fault_label(draw, (x, y, x + tile_w, y + tile_h), label, label_font, field.size)
            if is_selected:
                selected_rect = (x, y, x + tile_w, y + tile_h)

    if view == "thermal":
        thermal_scalar = thermal_boundary_response(
            thermal_scalar, healthy_module_support, seed_for(location_id, "healthy-boundary-response"),
            location_id=location_id, source_image=field,
            geometries=[geometry["polygon"] for geometry in thermal_geometries.values()],
            module_geometries=healthy_module_geometry,
        )
        thermal_scalar = thermal_refine_background_materials(thermal_scalar, field, location_id, healthy_module_geometry)
        image = finalize_solar_render_thermal_image(
            location_id, thermal_scalar, thermal_geometries, selected_row_id,
            selected_cell_id, fault_records, scene_distance,
        )
        if show_grid:
            overlay = ImageDraw.Draw(image)
            for geometry in thermal_geometries.values():
                overlay.rectangle(geometry["box"], outline=(56, 189, 248), width=1)
        return image

    if selected_rect:
        draw_corner_marks(draw, selected_rect, (250, 204, 21, 255), width=3 if view == "rgb" else 5)

    return field


def render_module_field_map(location_id, view, selected_row_id, selected_cell_id, fault_records=None, show_grid=True, image=None):
    if image is None:
        image = module_field_image(location_id, view, selected_row_id, selected_cell_id, fault_records, show_grid)
    uri = data_uri(image)
    width, height = image.size
    components.html(
        f"""
        <div class="module-field"><img src="{uri}" width="{width}" height="{height}"/></div>
        <style>
          .module-field {{ width:100%; overflow:auto; border:1px solid #263244; border-radius:10px; background:#050816; }}
          .module-field img {{ width:{width}px; max-width:none; height:auto; display:block; margin:auto; }}
        </style>
        """,
        height=min(820, max(420, height + 44)),
        scrolling=False,
    )


def selected_location():
    return location_by_id(qp_get("location", LOCATIONS[0]["id"]))


def legacy_main():
    st.title("PV Array Fault Tool")
    st.caption("Select row, array, module, and cell. The main view shows solar arrays made from four 40-cell modules.")

    location = selected_location()
    view = qp_get("view", "thermal")
    if view not in ("rgb", "thermal"):
        view = "rgb"
    inspection_mode = "close"

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
                    inspect=inspection_mode,
                )
            )
            st.rerun()

        chosen_view = st.radio("Camera view", ["RGB", "Thermal"], index=0 if view == "rgb" else 1, horizontal=True)
        chosen_view = chosen_view.lower()
        if chosen_view != view:
            qp_set(**navigation_params(location["id"], chosen_view, selected_row_id, selected_cell_id, inspect=inspection_mode))
            st.rerun()

        st.caption("Panel array view with single-module diagnostics.")
        show_cell_grid = False

    main_tab, output_tab, cell_faults_tab, data_tab, big_screen_tab = st.tabs(
        ["Main", "Output", "Cell Faults", "Data", "Big Screen Test"]
    )

    fault_scale = DEFAULT_FAULT_SCALE
    fault_type = "CellCracking"
    manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
    lock_manual_target = True
    current_custom_layout = custom_layout_enabled()
    available_fault_types = selectable_fault_types_for_location(location["id"])
    default_manual_fault = "CellCracking" if "CellCracking" in available_fault_types else available_fault_types[0]
    layout_key = (
        f"{location['id']}_{view}_{int(current_custom_layout)}_"
        f"{custom_row_count(location['id'])}_{custom_panels_per_row(location['id'])}"
    )
    show_bounding_boxes = False

    with output_tab:
        target_col, preview_col = st.columns([1, 1])
        with target_col:
            st.subheader("Fault Target")
            row_options = [row["id"] for row in all_rows]
            row_choice = st.selectbox(
                "PV row",
                row_options,
                index=row_options.index(selected_row_id),
                format_func=lambda row_id: row_by_id(location["id"], row_id)["label"],
                key=f"fault_row_{layout_key}",
            )
            if row_choice != selected_row_id:
                qp_set(**navigation_params(location["id"], view, row_choice, cells_for_row(location["id"], row_choice)[0]["id"], inspect=inspection_mode))
                st.rerun()

            row_cells_for_choice = cells_for_row(location["id"], selected_row_id)
            panel_numbers = sorted(
                {
                    cell.get("panel_number")
                    for cell in row_cells_for_choice
                    if cell.get("panel_number") is not None
                }
            )
            selected_panel_number = selected_cell.get("panel_number") or panel_number_for_module(selected_cell.get("module_number") or 1)
            if selected_panel_number not in panel_numbers and panel_numbers:
                selected_panel_number = panel_numbers[0]
            if panel_numbers:
                panel_choice = st.selectbox(
                    "Array in selected row",
                    panel_numbers,
                    index=panel_numbers.index(selected_panel_number),
                    format_func=lambda panel_number: f"Array {panel_number}",
                    key=f"fault_panel_{layout_key}_{selected_row_id}",
                )
                panel_cells_for_choice = [
                    cell for cell in row_cells_for_choice
                    if cell.get("panel_number") == panel_choice
                ]
                if selected_cell_id not in {cell["id"] for cell in panel_cells_for_choice}:
                    qp_set(**navigation_params(location["id"], view, selected_row_id, panel_cells_for_choice[0]["id"], inspect=inspection_mode))
                    st.rerun()
            else:
                panel_choice = selected_panel_number
                panel_cells_for_choice = row_cells_for_choice

            module_numbers = sorted(
                {
                    cell.get("module_number")
                    for cell in panel_cells_for_choice
                    if cell.get("module_number") is not None
                }
            )
            selected_module_number = selected_cell.get("module_number")
            if selected_module_number not in module_numbers and module_numbers:
                selected_module_number = module_numbers[0]
            if module_numbers:
                module_choice = st.selectbox(
                    "Module inside selected array (1 of 4)",
                    module_numbers,
                    index=module_numbers.index(selected_module_number),
                    format_func=lambda module_number: f"Module {(module_number - 1) % SOLAR_PANEL_MODULES + 1} of {SOLAR_PANEL_MODULES}",
                    key=f"fault_module_{layout_key}_{selected_row_id}_{panel_choice}",
                )
                module_cells_for_choice = [
                    cell for cell in panel_cells_for_choice
                    if cell.get("module_number") == module_choice
                ]
                if selected_cell_id not in {cell["id"] for cell in module_cells_for_choice}:
                    qp_set(**navigation_params(location["id"], view, selected_row_id, module_cells_for_choice[0]["id"], inspect=inspection_mode))
                    st.rerun()
            else:
                module_choice = selected_module_number
                module_cells_for_choice = row_cells_for_choice

            cell_options = [cell["id"] for cell in module_cells_for_choice]
            cell_choice = st.selectbox(
                "PV cell in selected module",
                cell_options,
                index=cell_options.index(selected_cell_id),
                format_func=lambda cell_id: cell_by_id(location["id"], selected_row_id, cell_id)["label"],
                key=f"fault_cell_{layout_key}_{selected_row_id}_{module_choice}",
            )
            if cell_choice != selected_cell_id:
                qp_set(**navigation_params(location["id"], view, selected_row_id, cell_choice, inspect=inspection_mode))
                st.rerun()

            fault_type = st.selectbox(
                "Diagnostic fault type",
                available_fault_types,
                index=available_fault_types.index(default_manual_fault),
                format_func=fault_display_name,
                key=f"manual_fault_type_{layout_key}",
            )
            manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
            show_bounding_boxes = st.checkbox("Show bounding boxes", value=False, key=f"show_boxes_{layout_key}")
            st.write(FAULT_TYPES[fault_type])
            st.caption("Fault overlays are bounded to the chosen PV cell inside the selected module.")
            if selected_cell.get("module_number"):
                st.caption(
                    f"{PV_PANEL_MODEL_NAME} {selected_cell.get('panel_number')} | "
                    f"{PV_MODULE_MODEL_NAME} {selected_cell.get('module_number_in_panel', selected_cell['module_number'])} "
                    f"of {SOLAR_PANEL_MODULES} | cell {selected_cell['module_cell_number']} "
                    f"({CUSTOM_MODULE_CELL_COLS}x{CUSTOM_MODULE_CELL_ROWS} module-cell grid)"
                )

        with preview_col:
            st.subheader("Close Panel Workflow")
            st.caption(
                "Use this tab to choose the row, panel, module, and target cell. "
                "Use Cell Faults to assign faults across the selected panel."
            )
            st.metric("Selected row", selected_row["label"])
            st.metric("Selected panel", selected_cell.get("panel_number"))
            st.metric("Selected cell", selected_cell["id"])

    with cell_faults_tab:
        left_spacer, panel_fault_col = st.columns([0.15, 0.85])
        with panel_fault_col:
            st.subheader("Selected Panel Fault Map")
            panel_fault_records = st.session_state.get("panel_cell_fault_records", [])
            panel_cell_ids = {cell["id"] for cell in panel_cells_for_choice}
            current_panel_faults = {
                record.get("target_cell_id"): canonical_fault_type(record.get("fault_type"))
                for record in panel_fault_records
                if record.get("location_id") == location["id"]
                and record.get("row_id") == selected_row_id
                and record.get("target_cell_id") in panel_cell_ids
            }
            fault_name_to_type = {"None": ""}
            for item in available_fault_types:
                fault_name_to_type[fault_display_name(item)] = item
            fault_options = list(fault_name_to_type.keys())
            editor_rows = [
                {
                    "Cell": cell["id"],
                    "Module": f"Module {cell.get('module_number_in_panel', cell.get('module_number'))}",
                    "Cell in module": cell.get("module_cell_number"),
                    "Fault": fault_display_name(current_panel_faults[cell["id"]]) if cell["id"] in current_panel_faults else "None",
                }
                for cell in panel_cells_for_choice
            ]
            edited_fault_rows = st.data_editor(
                editor_rows,
                hide_index=True,
                use_container_width=True,
                disabled=["Cell", "Module", "Cell in module"],
                column_config={
                    "Fault": st.column_config.SelectboxColumn(
                        "Fault",
                        options=fault_options,
                        required=True,
                    )
                },
                key=f"panel_fault_editor_{layout_key}_{selected_row_id}_{panel_choice}",
                height=360,
            )
            if hasattr(edited_fault_rows, "to_dict"):
                edited_fault_rows = edited_fault_rows.to_dict("records")
            if st.button("Apply selected panel faults", use_container_width=True):
                remaining_records = [
                    record for record in panel_fault_records
                    if not (
                        record.get("location_id") == location["id"]
                        and record.get("row_id") == selected_row_id
                        and record.get("target_cell_id") in panel_cell_ids
                    )
                ]
                new_panel_records = []
                cells_by_id_for_panel = {cell["id"]: cell for cell in panel_cells_for_choice}
                for row in edited_fault_rows:
                    mapped_fault = fault_name_to_type.get(row.get("Fault"), "")
                    if not mapped_fault:
                        continue
                    cell = cells_by_id_for_panel.get(row.get("Cell"))
                    if cell is None:
                        continue
                    new_panel_records.append(fault_record(location, view, selected_row, cell, mapped_fault, fault_scale))
                st.session_state["panel_cell_fault_records"] = remaining_records + new_panel_records
                st.rerun()
            if st.button("Clear selected panel faults", use_container_width=True):
                st.session_state["panel_cell_fault_records"] = [
                    record for record in panel_fault_records
                    if not (
                        record.get("location_id") == location["id"]
                        and record.get("row_id") == selected_row_id
                        and record.get("target_cell_id") in panel_cell_ids
                    )
                ]
                st.rerun()

    with main_tab:
        layout_col, hotspot_col = st.columns([1, 1])
        with layout_col:
            st.subheader("Rows and Arrays")
            custom_choice = st.checkbox("Realistic generated PV layout", value=current_custom_layout)
            if custom_choice != current_custom_layout:
                first_row = f"{row_prefix(location['id'])}1"
                qp_set(
                    location=location["id"],
                    view=view,
                    inspect=inspection_mode,
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
                "Arrays per Row",
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
                    inspect=inspection_mode,
                    custom_layout=1,
                    row_count=int(desired_rows),
                    panels_per_row=int(desired_panels),
                    selected_pv=first_row,
                    selected_cell=f"{first_row}-C001",
                )
                st.rerun()
            cells_per_module = CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS
            cells_per_panel = cells_per_module * SOLAR_PANEL_MODULES
            total_arrays = int(desired_rows) * int(desired_panels)
            total_modules = total_arrays * SOLAR_PANEL_MODULES
            total_cells = total_modules * cells_per_module
            st.caption(
                f"Total Arrays: {total_arrays:,} | Total Modules: {total_modules:,} | "
                f"Total PV Cells: {total_cells:,}"
            )
            st.caption(
                f"One array = {SOLAR_PANEL_MODULES} modules; one module = "
                f"{CUSTOM_MODULE_CELL_COLS}x{CUSTOM_MODULE_CELL_ROWS} = {cells_per_module} cells."
            )
            cap = LAYOUT_CAPS.get(location["id"])
            if cap:
                st.info(CAP_JUSTIFICATIONS[location["id"]])
                if int(desired_rows) >= cap["rows"] or int(desired_panels) >= cap["modules"]:
                    st.warning(
                        "At cap: row/panel limits preserve solar-panel aspect ratio, readable cell labels, "
                        "and prevent layouts from spilling into unusable image areas."
                    )
        with hotspot_col:
            st.subheader("Random Hotspot Generator")
            hotspot_fault_types = ["SingleHotSpot", "MultiHotSpot"]
            if "random_fault_types" in st.session_state:
                st.session_state["random_fault_types"] = [
                    canonical_fault_type(item)
                    for item in st.session_state["random_fault_types"]
                    if canonical_fault_type(item) in hotspot_fault_types
                ] or ["SingleHotSpot"]
            random_types = st.multiselect(
                "Hotspot type",
                hotspot_fault_types,
                default=["SingleHotSpot"],
                format_func=fault_display_name,
                key="random_fault_types",
            )
            max_faults = max(1, min(50, target_counts(location["id"])["pv_cells"]))
            random_count = st.number_input("Number of random hotspots", min_value=1, max_value=max_faults, value=min(3, max_faults), step=1)
            random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
            if st.button("Generate random hotspots", use_container_width=True):
                if not random_types:
                    st.warning("Choose at least one hotspot type.")
                else:
                    rng = random.Random(seed_for(location["id"], view, "|".join(random_types), int(random_count), int(random_seed), "main-hotspots"))
                    choices = [
                        (row, cell)
                        for row in panels(location["id"])
                        for cell in cells_for_row(location["id"], row["id"])
                    ]
                    st.session_state["random_fault_records"] = [
                        fault_record(location, view, row, cell, rng.choice(random_types), fault_scale)
                        for row, cell in (rng.choice(choices) for _ in range(int(random_count)))
                    ]
                    st.rerun()
            if st.button("Clear generated hotspots", use_container_width=True):
                st.session_state["random_fault_records"] = []
                st.rerun()

    generated_fault_records = [
        record for record in stored_fault_records()
        if record["location_id"] == location["id"]
    ]
    panel_fault_records_for_location = [
        record for record in st.session_state.get("panel_cell_fault_records", [])
        if record.get("location_id") == location["id"]
    ]
    output_fault_records = generated_fault_records + panel_fault_records_for_location
    main_display_records = records_for_view(generated_fault_records, view)
    display_records = (
        display_fault_records(manual_record, output_fault_records)
        if lock_manual_target
        else output_fault_records or [manual_record]
    )
    display_records = records_for_view(display_records, view)

    terrain = module_field_background(
        location["id"],
        view,
        module_field_static_canvas_size(location["id"], view),
        selected_row_id,
    )
    main_pv_scene = module_field_image(
        location["id"],
        view,
        selected_row_id,
        selected_cell_id,
        main_display_records,
        show_grid=False,
    ).convert("RGB")
    pv_scene = module_field_image(
        location["id"],
        view,
        selected_row_id,
        selected_cell_id,
        display_records,
        show_grid=show_bounding_boxes,
    ).convert("RGB")
    thermal_records = records_for_view(display_records, "thermal")
    if view == "thermal":
        big_screen_thermal = pv_scene
    else:
        big_screen_thermal = None
    main_metadata = scene_metadata(location, view, selected_row, selected_cell, main_display_records)
    metadata = scene_metadata(location, view, selected_row, selected_cell, display_records)

    with main_tab:
        st.caption(
            f"{main_metadata['location_name']} | {main_metadata['view']} | rows: {main_metadata['row_count']} | "
            f"arrays/row: {main_metadata['panels_per_row']} | modules/row: {main_metadata['modules_per_row']} | "
            f"hotspots: {main_metadata['fault_count']}"
        )
        st.subheader("Solar array")
        render_module_field_map(
            location["id"],
            view,
            selected_row_id,
            selected_cell_id,
            main_display_records,
            show_grid=False,
            image=main_pv_scene,
        )
        selected_module_number = selected_cell.get("module_number")
        module_cell_count = len(module_cells_for_target(location["id"], selected_row_id, selected_cell_id))
        st.caption(
            f"Row {selected_row['label']} | selected array {selected_cell.get('panel_number')} | "
            f"selected module {selected_cell.get('module_number_in_panel', selected_module_number)}. "
            f"Each visible tile is one solar array: {SOLAR_PANEL_MODULES} modules x "
            f"{CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS} cells/module."
        )
    with output_tab:
        st.subheader("Close diagnostic module")
        selected_module_number, module_cell_count = render_module_zoom_map(
            pv_scene,
            location["id"],
            view,
            selected_row_id,
            selected_cell_id,
            display_records,
            show_grid=show_bounding_boxes,
        )
        st.caption(
            f"Module {selected_module_number or selected_cell.get('panel_number')} fills the diagnostic view. "
            f"{module_cell_count} selectable PV cells are shown; click a cell to move the target."
        )

        st.subheader("Stable field context")
        st.image(pv_scene, use_container_width=True)
        st.caption("Same static scene framing as the main view; use the close diagnostic module for precise cell targeting.")

    with data_tab:
        st.subheader("Fault Summary")
        summary_rows = fault_summary_rows(display_records)
        if summary_rows:
            st.table(summary_rows)
        else:
            st.caption("No faults selected.")

        st.subheader("AI Classifier Check")
        if big_screen_thermal is not None:
            render_classifier_check(big_screen_thermal)
        else:
            st.caption("Switch Camera view to Thermal to run the classifier check without generating an extra thermal scene in RGB mode.")

        st.subheader("Fault Table")
        st.table(fault_table_rows(display_records))

        with st.expander("PV cell IDs and coordinates for selected row", expanded=True):
            st.table(
                [
                    {
                        "pv_cell_number": cell.get("pv_cell_number_in_row", cell.get("panel_number")),
                        "pv_cell_id": cell["id"],
                        "panel": cell.get("panel_number"),
                        "module_number": cell.get("module_number"),
                        "module_in_panel": cell.get("module_number_in_panel"),
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
            f"arrays/row: {metadata['panels_per_row']} | modules/row: {metadata['modules_per_row']} | "
            f"faults: {metadata['fault_count']}"
        )
        if big_screen_thermal is not None:
            st.image(big_screen_thermal, use_container_width=True)
        else:
            st.caption("Thermal preview is generated when Camera view is set to Thermal.")


WORKFLOW_SECTIONS = ["Scene Setup", "Layout", "Faults", "Review", "Export", "DATA COLLECTION", "Settings", "Help"]


def ui_css():
    st.markdown(
        """
        <style>
        :root {
          --pv-bg: #07111f;
          --pv-panel: #0d1828;
          --pv-panel-2: #111f32;
          --pv-border: #20324a;
          --pv-text: #e5edf7;
          --pv-muted: #8ba3bd;
          --pv-blue: #38bdf8;
          --pv-blue-2: #2563eb;
          --pv-warn: #f59e0b;
          --pv-danger: #ef4444;
          --pv-good: #22c55e;
          --pv-streamlit-header: 3.75rem;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
          min-height: 100%;
          overflow-y: auto !important;
          overflow-x: auto !important;
        }
        .stApp {
          background: radial-gradient(circle at top left, #10233b 0, #07111f 34%, #050a12 100%);
          color: var(--pv-text);
        }
        [data-testid="stHeader"] {
          background: rgba(5, 10, 18, .96) !important;
          z-index: 100 !important;
        }
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        main {
          width: 100%;
          min-height: 100vh;
          overflow: visible !important;
        }
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"] {
          overflow: visible !important;
        }
        [data-testid="stSidebar"] {
          background: #060d18;
          border-right: 1px solid #16253a;
          min-width: 210px !important;
          max-width: 260px !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e5edf7; }
        .block-container {
          padding: calc(var(--pv-streamlit-header) + .8rem) 1.1rem 2rem !important;
          max-width: none !important;
          width: 100% !important;
          min-width: 0 !important;
          overflow: visible !important;
        }
        [data-testid="stHorizontalBlock"] { gap: .8rem; min-width:0 !important; }
        [data-testid="column"] { min-width:0 !important; }
        .pv-shell-title { font-size: 1.35rem; font-weight: 800; letter-spacing: .02em; color: #f8fbff; margin: 0 0 .25rem; }
        .pv-kicker { color: var(--pv-muted); font-size: .88rem; margin-bottom: .8rem; }
        .pv-topbar, .pv-card, .pv-canvas-card {
          background: linear-gradient(180deg, rgba(17,31,50,.94), rgba(9,18,31,.94));
          border: 1px solid var(--pv-border);
          border-radius: 10px;
          box-shadow: 0 18px 50px rgba(0,0,0,.26);
          position: relative;
          z-index: 1;
        }
        .pv-topbar { padding: .65rem .8rem; margin-bottom: .65rem; }
        .pv-card { padding: .68rem; margin-bottom: .62rem; }
        .pv-canvas-card { padding: .45rem; }
        .pv-metric-grid { display:grid; grid-template-columns: repeat(6, minmax(92px,1fr)); gap:.55rem; }
        .pv-metric { background:#07111f; border:1px solid #1b2b43; border-radius:12px; padding:.55rem .65rem; }
        .pv-metric span { display:block; color:#8ba3bd; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; }
        .pv-metric strong { display:block; color:#f8fbff; font-size:1.02rem; margin-top:.15rem; }
        .pv-section-title { color:#f8fbff; font-weight:800; font-size:.88rem; margin:0 0 .38rem; }
        .pv-muted { color: var(--pv-muted); }
        .pv-chip-row { display:flex; flex-wrap:wrap; gap:.4rem; }
        .pv-chip { border:1px solid #263a58; border-radius:999px; padding:.22rem .55rem; color:#cfe8ff; background:#081425; font-size:.78rem; }
        .pv-chip-hot { border-color:#7f1d1d; background:#2b1118; color:#fecaca; }
        .pv-status-ok { color:#86efac; }
        .pv-status-warn { color:#fcd34d; }
        .pv-table-wrap { max-height: 440px; overflow:auto; border:1px solid #1b2b43; border-radius:12px; }
        .pv-focus-title { color:#8ba3bd; font-size:.82rem; margin:.05rem 0 .45rem; }
        .pv-context-thumb img { max-height:150px; object-fit:contain; }
        .pv-nav-compact button, .pv-card button, [data-testid="stButton"] button {
          white-space: normal !important;
          overflow-wrap: anywhere;
          line-height: 1.15;
          height: auto;
          min-height: 2.05rem;
        }
        @media (max-width: 1400px) {
          [data-testid="stSidebar"] { min-width:190px !important; max-width:220px !important; }
          .block-container {
            padding-left:.75rem !important;
            padding-right:.75rem !important;
            padding-top:calc(var(--pv-streamlit-header) + .65rem) !important;
          }
          .pv-card { padding:.58rem; }
        }
        button[kind="primary"] { background: linear-gradient(135deg, #2563eb, #38bdf8) !important; border:0 !important; }
        div[data-testid="stRadio"] label, div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label {
          color:#cbd5e1 !important; font-weight:700;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] p,
        [data-testid="stWidgetLabel"] p, [data-testid="stFileUploader"] label p {
          color:#dbeafe !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if qp_get("focus", "0") == "1":
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { display:none !important; }
            .block-container { padding:.45rem .65rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def app_state_defaults():
    st.session_state.setdefault("workflow_section", qp_get("section", "Faults"))
    st.session_state.setdefault("filter_mode", qp_get("filter", "Show All"))
    st.session_state.setdefault("selected_module_number", int(qp_get("selected_module", 1) or 1))
    st.session_state.setdefault("recent_arrays", [])
    st.session_state.setdefault("fault_undo_stack", [])
    st.session_state.setdefault("fault_redo_stack", [])


def fault_snapshot():
    return {
        "random_fault_records": [dict(item) for item in st.session_state.get("random_fault_records", [])],
        "panel_cell_fault_records": [dict(item) for item in st.session_state.get("panel_cell_fault_records", [])],
    }


def restore_fault_snapshot(snapshot):
    st.session_state["random_fault_records"] = [dict(item) for item in snapshot.get("random_fault_records", [])]
    st.session_state["panel_cell_fault_records"] = [dict(item) for item in snapshot.get("panel_cell_fault_records", [])]


def push_fault_history():
    st.session_state.setdefault("fault_undo_stack", []).append(fault_snapshot())
    st.session_state["fault_undo_stack"] = st.session_state["fault_undo_stack"][-30:]
    st.session_state["fault_redo_stack"] = []


def undo_fault_action():
    stack = st.session_state.setdefault("fault_undo_stack", [])
    if not stack:
        return False
    st.session_state.setdefault("fault_redo_stack", []).append(fault_snapshot())
    restore_fault_snapshot(stack.pop())
    return True


def redo_fault_action():
    stack = st.session_state.setdefault("fault_redo_stack", [])
    if not stack:
        return False
    st.session_state.setdefault("fault_undo_stack", []).append(fault_snapshot())
    restore_fault_snapshot(stack.pop())
    return True


def records_for_location(location_id, view):
    generated = [record for record in stored_fault_records() if record.get("location_id") == location_id]
    panel_records = [
        record for record in st.session_state.get("panel_cell_fault_records", [])
        if record.get("location_id") == location_id
    ]
    return records_for_view(generated + panel_records, view)


def selected_array_from_cell(location_id, selected_row_id, selected_cell_id):
    cell = cell_by_id(location_id, selected_row_id, selected_cell_id)
    return {
        "row": int(row_number_from_id(location_id, selected_row_id)),
        "col": int(cell.get("panel_number") or 1),
        "key": f"R{int(row_number_from_id(location_id, selected_row_id))}-A{int(cell.get('panel_number') or 1):02d}",
    }


def default_cell_for_array(location_id, row_number, array_col, module_number=1, module_cell_number=1):
    row_id = f"{row_prefix(location_id)}{row_number}"
    try:
        row_cells = cells_for_row(location_id, row_id)
    except Exception:
        row_id = panels(location_id)[0]["id"]
        row_cells = cells_for_row(location_id, row_id)
    candidates = [
        cell for cell in row_cells
        if int(cell.get("panel_number") or 1) == int(array_col)
        and int(cell.get("module_number_in_panel") or 1) == int(module_number)
        and int(cell.get("module_cell_number") or 1) == int(module_cell_number)
    ]
    if not candidates:
        candidates = [
            cell for cell in row_cells
            if int(cell.get("panel_number") or 1) == int(array_col)
            and int(cell.get("module_number_in_panel") or 1) == int(module_number)
        ]
    if not candidates:
        candidates = [cell for cell in row_cells if int(cell.get("panel_number") or 1) == int(array_col)]
    return row_id, (candidates[0] if candidates else row_cells[0])


def selected_array_key(location_id, selected_row_id, selected_cell_id):
    query_key = qp_get("selected_array", "")
    if query_key:
        return str(query_key)
    return selected_array_from_cell(location_id, selected_row_id, selected_cell_id)["key"]


def array_regions(location_id, rgb_scene):
    cached = cached_array_regions(
        location_id,
        tuple(rgb_scene.size),
        solar_render_cache_signature(location_id, rgb_scene.size),
    )
    if cached:
        return cached
    return array_regions_from_image(location_id, rgb_scene)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_array_regions(location_id, render_size, render_signature):
    rgb_scene = solar_render_image(location_id, render_size)
    if rgb_scene is None:
        return []
    return array_regions_from_image(location_id, rgb_scene)


def array_regions_from_image(location_id, rgb_scene):
    metadata = load_solar_render_metadata(location_id)
    mask = solar_render_panel_mask(rgb_scene, location_id)
    geometries = solar_render_metadata_array_geometries(mask, metadata)
    transforms = ((metadata or {}).get("topology") or {}).get("array_transforms") or []
    transforms = sorted(transforms, key=lambda item: (int(item.get("row", 0)), int(item.get("col", 0))))
    regions = []
    if transforms and geometries:
        for transform in transforms:
            row_num = int(transform.get("row", 0))
            col_num = int(transform.get("col", 0))
            geometry = geometries.get((row_num, col_num))
            if not geometry:
                continue
            modules = array_module_polygons(geometry)
            regions.append(
                {
                    "key": f"R{row_num}-A{col_num:02d}",
                    "row": row_num,
                    "col": col_num,
                    "array_id": int(transform.get("array_id", len(regions) + 1)),
                    "orientation": transform.get("orientation", "landscape"),
                    "string_id": transform.get("string_id"),
                    "box": tuple(int(v) for v in geometry["box"]),
                    "polygon": geometry["polygon"],
                    "modules": modules,
                }
            )
        return regions

    fallback_boxes = solar_render_array_boxes(mask, custom_row_count(location_id), custom_panels_per_row(location_id))
    for (row_num, col_num), box in fallback_boxes.items():
        modules = array_module_polygons({"box": tuple(int(v) for v in box), "polygon": box_polygon(box)})
        regions.append(
            {
                "key": f"R{row_num}-A{col_num:02d}",
                "row": row_num,
                "col": col_num,
                "array_id": len(regions) + 1,
                "orientation": "unknown",
                "string_id": None,
                "box": tuple(int(v) for v in box),
                "polygon": box_polygon(box),
                "modules": modules,
            }
        )
    return regions


def fault_count_for_region(records, region):
    count = 0
    for record in records:
        if row_number_from_id(record.get("location_id"), record.get("row_id")) != region["row"]:
            continue
        try:
            cell = cell_by_id(record.get("location_id"), record.get("row_id"), record.get("target_cell_id"))
        except Exception:
            continue
        if int(cell.get("panel_number") or 1) == int(region["col"]):
            count += 1
    return count


def crop_with_padding(image, box, pad_ratio=0.08, min_pad=6, max_pad_ratio=0.18):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    pad_ratio = min(float(max_pad_ratio), max(0, float(pad_ratio)))
    pad_x = max(int(min_pad), int(w * pad_ratio))
    pad_y = max(int(min_pad), int(h * pad_ratio))
    crop = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image.width, x2 + pad_x),
        min(image.height, y2 + pad_y),
    )
    return image.crop(crop), crop


def module_box_in_array(region_box, module_number):
    x1, y1, x2, y2 = [int(v) for v in region_box]
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    boxes = {
        1: (x1, y1, mid_x, mid_y),
        2: (mid_x, y1, x2, mid_y),
        3: (x1, mid_y, mid_x, y2),
        4: (mid_x, mid_y, x2, y2),
    }
    return boxes.get(int(module_number), boxes[1])


def module_cell_fault_numbers(location_id, region, module_number, fault_records):
    row_id = f"{row_prefix(location_id)}{region['row']}"
    faulted = set()
    for record in fault_records or []:
        if record.get("location_id") != location_id or record.get("row_id") != row_id:
            continue
        try:
            cell = cell_by_id(location_id, row_id, record.get("target_cell_id"))
        except Exception:
            continue
        if int(cell.get("panel_number") or 1) != int(region["col"]):
            continue
        if int(cell.get("module_number_in_panel") or 1) != int(module_number):
            continue
        faulted.add(int(cell.get("module_cell_number") or 1))
    return faulted


def scaled_crop(image, box, pad_ratio, target_width=680, max_scale=3):
    crop, origin = crop_with_padding(image, box, pad_ratio)
    scale = max(1, min(max_scale, math.ceil(target_width / max(1, crop.width))))
    if scale == 1:
        return crop, origin, scale
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.BICUBIC), origin, scale


def scaled_polygon_crop(image, polygon, pad_ratio, target_width=760, max_scale=3):
    crop, origin = crop_from_polygon(image, polygon, pad_ratio)
    scale = max(1, min(max_scale, math.ceil(target_width / max(1, crop.width))))
    if scale == 1:
        return crop, origin, scale
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.BICUBIC), origin, scale


def show_crop_pair(left_image, right_image, polygon, title_left, title_right, pad_ratio=0.08):
    left_crop, _ = crop_from_polygon(left_image, polygon, pad_ratio)
    right_crop, _ = crop_from_polygon(right_image, polygon, pad_ratio)
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(title_left)
        st.image(left_crop, use_container_width=True)
    with col_b:
        st.caption(title_right)
        st.image(right_crop, use_container_width=True)


def scene_context_image(image, selected_region):
    context = image.convert("RGBA")
    if selected_region:
        overlay = Image.new("RGBA", context.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        polygon = selected_region.get("polygon") or box_polygon(selected_region["box"])
        draw.line([tuple(point) for point in polygon] + [tuple(polygon[0])], fill=(56, 189, 248, 235), width=4)
        draw.polygon([tuple(point) for point in polygon], fill=(56, 189, 248, 35))
        context = Image.alpha_composite(context, overlay)
    return context.convert("RGB")


def render_interactive_scene_canvas(
    image,
    location,
    view,
    regions,
    selected_key,
    records,
    filter_mode,
    compact=False,
    show_fault_markers=True,
):
    uri = data_uri(image)
    width, height = image.size
    expected_total = ((load_solar_render_metadata(location["id"]) or {}).get("counts") or {}).get("arrays", len(regions))
    selected_region = next((item for item in regions if item["key"] == selected_key), None)
    debug_fit = qp_get("debug_viewport", "0") == "1"
    debug_geometry = qp_get("debug_geometry", "0") == "1"
    canvas_class = "pv-canvas pv-canvas-compact" if compact else "pv-canvas pv-canvas-fit"
    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        polygon = region.get("polygon") or box_polygon(region["box"])
        fault_count = fault_count_for_region(records, region)
        fault_marked = show_fault_markers and fault_count > 0
        dim = filter_mode == "Faulty Only" and fault_count == 0
        if dim:
            svg.append(f'<polygon points="{svg_points(polygon)}" fill="#020617" fill-opacity=".48"/>')
        selected = region["key"] == selected_key
        stroke = "#38bdf8" if selected else ("#ef4444" if fault_marked else "#7dd3fc")
        opacity = ".95" if selected else (".72" if fault_marked else (".16" if debug_geometry else "0"))
        stroke_width = "2.2" if selected else ("1.7" if fault_marked else ("1" if debug_geometry else "0"))
        row_id, cell = default_cell_for_array(location["id"], region["row"], region["col"])
        params = navigation_params(
            location["id"],
            view,
            row_id,
            cell["id"],
            section=qp_get("section", "Faults"),
            selected_array=region["key"],
            selected_module="",
            cell_active=0,
        )
        target = f"?{urlencode(params)}"
        title = f'{region["key"]} | {region.get("orientation", "unknown")} | {fault_count} fault(s)'
        svg.append(
            f'<a href="{esc(target)}" target="_parent"><title>{esc(title)}</title>'
            f'<polygon points="{svg_points(polygon)}" '
            f'fill="#38bdf8" fill-opacity=".001" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-opacity="{opacity}" pointer-events="all"/></a>'
        )
        if fault_marked:
            badge_x = max(0, x2 - 18)
            badge_y = max(0, y1 + 10)
            svg.append(
                f'<circle cx="{badge_x}" cy="{badge_y}" r="10" fill="#ef4444" fill-opacity=".88"/>'
                f'<text x="{badge_x}" y="{badge_y + 4}" text-anchor="middle" font-size="10" '
                f'font-family="Inter, Arial" fill="white" font-weight="800">{fault_count}</text>'
            )
    if selected_region:
        polygon = selected_region.get("polygon") or box_polygon(selected_region["box"])
        svg.append(
            f'<polygon points="{svg_points(polygon)}" '
            f'fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-opacity=".9"/>'
        )
    if debug_geometry:
        for region in regions:
            svg.append(f'<polygon points="{svg_points(region.get("polygon") or box_polygon(region["box"]))}" fill="none" stroke="#2563eb" stroke-width="2" stroke-opacity=".9"/>')
            for module in (region.get("modules") or {}).values():
                svg.append(f'<polygon points="{svg_points(module["polygon"])}" fill="none" stroke="#22c55e" stroke-width="1.3" stroke-opacity=".9"/>')
    svg.append("</svg>")
    debug_html = ""
    if debug_fit:
        debug_html = (
            f'<div class="pv-viewport-debug">Source: {width} x {height} | '
            f'Fit Scene: contain | Visible arrays: {len(regions)} / {expected_total}</div>'
        )
    components.html(
        f"""
        <div class="{canvas_class}">
          {''.join(svg)}
          {debug_html}
        </div>
        <style>
          html, body {{ margin:0; overflow:auto; background:transparent; }}
          .pv-canvas {{
            width:100%;
            min-width:0;
            border:1px solid #20324a;
            border-radius:14px;
            overflow:auto;
            background:#020617;
            box-sizing:border-box;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:flex-start;
          }}
          .pv-canvas svg {{
            display:block;
            width:auto;
            height:auto;
            max-width:100%;
            max-height:calc(100vh - 170px);
            flex:0 0 auto;
          }}
          .pv-canvas-compact svg {{ max-height:168px; }}
          .pv-viewport-debug {{
            align-self:stretch;
            color:#bae6fd;
            font:12px/1.4 Inter, Arial, sans-serif;
            background:rgba(2,6,23,.88);
            border-top:1px solid rgba(125,211,252,.28);
            padding:6px 10px;
          }}
          .pv-canvas a:hover polygon {{ stroke-opacity:0; fill-opacity:.001; filter:none; }}
        </style>
        """,
        height=(230 if compact else min(980, max(660, int(height * 0.52)))),
        scrolling=True,
    )


def render_module_crop_map(scene_image_obj, location, view, region, selected_module, selected_cell_id, fault_records=None):
    module_geometry = (region.get("modules") or array_module_polygons(region)).get(int(selected_module))
    if not module_geometry:
        module_geometry = array_module_polygons(region)[1]
    module_polygon = module_geometry["polygon"]
    display, crop_origin, scale = scaled_polygon_crop(scene_image_obj, module_polygon, 0.065, target_width=880, max_scale=3)
    uri = data_uri(display)
    width, height = display.size
    display_module_polygon = polygon_to_crop(module_polygon, crop_origin, scale)
    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>',
        f'<defs><clipPath id="module-clip"><polygon points="{svg_points(display_module_polygon)}"/></clipPath></defs>',
    ]
    selected_cell = None
    try:
        selected_cell = cell_by_id(location["id"], f"{row_prefix(location['id'])}{region['row']}", selected_cell_id)
    except Exception:
        selected_cell = None
    faulted_cells = module_cell_fault_numbers(location["id"], region, selected_module, fault_records)
    svg.append('<g clip-path="url(#module-clip)">')
    for cell_no, cell_geometry in module_cell_polygons(module_polygon).items():
        display_cell_polygon = polygon_to_crop(cell_geometry["polygon"], crop_origin, scale)
        row_id, target_cell = default_cell_for_array(location["id"], region["row"], region["col"], selected_module, cell_no)
        active = (
            selected_cell is not None
            and int(selected_cell.get("panel_number") or 1) == int(region["col"])
            and int(selected_cell.get("module_number_in_panel") or 1) == int(selected_module)
            and int(selected_cell.get("module_cell_number") or 1) == cell_no
        )
        faulted = cell_no in faulted_cells
        params = navigation_params(
            location["id"],
            view,
            row_id,
            target_cell["id"],
            section="Faults",
            selected_array=region["key"],
            selected_module=selected_module,
            cell_active=1,
        )
        target = f"?{urlencode(params)}"
        stroke = "#38bdf8" if active else ("#ef4444" if faulted else "#e0f2fe")
        opacity = ".92" if active else (".78" if faulted else ".18")
        fill = "#38bdf8" if active else ("#ef4444" if faulted else "#38bdf8")
        fill_opacity = ".11" if active else (".08" if faulted else ".001")
        stroke_width = "1.6" if active else ("1.2" if faulted else ".75")
        svg.append(
            f'<a href="{esc(target)}" target="_parent"><title>Cell {cell_no}</title>'
            f'<polygon points="{svg_points(display_cell_polygon)}" '
            f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="{opacity}" '
            f'pointer-events="all"/></a>'
        )
    svg.append("</g>")
    svg.append(f'<polygon points="{svg_points(display_module_polygon)}" fill="none" stroke="#bae6fd" stroke-width="1.2" stroke-opacity=".35"/>')
    svg.append("</svg>")
    components.html(
        f"""
        <div class="pv-module-crop">{''.join(svg)}</div>
        <style>
          .pv-module-crop {{ width:100%; overflow:auto; border:1px solid #20324a; border-radius:14px; background:#020617; }}
          .pv-module-crop svg {{ width:{width}px; max-width:100%; height:auto; display:block; margin:auto; }}
          .pv-module-crop a:hover polygon {{ stroke:#38bdf8; stroke-opacity:1; stroke-width:1.8; fill-opacity:.13; }}
        </style>
        """,
        height=min(820, max(560, height + 42)),
        scrolling=False,
    )
    return display


def render_array_module_selector(
    location,
    view,
    scene_image_obj,
    region,
    selected_module=0,
    selected_cell_id=None,
    fault_records=None,
    show_boxes=True,
    show_cell_map=False,
):
    display, origin, modules, scale = crop_array_from_module_union(
        scene_image_obj,
        region,
        pad_ratio=0.14,
        target_width=None,
        max_scale=1,
    )
    uri = data_uri(display)
    width, height = display.size
    row_id = f"{row_prefix(location['id'])}{region['row']}"
    selected_cell = None
    if selected_cell_id:
        try:
            selected_cell = cell_by_id(location["id"], row_id, selected_cell_id)
        except Exception:
            selected_cell = None
    faults_by_module_cell = {}
    for record in fault_records or []:
        if record.get("location_id") != location["id"] or record.get("row_id") != row_id:
            continue
        try:
            record_cell = cell_by_id(location["id"], row_id, record.get("target_cell_id"))
        except Exception:
            continue
        if int(record_cell.get("panel_number") or 1) != int(region["col"]):
            continue
        key = (
            int(record_cell.get("module_number_in_panel") or 1),
            int(record_cell.get("module_cell_number") or 1),
        )
        faults_by_module_cell.setdefault(key, []).append(record)
    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<image href="{uri}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>',
    ]
    debug_geometry = bool(st.session_state.get("debug_geometry_overlay", False))
    for module_number, module in modules.items():
        outer_polygon = module_outer_polygon(module)
        active_polygon = module_active_polygon(module)
        display_polygon = polygon_to_crop(outer_polygon, origin, scale)
        display_active_polygon = polygon_to_crop(active_polygon, origin, scale)
        bx1, by1, bx2, by2 = polygon_bounds(display_polygon)
        active = module_number == selected_module
        stroke = "#38bdf8" if active else "#bae6fd"
        stroke_opacity = ".20" if active else ".07"
        stroke_width = "1.05" if active else ".75"
        badge_w, badge_h = 27, 18
        badge_x = bx1 + 6
        badge_y = by1 + 6
        if show_boxes or debug_geometry:
            svg.append(
                f'<polygon points="{svg_points(display_polygon)}" '
                f'fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="{stroke_opacity}" pointer-events="none"/>'
                + (
                    f'<polygon points="{svg_points(display_polygon)}" fill="none" stroke="#ef4444" stroke-width="1.1" stroke-opacity=".84" pointer-events="none"/>'
                    f'<polygon points="{svg_points(display_active_polygon)}" fill="none" stroke="#22c55e" stroke-width=".9" stroke-opacity=".72" pointer-events="none"/>'
                    if debug_geometry else ""
                )
                + f'<g pointer-events="none" opacity="{".92" if active else ".66"}">'
                f'<rect x="{badge_x:.1f}" y="{badge_y:.1f}" width="{badge_w}" height="{badge_h}" rx="5" ry="5" '
                f'fill="#020617" fill-opacity=".64" stroke="#bae6fd" stroke-opacity=".35"/>'
                f'<text x="{badge_x + badge_w / 2:.1f}" y="{badge_y + 13:.1f}" text-anchor="middle" font-size="10.5" '
                f'font-family="Inter, Arial" fill="#e0f2fe" font-weight="700">M{module_number}</text></g>'
            )
        for cell_no, cell_geometry in module_cell_polygons(module).items():
            _, target_cell = default_cell_for_array(location["id"], region["row"], region["col"], module_number, cell_no)
            target = f"?{urlencode(navigation_params(
                location["id"],
                view,
                row_id,
                target_cell["id"],
                section="Faults",
                selected_array=region["key"],
                selected_module=module_number,
                cell_active=1,
            ))}"
            display_cell_polygon = polygon_to_crop(cell_geometry["polygon"], origin, scale)
            selected = bool(
                selected_cell
                and int(selected_cell.get("panel_number") or 1) == int(region["col"])
                and int(selected_cell.get("module_number_in_panel") or 1) == int(module_number)
                and int(selected_cell.get("module_cell_number") or 1) == int(cell_no)
            )
            cell_faults = faults_by_module_cell.get((int(module_number), int(cell_no)), [])
            if show_cell_map or debug_geometry:
                fill = "#38bdf8"
                fill_opacity = ".006"
                cell_stroke = "#67e8f9" if debug_geometry else "#bae6fd"
                cell_stroke_opacity = ".52" if debug_geometry else ".12"
                cell_stroke_width = ".65"
            elif selected:
                fill = "#38bdf8"
                fill_opacity = ".001"
                cell_stroke = "#38bdf8"
                cell_stroke_opacity = "0"
                cell_stroke_width = ".65"
            else:
                fill = "#38bdf8"
                fill_opacity = ".001"
                cell_stroke = "#38bdf8"
                cell_stroke_opacity = "0"
                cell_stroke_width = ".65"
            title = f"Module {module_number}, Cell {cell_no}"
            svg.append(
                f'<a class="pv-cell-hit" href="{esc(target)}" target="_parent"><title>{esc(title)}</title>'
                f'<polygon points="{svg_points(display_cell_polygon)}" fill="{fill}" fill-opacity="{fill_opacity}" '
                f'stroke="{cell_stroke}" stroke-opacity="{cell_stroke_opacity}" stroke-width="{cell_stroke_width}" '
                f'pointer-events="all"/></a>'
            )
            if show_boxes and cell_faults:
                fault_stroke = "#fb923c"
                if any(record.get("fault_type") in HOTSPOT_FAULTS for record in cell_faults):
                    fault_stroke = "#facc15"
                svg.append(
                    f'<polygon points="{svg_points(display_cell_polygon)}" fill="none" '
                    f'stroke="{fault_stroke}" stroke-width="1.4" stroke-opacity=".86" pointer-events="none"/>'
                )
    svg.append("</svg>")
    components.html(
        f"""
        <div class="pv-array-crop">{''.join(svg)}</div>
        <style>
          .pv-array-crop {{ width:100%; overflow:hidden; border:1px solid #20324a; border-radius:14px; background:#020617; }}
          .pv-array-crop svg {{ width:{width}px; max-width:100%; height:auto; display:block; margin:auto; }}
          .pv-array-crop .pv-cell-hit:hover polygon {{
            stroke-opacity:0;
            fill-opacity:.001;
            filter:none;
          }}
        </style>
        """,
        height=max(540, height + 42),
        scrolling=False,
    )
    return display


def export_current_view_bytes(image):
    return image_png_bytes(image.convert("RGB"))


def generated_export_geometry(location_id, scene, regions, region_size, records):
    """Export the applied effects and physical polygons in exported-image pixels."""
    geometries = {}
    modules = []
    support = np.zeros((scene.height, scene.width), np.uint8)
    for source_region in regions:
        region = scale_region_to_image(source_region, region_size, scene.size)
        row, col = int(region["row"]), int(region["col"])
        geometries[(row,col)] = region
        for number, module in array_module_polygons(region).items():
            outer, active = module_outer_polygon(module), module_active_polygon(module)
            modules.append({"module_id": f"R{row}-A{col:02d}-M{number}", "row":row, "array":col,
                            "module_number_in_panel":number, "outer_polygon_px":outer,
                            "active_polygon_px":active, "coordinate_space":"exported_scene_pixels"})
            cv2.fillConvexPoly(support, np.rint(active).astype(np.int32), 255, lineType=cv2.LINE_8)
    projected = []
    for source in thermal_records_for_distance(records, "far"):
        record = dict(source)
        cell = rendered_fault_cell_from_record(location_id, record, record.get("row_id"), record.get("target_cell_id"), geometries)
        if cell is None:
            continue
        x1,y1,x2,y2 = polygon_bounds(cell["source_polygon"])
        record["target_cell_bbox_px"] = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
        record["bbox_px"] = dict(cell["active_module_bbox_px"])
        record["cell_polygon_px"] = cell["source_polygon"]
        record["module_polygon_px"] = cell["active_module_polygon"]
        record["module_id"] = f"R{row_number_from_id(location_id,record['row_id'])}-A{int(cell.get('panel_number') or 1):02d}-M{int(cell.get('module_number_in_panel') or 1)}"
        record["fault_bbox_px"] = record["visible_fault_bbox_px"] = cell["active_module_bbox_px"]
        record["annotation_geometry"] = "physical_module_support_bounds; not pixel fault segmentation"
        record["coordinate_space"] = "exported_scene_pixels"
        record["renderer_class"] = "PRODUCTION_REFERENCE" if canonical_fault_type(record.get("fault_type")) in {"SingleHotSpot","SingleDiode","SingleByPassed"} else "SYNTHETIC_DIAGNOSTIC"
        record["radiometric"] = False
        record["image_size"] = list(scene.size)
        fx,fy = polygon_centroid(cell["source_polygon"])
        record["fault_origin_px"] = {"x":fx,"y":fy}
        bounds = record["bbox_px"]
        cx,cy = (bounds["x1"]+bounds["x2"])*0.5,(bounds["y1"]+bounds["y2"])*0.5
        if canonical_fault_type(record["fault_type"]) in (set(HOTSPOT_FAULTS) | set(SURFACE_OBSTRUCTION_FAULTS) | set(DIODE_FAULTS)):
            cx, cy = fx, fy
        record["center_px"] = {"x":cx,"y":cy}
        record["center_normalized"] = {"x":cx/scene.width,"y":cy/scene.height}
        projected.append(record)
    return projected, modules, Image.fromarray(support,"L")


def generated_geometry_dataset(location, view, row, cell, records, scene, rgb, crop, bounds, modules, support, healthy=None):
    metadata = scene_metadata(location,view,row,cell,records)
    metadata.update({"image_size":list(scene.size), "coordinate_space":"exported_scene_pixels",
                     "crop_bounds_px":list(bounds), "crop_contents":"selected array context",
                     "renderer":st.session_state.get("thermal_renderer_mode","Current Thermal"),
                     "relative_simulation":True, "radiometric":False,
                     "fault_bounds":"physical module support, not pixel fault segmentation",
                     "cell_geometry":"synthetic electrical layout"})
    package = BytesIO()
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("pv_scene.png",image_png_bytes(scene))
        archive.writestr("source_rgb.png",image_png_bytes(rgb))
        archive.writestr("selected_crop.png",image_png_bytes(crop))
        archive.writestr("cropped_row.png",image_png_bytes(crop))
        archive.writestr("faults.json",json.dumps(json_safe_export_value(records),indent=2))
        archive.writestr("faults.csv",fault_records_csv(records))
        archive.writestr("metadata.json",json.dumps(json_safe_export_value(metadata),indent=2))
        archive.writestr("module_geometry.json",json.dumps(json_safe_export_value(modules),indent=2))
        archive.writestr("pv_union_mask.png",image_png_bytes(support))
        if healthy is not None:
            archive.writestr("healthy_thermal.png",image_png_bytes(healthy))
    return package.getvalue()


def parse_panel_search(text):
    import re

    raw = (text or "").strip().upper().replace(" ", "")
    match = re.match(r"R(\d+)-?A(\d+)", raw)
    if match:
        return ("array", int(match.group(1)), int(match.group(2)))
    match = re.match(r"R(\d+)-?C(\d+)", raw)
    if match:
        return ("cell", int(match.group(1)), int(match.group(2)))
    return None


def fault_target_cells(location_id):
    targets = []
    for row in panels(location_id):
        for cell in cells_for_row(location_id, row["id"]):
            targets.append((row, cell))
    return targets


def top_summary_bar(location, view, metadata, active_faults):
    st.markdown(
        f"""
        <div class="pv-topbar">
          <div class="pv-shell-title">{esc(location["name"])}</div>
          <div class="pv-metric-grid">
            <div class="pv-metric"><span>View</span><strong>{esc(view.upper())}</strong></div>
            <div class="pv-metric"><span>Rows</span><strong>{metadata["row_count"]}</strong></div>
            <div class="pv-metric"><span>Arrays / Row</span><strong>{metadata["arrays_per_row"]}</strong></div>
            <div class="pv-metric"><span>Total Arrays</span><strong>{metadata["total_arrays"]:,}</strong></div>
            <div class="pv-metric"><span>Total Cells</span><strong>{metadata.get("total_cells") or 0:,}</strong></div>
            <div class="pv-metric"><span>Faults</span><strong>{active_faults}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="PV Array Fault Tool", layout="wide")
    if qp_get("section") == "legacy" and not reference_demo_selected() and qp_get("thermal_renderer", "current") != "reference":
        legacy_main()
        return

    ui_css()
    app_state_defaults()

    base_source = qp_get("base_source", "generated")
    if base_source not in {"generated", "real_farm"}:
        base_source = "generated"
    with st.sidebar:
        st.markdown("### Base Source")
        source_choice = st.radio(
            "Source mode",
            ["Generated PV Farm", "Real PV Farm Image"],
            index=1 if base_source == "real_farm" else 0,
            label_visibility="collapsed",
        )
        chosen_source = "real_farm" if source_choice == "Real PV Farm Image" else "generated"
        if chosen_source != base_source:
            qp_set(base_source=chosen_source)
            st.rerun()
        if base_source != "real_farm":
            reference_demo_selector()

    if base_source == "real_farm":
        real_farm_main()
        return

    st.session_state.pop("real_farm_generated_cache_released", None)
    location = selected_location()
    view = qp_get("view", "thermal")
    if view not in ("rgb", "thermal"):
        view = "rgb"

    current_custom_layout = custom_layout_enabled()
    all_rows = panels(location["id"])
    selected_row_id = qp_get("selected_pv", all_rows[0]["id"])
    if selected_row_id not in {row["id"] for row in all_rows}:
        selected_row_id = all_rows[0]["id"]
    row_cells = cells_for_row(location["id"], selected_row_id)
    selected_cell_id = qp_get("selected_cell", row_cells[0]["id"])
    if selected_cell_id not in {cell["id"] for cell in row_cells}:
        selected_cell_id = row_cells[0]["id"]
    selected_row = row_by_id(location["id"], selected_row_id)
    selected_cell = cell_by_id(location["id"], selected_row_id, selected_cell_id)
    selected_module_raw = qp_get("selected_module", "")
    module_selected = str(selected_module_raw).isdigit() and int(selected_module_raw) > 0
    selected_module = int(selected_module_raw) if module_selected else 1
    selected_module = max(1, min(4, selected_module))
    cell_active = module_selected and qp_get("cell_active", "0") == "1"
    st.session_state["selected_module_number"] = selected_module
    section = qp_get("section", st.session_state.get("workflow_section", "Faults"))
    if section not in WORKFLOW_SECTIONS:
        section = "Faults"

    with st.sidebar:
        st.markdown("### PV Inspection")
        section_choice = st.radio("Workflow", WORKFLOW_SECTIONS, index=WORKFLOW_SECTIONS.index(section), label_visibility="collapsed")
        if section_choice != section:
            qp_set(**navigation_params(
                location["id"],
                view,
                selected_row_id,
                selected_cell_id,
                section=section_choice,
                selected_array=qp_get("selected_array", ""),
                selected_module=qp_get("selected_module", ""),
                cell_active=qp_get("cell_active", "0"),
            ))
            st.rerun()
        st.divider()
        st.caption("Settings")
        chosen_view = st.radio("RGB / Thermal", ["RGB", "Thermal"], index=0 if view == "rgb" else 1, horizontal=True)
        if chosen_view.lower() != view:
            qp_set(**navigation_params(
                location["id"],
                chosen_view.lower(),
                selected_row_id,
                selected_cell_id,
                section=section,
                selected_array=qp_get("selected_array", ""),
                selected_module=qp_get("selected_module", ""),
                cell_active=qp_get("cell_active", "0"),
            ))
            st.rerun()
        filter_choice = st.radio("Field Filter", ["Show All", "Faulty Only"], index=0 if qp_get("filter", "Show All") == "Show All" else 1)
        if filter_choice != qp_get("filter", "Show All"):
            qp_set(**navigation_params(location["id"], view, selected_row_id, selected_cell_id, section=section, filter=filter_choice))
            st.rerun()
        if st.button("Clear all faults", use_container_width=True):
            push_fault_history()
            st.session_state["random_fault_records"] = []
            st.session_state["panel_cell_fault_records"] = []
            st.toast("All faults cleared")
            st.rerun()
        undo_col, redo_col = st.columns(2)
        with undo_col:
            if st.button("Undo", use_container_width=True, disabled=not st.session_state.get("fault_undo_stack")):
                if undo_fault_action():
                    st.rerun()
        with redo_col:
            if st.button("Redo", use_container_width=True, disabled=not st.session_state.get("fault_redo_stack")):
                if redo_fault_action():
                    st.rerun()

    fault_scale = DEFAULT_FAULT_SCALE
    records = records_for_location(location["id"], view)
    experimental_thermal = reference_demo_selected()
    if experimental_thermal:
        reference_demo_notice()
        records = []
    metadata = scene_metadata(location, view, selected_row, selected_cell, thermal_records_for_distance(records, "far"))
    inspection_mode = st.session_state.get("inspection_crop_mode", "Current")
    inspection_geometry = st.session_state.get("inspection_geometry", "Natural")
    if inspection_geometry not in ("Natural", "Rectified"):
        inspection_geometry = "Natural"
    current_array_key = selected_array_key(location["id"], selected_row_id, selected_cell_id)
    array_selected = bool(qp_get("selected_array", ""))
    inspection_state = "scene" if not array_selected else ("module" if module_selected else "array")
    focus_mode = qp_get("focus", "0") == "1"

    rgb_scene = None
    thermal_scene = None
    scene = None
    regions = None
    selected_region = None

    def get_rgb_scene():
        nonlocal rgb_scene
        if rgb_scene is None:
            rgb_scene = module_field_image(location["id"], "rgb", selected_row_id, selected_cell_id, [], show_grid=False).convert("RGB")
            from pv_far_surface import generated_rgb_scene
            rgb_scene, _ = generated_rgb_scene(globals(), rgb_scene, location["id"], selected_row_id, selected_cell_id, records)
        return rgb_scene

    def get_thermal_scene():
        nonlocal thermal_scene
        if thermal_scene is None:
            thermal_distance = "close" if section == "Faults" and bool(qp_get("selected_array", "")) else "far"
            thermal_scene = module_field_image(
                location["id"],
                "thermal",
                selected_row_id,
                selected_cell_id,
                records,
                show_grid=False,
                scene_distance=thermal_distance,
            ).convert("RGB")
        return thermal_scene

    def get_scene():
        nonlocal scene
        if scene is None:
            scene = get_thermal_scene() if view == "thermal" else get_rgb_scene()
        return scene

    def get_regions():
        nonlocal regions
        if regions is None:
            clean_rgb = module_field_image(location["id"], "rgb", selected_row_id, selected_cell_id, [], show_grid=False).convert("RGB")
            regions = array_regions(location["id"], clean_rgb)
        return regions

    def get_selected_region():
        nonlocal selected_region, current_array_key
        if selected_region is None:
            available_regions = get_regions()
            selected_region = next((item for item in available_regions if item["key"] == current_array_key), None)
            if selected_region is None and available_regions:
                selected_region = available_regions[0]
                current_array_key = selected_region["key"]
            if selected_region:
                recent = [item for item in st.session_state.get("recent_arrays", []) if item != current_array_key]
                st.session_state["recent_arrays"] = [current_array_key] + recent[:7]
        return selected_region

    far_observation_types = set(globals().get("FAR_SCOUT_FAULT_TYPES", [])) | set(HOTSPOT_FAULTS)

    def jump_to_fault(record):
        row_id = record["row_id"]
        cell = cell_by_id(location["id"], row_id, record["target_cell_id"])
        array_col = int(cell.get("panel_number") or 1)
        qp_set(**navigation_params(
            location["id"],
            view,
            row_id,
            cell["id"],
            section="Faults",
            selected_array=f"R{row_number_from_id(location['id'], row_id)}-A{array_col:02d}",
            selected_module=cell.get("module_number_in_panel", 1),
            cell_active=1,
        ))

    def jump_to_region(region):
        row_id, cell = default_cell_for_array(location["id"], region["row"], region["col"], 1, 1)
        qp_set(**navigation_params(
            location["id"],
            view,
            row_id,
            cell["id"],
            section="Faults",
            selected_array=region["key"],
            selected_module="",
            cell_active=0,
        ))

    def enter_close_view():
        diagnostic_records = [
            record for record in records
            if record.get("location_id") == location["id"]
            and record.get("fault_type") not in far_observation_types
        ]
        observation_records = [
            record for record in records
            if record.get("location_id") == location["id"]
            and record.get("fault_type") in far_observation_types
        ]
        # Close View is the observation-to-diagnosis workflow. Start on the
        # first generated hotspot and keep existing diagnoses available after it.
        target_record = (observation_records or diagnostic_records or [None])[0]
        if target_record:
            row_id = target_record["row_id"]
            cell = cell_by_id(location["id"], row_id, target_record["target_cell_id"])
            array_col = int(cell.get("panel_number") or 1)
            array_key = f"R{row_number_from_id(location['id'], row_id)}-A{array_col:02d}"
        else:
            region = get_selected_region()
            row_id, cell = default_cell_for_array(location["id"], region["row"], region["col"], 1, 1)
            array_key = region["key"]

        qp_set(**navigation_params(
            location["id"],
            view,
            row_id,
            cell["id"],
            section="Faults",
            selected_array=array_key,
            selected_module=cell.get("module_number_in_panel", 1),
            cell_active=1,
        ))

    top_summary_bar(location, view, metadata, metadata["fault_count"])
    if section == "Faults":
        from pv_data_collection import active_case_banner
        active_case_banner("Generated")

    if section == "Scene Setup":
        scene = get_scene()
        regions = get_regions()
        selected_region = get_selected_region()
        left, right = st.columns([.44, .56])
        with left:
            st.markdown('<div class="pv-card"><div class="pv-section-title">Scene Setup</div>', unsafe_allow_html=True)
            names = [item["name"] for item in LOCATIONS]
            current_index = [item["id"] for item in LOCATIONS].index(location["id"])
            chosen = st.selectbox("Terrain / location", names, index=current_index)
            chosen_location = LOCATIONS[names.index(chosen)]
            if chosen_location["id"] != location["id"]:
                new_row = panels(chosen_location["id"])[0]["id"]
                qp_set(**navigation_params(chosen_location["id"], view, new_row, cells_for_row(chosen_location["id"], new_row)[0]["id"], section="Scene Setup"))
                st.rerun()
            row_value = custom_row_count(location["id"]) if current_custom_layout else min(len(base_row_defs_for_location(location["id"])), max_custom_rows(location["id"]))
            panel_value = custom_panels_per_row(location["id"]) if current_custom_layout else default_panels_per_row(location["id"])
            desired_rows = st.number_input("Rows", min_value=1, max_value=max_custom_rows(location["id"]), value=int(row_value), step=1)
            desired_arrays = st.number_input("Arrays per Row", min_value=1, max_value=max_custom_panels_per_row(location["id"]), value=int(panel_value), step=1)
            if int(desired_rows) != custom_row_count(location["id"]) or int(desired_arrays) != custom_panels_per_row(location["id"]) or not current_custom_layout:
                if st.button("Apply Layout", type="primary", use_container_width=True):
                    first_row = f"{row_prefix(location['id'])}1"
                    qp_set(
                        location=location["id"],
                        view=view,
                        section="Layout",
                        custom_layout=1,
                        row_count=int(desired_rows),
                        panels_per_row=int(desired_arrays),
                        selected_pv=first_row,
                        selected_cell=f"{first_row}-C001",
                    )
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="pv-canvas-card">', unsafe_allow_html=True)
            render_interactive_scene_canvas(scene, location, view, regions, current_array_key, records, qp_get("filter", "Show All"))
            st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Layout":
        scene = get_scene()
        regions = get_regions()
        selected_region = get_selected_region()
        main_col, info_col = st.columns([.72, .28])
        with main_col:
            st.markdown('<div class="pv-canvas-card">', unsafe_allow_html=True)
            render_interactive_scene_canvas(scene, location, view, regions, current_array_key, records, qp_get("filter", "Show All"))
            st.markdown("</div>", unsafe_allow_html=True)
        with info_col:
            st.markdown('<div class="pv-card"><div class="pv-section-title">Topology</div>', unsafe_allow_html=True)
            meta = load_solar_render_metadata(location["id"]) or {}
            topology = meta.get("topology") or {}
            st.write(f"Topology: `{topology.get('topology_type', 'generated')}`")
            st.write(f"Rows: `{topology.get('requested_rows', metadata['row_count'])}`")
            st.write(f"Row counts: `{topology.get('row_lengths', [])}`")
            st.write(f"Orientation: `{topology.get('orientation_strategy', 'mixed')}`")
            st.write(f"Strings: `{topology.get('string_count', 'n/a')}`")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="pv-card"><div class="pv-section-title">Recently Viewed</div><div class="pv-chip-row">', unsafe_allow_html=True)
            for key in st.session_state.get("recent_arrays", [])[:8]:
                st.markdown(f'<span class="pv-chip">{esc(key)}</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    elif section == "Faults":
        rgb_scene = get_rgb_scene()
        scene = get_scene()
        regions = get_regions()
        selected_region = get_selected_region()
        show_fault_boxes = bool(st.session_state.get("show_fault_boxes", False))
        show_cell_map = bool(st.session_state.get("show_cell_map", False))
        if focus_mode:
            if st.button("Exit Full Screen", use_container_width=False):
                qp_set(**navigation_params(
                    location["id"],
                    view,
                    selected_row_id,
                    selected_cell_id,
                    section="Faults",
                    selected_array=qp_get("selected_array", ""),
                    selected_module=qp_get("selected_module", ""),
                    cell_active=qp_get("cell_active", "0"),
                    focus=0,
                ))
                st.rerun()
            canvas_col = st.container()
            inspector_col = None
        else:
            canvas_col, inspector_col = st.columns([.82, .18])
        with canvas_col:
            if selected_region:
                crop_scene = scene
                if inspection_mode == "RGB":
                    crop_scene = rgb_scene
                elif inspection_mode == "Thermal":
                    if thermal_scene is None:
                        thermal_scene = get_thermal_scene()
                    crop_scene = thermal_scene
                if inspection_state in ("array", "module"):
                    st.markdown('<div class="pv-card"><div class="pv-section-title">Array Inspection</div>', unsafe_allow_html=True)
                    crumb = f'{esc(location["name"])} &gt; {esc(current_array_key)}'
                    if module_selected:
                        crumb += f' &gt; Module {selected_module}'
                    if cell_active:
                        crumb += f' &gt; {esc(selected_cell_id)}'
                    st.markdown(f'<div class="pv-focus-title">{crumb}</div>', unsafe_allow_html=True)
                    active_names = []
                    for active_record in thermal_records_for_distance(records, "close"):
                        if active_record.get("row_id") != selected_row_id:
                            continue
                        target = cell_by_id(location["id"], selected_row_id, active_record["target_cell_id"])
                        if int(target.get("panel_number") or 1) != int(selected_region["col"]):
                            continue
                        kind = canonical_fault_type(active_record["fault_type"])
                        label = fault_display_name(kind)
                        if kind in REAL_FARM_SYNTHETIC_DIAGNOSTIC_MODELS:
                            label += " · synthetic / unvalidated"
                        active_names.append(label)
                    st.caption(" | ".join(active_names) if active_names else "Healthy · no active faults")
                    if inspection_mode == "Compare":
                        if thermal_scene is None:
                            thermal_scene = get_thermal_scene()
                        show_crop_pair(rgb_scene, thermal_scene, selected_region.get("polygon") or box_polygon(selected_region["box"]), "RGB", "Thermal", 0.075)
                    else:
                        native_scene = close_source_scene(location["id"], "thermal" if inspection_mode == "Thermal" else "rgb" if inspection_mode == "RGB" else view,
                                                          selected_row_id, selected_cell_id, records, crop_scene)
                        if inspection_mode == "RGB" or (inspection_mode == "Current" and view == "rgb"):
                            from pv_far_surface import generated_rgb_scene
                            native_scene, _ = generated_rgb_scene(globals(), native_scene, location["id"], selected_row_id, selected_cell_id, records)
                        native_region = scale_region_to_image(selected_region, crop_scene.size, native_scene.size)
                        render_array_module_selector(
                            location,
                            view,
                            native_scene,
                            native_region,
                            selected_module if module_selected else 0,
                            selected_cell_id if cell_active else None,
                            records,
                            show_fault_boxes,
                            show_cell_map,
                        )
                    st.caption("Close View uses native model detail with four physical modules; click a cell to select it.")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pv-canvas-card">', unsafe_allow_html=True)
                    render_interactive_scene_canvas(
                        scene,
                        location,
                        view,
                        regions,
                        "",
                        records,
                        qp_get("filter", "Show All"),
                        show_fault_markers=show_fault_boxes,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="pv-canvas-card">', unsafe_allow_html=True)
                render_interactive_scene_canvas(
                    scene,
                    location,
                    view,
                    regions,
                    "",
                    records,
                    qp_get("filter", "Show All"),
                    show_fault_markers=show_fault_boxes,
                )
                st.markdown("</div>", unsafe_allow_html=True)
        if inspector_col is None:
            pass
        else:
            with inspector_col:
                st.markdown('<div class="pv-card"><div class="pv-section-title">Inspector</div>', unsafe_allow_html=True)
                if inspection_state in ("array", "module") and selected_region:
                    st.markdown('<div class="pv-context-thumb">', unsafe_allow_html=True)
                    st.image(scene_context_image(scene, selected_region), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    if st.button("Far View", use_container_width=True):
                        qp_set(**navigation_params(
                            location["id"],
                            view,
                            selected_row_id,
                            selected_cell_id,
                            section="Faults",
                            selected_array="",
                            selected_module="",
                            cell_active=0,
                        ))
                        st.rerun()
                if inspection_state == "module" and selected_region:
                    st.caption("Selected Array")
                    st.image(array_context_thumbnail(scene, selected_region, selected_module), use_container_width=True)
                    if st.button("Back to Array", use_container_width=True):
                        qp_set(**navigation_params(
                            location["id"],
                            view,
                            selected_row_id,
                            selected_cell_id,
                            section="Faults",
                            selected_array=current_array_key,
                            selected_module="",
                            cell_active=0,
                        ))
                        st.rerun()
                if inspection_state == "scene":
                    st.write("**Scene Selection**")
                    st.caption("Click an array in the main scene to open the large array inspection view.")
                    if st.button("Close View", type="primary", use_container_width=True):
                        enter_close_view()
                        st.rerun()
                elif selected_region:
                    st.write(f"**{current_array_key}**")
                    st.caption(f"Row {selected_region['row']} | Array {selected_region['col']} | {selected_region.get('orientation', 'unknown')} | String {selected_region.get('string_id') or 'n/a'}")
                    array_fault_count = fault_count_for_region(thermal_records_for_distance(records,"far"), selected_region)
                    st.caption(f"4 modules | {array_fault_count} fault(s)")
                if inspection_state != "scene" and module_selected:
                    st.write(f"**Module:** {selected_module}")
                elif inspection_state != "scene":
                    st.caption("Select a suspicious cell from the real array crop.")
                if cell_active:
                    st.write(f"**Cell:** `{selected_cell_id}`")
                    selected_faults = [
                        record for record in records
                        if record.get("row_id") == selected_row_id
                        and record.get("target_cell_id") == selected_cell_id
                    ]
                    far_observations = [record for record in selected_faults if record.get("fault_type") in far_observation_types]
                    diagnostic_faults = [record for record in selected_faults if record.get("fault_type") not in far_observation_types]
                    if far_observations:
                        st.caption("Observed thermal phenomenon")
                        for record in far_observations:
                            st.markdown(f'<span class="pv-chip">{esc(fault_display_name(record["fault_type"]))}</span>', unsafe_allow_html=True)
                        all_observations = [
                            record for record in records
                            if record.get("location_id") == location["id"]
                            and record.get("fault_type") in far_observation_types
                        ]
                        observation_index = next(
                            (
                                index for index, record in enumerate(all_observations)
                                if record.get("row_id") == selected_row_id
                                and record.get("target_cell_id") == selected_cell_id
                            ),
                            -1,
                        )
                        if observation_index >= 0:
                            st.caption(f"Hotspot {observation_index + 1} of {len(all_observations)}")
                    if diagnostic_faults:
                        st.caption("Assigned diagnostic fault(s)")
                        for record in diagnostic_faults:
                            st.markdown(f'<span class="pv-chip pv-chip-hot">{esc(fault_display_name(record["fault_type"]))}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("No diagnostic fault assigned to selected cell.")
                elif module_selected:
                    st.caption("Select a cell from the four-module array crop to edit diagnostic faults.")
                inspection_choice = st.radio(
                    "Crop View",
                    ["Current", "RGB", "Thermal", "Compare"],
                    index=["Current", "RGB", "Thermal", "Compare"].index(inspection_mode if inspection_mode in ["Current", "RGB", "Thermal", "Compare"] else "Current"),
                    horizontal=True,
                    key="inspection_crop_mode",
                )
                if inspection_choice != inspection_mode:
                    st.rerun()
                if st.button("Full Screen", use_container_width=True):
                    qp_set(**navigation_params(
                        location["id"],
                        view,
                        selected_row_id,
                        selected_cell_id,
                        section="Faults",
                        selected_array=qp_get("selected_array", ""),
                        selected_module=qp_get("selected_module", ""),
                        cell_active=qp_get("cell_active", "0"),
                        focus=1,
                    ))
                    st.rerun()
                st.checkbox("Show fault bounding boxes", value=show_fault_boxes, key="show_fault_boxes")
                st.checkbox("Show cell map", value=show_cell_map, key="show_cell_map")
                st.checkbox("Debug geometry overlay", value=bool(st.session_state.get("debug_geometry_overlay", False)), key="debug_geometry_overlay")
                if qp_get("debug_fault_visibility", "0") == "1" and cell_active:
                    debug_payload = live_fault_scalar_debug(
                        location["id"],
                        selected_row_id,
                        selected_cell_id,
                        records,
                        "close",
                    )
                    with st.expander("Developer Fault Visibility Debug", expanded=True):
                        if not debug_payload:
                            st.caption("No selected-cell diagnostic fault is available for scalar debug.")
                        else:
                            st.json(
                                {
                                    "fault_type": debug_payload["record"].get("fault_type"),
                                    "target_cell_id": debug_payload["record"].get("target_cell_id"),
                                    "severity": debug_payload["record"].get("fault_scale"),
                                    "applied": debug_payload["applied"],
                                    "reference_dispatch": production_submodule_enabled(debug_payload["record"].get("fault_type")),
                                    "support_bounds": debug_payload.get("support_bounds"),
                                    "metrics": debug_payload.get("metrics"),
                                }
                            )
                            dbg_cols = st.columns(5)
                            for dbg_col, (label, key) in zip(
                                dbg_cols,
                                [
                                    ("Healthy scalar", "healthy_scalar"),
                                    ("Faulted scalar", "faulted_scalar"),
                                    ("Delta", "delta"),
                                    ("Support", "support"),
                                    ("Final palette", "final_palette"),
                                ],
                            ):
                                with dbg_col:
                                    st.caption(label)
                                    st.image(debug_payload[key], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if inspection_state == "scene":
                    from pv_far_surface import generated_controls
                    if not experimental_thermal:
                        generated_controls(globals(), location, selected_row_id, selected_cell_id)
                    st.markdown('<div class="pv-card"><div class="pv-section-title">Far View Thermal Observations</div>', unsafe_allow_html=True)
                    st.caption("Optional thermal-only observations. Hotspots do not imply visible RGB scorching. Use Visible Far conditions above for paired surface appearance and thermal response.")
                    far_fault_types = [
                        fault_type for fault_type in globals().get(
                            "FAR_SCOUT_FAULT_TYPES",
                            ["SingleHotSpot", "MultiHotSpot", "PartialShading"],
                        )
                        if fault_available_in_location(location["id"], fault_type)
                    ]
                    if not far_fault_types:
                        far_fault_types = ["SingleHotSpot", "MultiHotSpot"]
                    if "modern_random_fault_types" in st.session_state:
                        st.session_state["modern_random_fault_types"] = [
                            canonical_fault_type(item)
                            for item in st.session_state["modern_random_fault_types"]
                            if canonical_fault_type(item) in far_fault_types
                        ] or [far_fault_types[0]]
                    scout_types = st.multiselect(
                        "Observation Type",
                        far_fault_types,
                        default=st.session_state.get("modern_random_fault_types", [far_fault_types[0]]),
                        format_func=fault_display_name,
                        key="modern_random_fault_types",
                    )
                    scout_max = max(1, min(50, int(metadata.get("total_cells") or 1)))
                    scout_count = st.number_input("Count", min_value=1, max_value=scout_max, value=min(3, scout_max), step=1, key="modern_random_fault_count")
                    scout_seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="modern_random_fault_seed")
                    if st.button("Generate Observations", use_container_width=True, disabled=experimental_thermal or not scout_types):
                        available_cells = fault_target_cells(location["id"])
                        if not available_cells:
                            st.warning("No available cells for this scene.")
                            st.stop()
                        push_fault_history()
                        rng = random.Random(seed_for(location["id"], view, "|".join(scout_types), int(scout_count), int(scout_seed), "modern-main-hotspots"))
                        chosen = rng.sample(available_cells, min(int(scout_count), len(available_cells)))
                        st.session_state["random_fault_records"] = [
                            fault_record(location, view, row, cell, rng.choice(scout_types), fault_scale)
                            for row, cell in chosen
                        ]
                        st.rerun()
                    if st.button("Clear Observations", use_container_width=True, disabled=not st.session_state.get("random_fault_records")):
                        push_fault_history()
                        st.session_state["random_fault_records"] = []
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                if cell_active:
                    st.markdown('<div class="pv-card"><div class="pv-section-title">Fault Editor</div>', unsafe_allow_html=True)
                    available_fault_types = [
                        fault_type for fault_type in globals().get("CLOSE_DIAGNOSTIC_FAULT_TYPES", selectable_fault_types_for_location(location["id"]))
                        if fault_available_in_location(location["id"], fault_type)
                        and fault_type not in far_observation_types
                    ] or [
                        fault_type for fault_type in selectable_fault_types_for_location(location["id"])
                        if fault_type not in far_observation_types
                    ]
                    fault_type = st.selectbox("Diagnostic Fault Type", available_fault_types, format_func=fault_display_name, key="new_fault_type")
                    if fault_type in REAL_FARM_SYNTHETIC_DIAGNOSTIC_MODELS:
                        st.caption("Synthetic relative diagnostic morphology; not a production-validated diagnosis or measured temperature.")
                    severity = st.number_input(
                        "Severity", min_value=1, max_value=10,
                        value=int(DEFAULT_FAULT_SCALE), step=1, key="generated_fault_severity",
                    )
                    add_col, clear_col = st.columns(2)
                    with add_col:
                        if st.button("Apply Fault", type="primary", use_container_width=True, disabled=experimental_thermal):
                            push_fault_history()
                            existing = [
                                record for record in st.session_state.get("panel_cell_fault_records", [])
                                if not (
                                    record.get("location_id") == location["id"]
                                    and record.get("row_id") == selected_row_id
                                    and record.get("target_cell_id") == selected_cell_id
                                )
                            ]
                            existing.append(fault_record(location, view, selected_row, selected_cell, fault_type, severity))
                            st.session_state["panel_cell_fault_records"] = existing
                            st.toast(f"Fault applied to {selected_cell_id}")
                            st.rerun()
                    with clear_col:
                        if st.button("Clear Cell", use_container_width=True):
                            push_fault_history()
                            st.session_state["panel_cell_fault_records"] = [
                                record for record in st.session_state.get("panel_cell_fault_records", [])
                                if not (
                                    record.get("location_id") == location["id"]
                                    and record.get("row_id") == selected_row_id
                                    and record.get("target_cell_id") == selected_cell_id
                                )
                            ]
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="pv-card"><div class="pv-section-title">Fault Navigation</div>', unsafe_allow_html=True)
                observation_records = [
                    record for record in records
                    if record.get("location_id") == location["id"]
                    and record.get("fault_type") in far_observation_types
                ]
                observation_idx = next(
                    (
                        i for i, record in enumerate(observation_records)
                        if record.get("row_id") == selected_row_id
                        and record.get("target_cell_id") == selected_cell_id
                    ),
                    -1,
                )
                if inspection_state != "scene" and observation_records:
                    current_observation_number = observation_idx + 1 if observation_idx >= 0 else 0
                    st.caption(f"Generated hotspots: {len(observation_records)} | Current: {current_observation_number or 'none'}")
                    hotspot_options = list(range(len(observation_records)))
                    selected_hotspot_option = observation_idx if observation_idx >= 0 else 0
                    hotspot_choice = st.selectbox(
                        "Select Generated Hotspot",
                        hotspot_options,
                        index=selected_hotspot_option,
                        format_func=lambda index: (
                            f"Hotspot {index + 1} | "
                            f"{observation_records[index].get('row_id')} | "
                            f"Array {observation_records[index].get('panel_number_in_row')} | "
                            f"M{observation_records[index].get('module_number_in_panel')} | "
                            f"Cell {observation_records[index].get('module_cell_number')}"
                        ),
                        # Scope widget state to the active target. Otherwise a
                        # Next/Previous navigation rerun replays the old selectbox
                        # value and immediately jumps back to the previous fault.
                        key=(
                            f"hotspot_target_{location['id']}_{len(observation_records)}_"
                            f"{selected_row_id}_{selected_cell_id}"
                        ),
                    )
                    if hotspot_choice != observation_idx:
                        jump_to_fault(observation_records[hotspot_choice])
                        st.rerun()
                    hotspot_prev_col, hotspot_next_col = st.columns(2)
                    with hotspot_prev_col:
                        if st.button("Previous Hotspot", use_container_width=True):
                            previous_index = observation_idx - 1 if observation_idx >= 0 else -1
                            jump_to_fault(observation_records[previous_index % len(observation_records)])
                            st.rerun()
                    with hotspot_next_col:
                        if st.button("Next Hotspot", use_container_width=True):
                            next_index = observation_idx + 1 if observation_idx >= 0 else 0
                            jump_to_fault(observation_records[next_index % len(observation_records)])
                            st.rerun()
                faulty_records = [
                    record for record in records
                    if record.get("location_id") == location["id"]
                ]
                idx = next(
                    (
                        i for i, record in enumerate(faulty_records)
                        if record.get("row_id") == selected_row_id
                        and record.get("target_cell_id") == selected_cell_id
                    ),
                    -1,
                )
                array_idx = next((i for i, region in enumerate(regions) if region["key"] == current_array_key), -1)
                array_prev_col, array_next_col = st.columns(2)
                with array_prev_col:
                    if st.button("Previous Array", use_container_width=True, disabled=not regions):
                        jump_to_region(regions[(array_idx - 1) % len(regions)])
                        st.rerun()
                with array_next_col:
                    if st.button("Next Array", use_container_width=True, disabled=not regions):
                        jump_to_region(regions[(array_idx + 1) % len(regions)])
                        st.rerun()
                if inspection_state != "scene":
                    prev_col, next_col = st.columns(2)
                    with prev_col:
                        if st.button("Previous Fault", use_container_width=True, disabled=not faulty_records):
                            jump_to_fault(faulty_records[(idx - 1) % len(faulty_records)])
                            st.rerun()
                    with next_col:
                        if st.button("Next Fault", use_container_width=True, disabled=not faulty_records):
                            jump_to_fault(faulty_records[(idx + 1) % len(faulty_records)])
                            st.rerun()
                recent_regions = [
                    region for key in st.session_state.get("recent_arrays", [])[:6]
                    for region in regions
                    if region["key"] == key
                ]
                if recent_regions:
                    st.caption("Recently Viewed")
                    recent_cols = st.columns(min(3, len(recent_regions)))
                    for i, region in enumerate(recent_regions):
                        with recent_cols[i % len(recent_cols)]:
                            if st.button(region["key"], key=f"recent_array_{region['key']}", use_container_width=True):
                                jump_to_region(region)
                                st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="pv-card"><div class="pv-section-title">Search</div>', unsafe_allow_html=True)
                query = st.text_input("Search Panel / Cell ID", placeholder="R3-A02 or R3-C017")
                if st.button("Go", use_container_width=True):
                    parsed = parse_panel_search(query)
                    if parsed and parsed[0] == "array":
                        _, row_num, array_col = parsed
                        row_id, cell = default_cell_for_array(location["id"], row_num, array_col)
                        qp_set(**navigation_params(location["id"], view, row_id, cell["id"], section="Faults", selected_array=f"R{row_num}-A{array_col:02d}", selected_module="", cell_active=0))
                        st.rerun()
                    elif parsed and parsed[0] == "cell":
                        _, row_num, cell_num = parsed
                        row_id = f"{row_prefix(location['id'])}{row_num}"
                        cell_id = f"{row_id}-C{cell_num:03d}"
                        cell = cell_by_id(location["id"], row_id, cell_id)
                        qp_set(**navigation_params(location["id"], view, row_id, cell["id"], section="Faults", selected_array=f"R{row_num}-A{int(cell.get('panel_number') or 1):02d}", selected_module=cell.get("module_number_in_panel", 1), cell_active=1))
                        st.rerun()
                    else:
                        st.warning("Use a form like R1-A03 or R1-C017.")
                st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Review":
        summary_col, analysis_col = st.columns([.45, .55])
        with summary_col:
            st.markdown('<div class="pv-card"><div class="pv-section-title">Fault Summary</div>', unsafe_allow_html=True)
            summary = fault_summary_rows(records)
            if summary:
                st.table(summary)
            else:
                st.caption("No faults have been assigned yet.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="pv-card"><div class="pv-section-title">Fault Legend</div><div class="pv-chip-row">', unsafe_allow_html=True)
            for fault_type in selectable_fault_types_for_location(location["id"]):
                st.markdown(f'<span class="pv-chip">{esc(fault_display_name(fault_type))}</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        with analysis_col:
            st.markdown('<div class="pv-card"><div class="pv-section-title">AI Classifier Check</div>', unsafe_allow_html=True)
            if view == "thermal" and not experimental_thermal:
                scene = get_thermal_scene()
                render_classifier_check(scene)
            elif experimental_thermal:
                st.caption("Fault classification is unsupported for the experimental healthy renderer.")
            else:
                st.caption("Switch to Thermal view to run the classifier check on the current thermal scene.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="pv-card"><div class="pv-section-title">Fault List</div>', unsafe_allow_html=True)
        st.dataframe(fault_table_rows(records), use_container_width=True, hide_index=True)
        with st.expander("Advanced Data", expanded=False):
            if st.checkbox("Load advanced metadata and selected-row cells", value=False):
                st.json(load_solar_render_metadata(location["id"]) or {})
                st.dataframe(
                    [
                        {
                            "pv_cell_id": cell["id"],
                            "row": selected_row_id,
                            "array": cell.get("panel_number"),
                            "module": cell.get("module_number_in_panel"),
                            "module_cell": cell.get("module_cell_number"),
                            "x1": cell["x1"],
                            "y1": cell["y1"],
                            "x2": cell["x2"],
                            "y2": cell["y2"],
                        }
                        for cell in row_cells
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Advanced metadata is loaded on demand to keep the Review page responsive.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "DATA COLLECTION":
        from pv_data_collection import render_data_collection
        from pv_data_collection_prepare import prepare_generated, restore_previous

        render_data_collection("Generated", on_prepare=lambda case, options: prepare_generated(globals(), case, options),
            generated_scenes=LOCATIONS, selected_scene=location["id"],
            on_restore_simulator=lambda: restore_previous(globals()),
            on_restore_original=lambda: restore_previous(globals(), original=True), simulator_ns=globals())

    elif section == "Export":
        scene = get_scene()
        selected_region = get_selected_region()
        if experimental_thermal:
            st.markdown("#### Experimental healthy export")
            st.download_button("Export Experimental Healthy View", export_current_view_bytes(scene),
                               file_name=f"{location['id']}_reference_healthy.png", mime="image/png")
            st.caption("Healthy preview PNG only. Switch to Current Thermal for fault datasets and metadata.")
            return
        st.markdown('<div class="pv-card"><div class="pv-section-title">Export</div>', unsafe_allow_html=True)
        export_records, export_modules, export_support = generated_export_geometry(
            location["id"], scene, get_regions(), get_rgb_scene().size, records)
        export_region = scale_region_to_image(selected_region,get_rgb_scene().size,scene.size) if selected_region else None
        crop_image, crop_bounds = crop_from_polygon(scene, export_region.get("polygon") or box_polygon(export_region["box"]), 0.075) if export_region else (scene, (0, 0, scene.width, scene.height))
        export_cols = st.columns(4)
        with export_cols[0]:
            st.download_button("Export Current View", export_current_view_bytes(scene), file_name=f"{location['id']}_{view}_scene.png", mime="image/png", use_container_width=True)
        with export_cols[1]:
            st.download_button("Export Selected Crop", export_current_view_bytes(crop_image), file_name=f"{location['id']}_{current_array_key}_{view}.png", mime="image/png", use_container_width=True)
        with export_cols[2]:
            st.download_button("Export CSV", fault_records_csv(export_records), file_name="faults.csv", mime="text/csv", use_container_width=True)
        with export_cols[3]:
            healthy_export = module_field_image(location["id"], "thermal", selected_row_id, selected_cell_id, [], show_grid=False) if view == "thermal" else None
            source_rgb_export = module_field_image(location["id"], "rgb", selected_row_id, selected_cell_id, [], show_grid=False).convert("RGB")
            dataset = generated_geometry_dataset(location, view, selected_row, selected_cell, export_records, scene, source_rgb_export, crop_image, crop_bounds, export_modules, export_support, healthy_export)
            st.download_button("Export Dataset", dataset, file_name="pv_fault_dataset.zip", mime="application/zip", use_container_width=True, on_click="ignore")
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Settings":
        st.markdown('<div class="pv-card"><div class="pv-section-title">Settings</div>', unsafe_allow_html=True)
        st.caption("Generation settings are intentionally lightweight in the UI. Blender generation, topology, and safe placement are controlled by the renderer scripts.")
        st.write(f"Custom layout: `{custom_layout_enabled()}`")
        st.write(f"Rows: `{custom_row_count(location['id'])}`")
        st.write(f"Nominal arrays per row: `{custom_panels_per_row(location['id'])}`")
        if st.button("Open Legacy Interface", use_container_width=True, disabled=experimental_thermal):
            qp_set(**navigation_params(location["id"], view, selected_row_id, selected_cell_id, section="legacy"))
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Help":
        st.markdown(
            """
            <div class="pv-card">
              <div class="pv-section-title">Workflow</div>
              <p class="pv-muted">Choose a scene, set rows and arrays per row, inspect the generated installation, click real arrays, drill into real crops, assign faults, review, and export.</p>
              <p class="pv-muted">The UI uses the existing RGB, thermal, fault, and export pipeline. It does not rerender Blender during inspection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
