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
        f'<image href="{uri}" width="{BASE_SIZE[0]}" height="{BASE_SIZE[1]}" preserveAspectRatio="xMidYMid slice"/>',
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
          .frame {{ width:100%; overflow:hidden; border:1px solid #263244; border-radius:10px; background:#050816; }}
          svg {{ width:100%; height:auto; display:block; }}
          a {{ cursor:pointer; }}
          a:hover rect {{ opacity:.55; stroke:#38bdf8; stroke-width:2; }}
        </style>
        """,
        height=430,
        scrolling=False,
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
    cols = CUSTOM_MODULE_CELL_ROWS   # display columns (8 for a 40-cell module laid landscape)
    rows = CUSTOM_MODULE_CELL_COLS   # display rows (5)
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


def draw_close_thermal_module(module_cells, selected_cell_id, fault_records, size, location_id):
    width, height = size
    geo = close_module_geometry(size)
    rects = close_module_cell_rects(module_cells, size)
    cell_map = {c["id"]: c for c in module_cells}
    pitch = geo["pitch"]
    rng = np.random.default_rng(seed_for(location_id, selected_cell_id, "close-thermal-module"))

    # Background (sky / ground behind the module)
    temp = np.full((height, width), 70, np.float32)
    temp += _noise_field((height, width), seed_for(location_id, "close-thermal-bg"), 90, 2) * 10

    # Frame (cooler, metal) and backsheet gaps
    fx1, fy1, fx2, fy2 = geo["frame"]
    ix1, iy1, ix2, iy2 = geo["inner"]
    temp[fy1:fy2, fx1:fx2] = 104
    temp[iy1:iy2, ix1:ix2] = 126
    iw, ih = ix2 - ix1, iy2 - iy1

    # Cells: healthy baseline in the red/orange band, with clear cell-to-cell thermal variety.
    for cell in module_cells:
        x1, y1, x2, y2 = rects[cell["id"]]
        w, h = x2 - x1, y2 - y1
        row_pos = (float(cell.get("module_cell_col") or 1) - 1.0) / max(1.0, geo["rows"] - 1.0)
        col_pos = (float(cell.get("module_cell_row") or 1) - 1.0) / max(1.0, geo["cols"] - 1.0)
        base = (
            166
            + 7.5 * (1.0 - row_pos)
            + 3.5 * math.sin((col_pos * math.pi * 2.0) + float(rng.normal(0, 0.20)))
            + float(rng.normal(0, 3.4))
        )
        ex = 1 - np.abs(np.linspace(-1, 1, w, dtype=np.float32))[None, :]
        ey = 1 - np.abs(np.linspace(-1, 1, h, dtype=np.float32))[:, None]
        bowl = np.clip(np.minimum(ex, ey) * 4, 0, 1)
        local_haze = (_noise_field((h, w), seed_for(cell["id"], location_id, "thermal-cell-haze"), max(5.0, min(w, h) * 0.45), 2) - 0.5) * 5.0
        tile = base + 5.0 * bowl - 4.0 + local_haze
        m = pv_cell_chamfer_mask(w, h)
        region = temp[y1:y2, x1:x2]
        region[:] = region * (1 - m) + tile * m

    # Thermal dispersion / Hughes-like hue variety: broad gradients and soft warm/cool islands
    # are added before the LUT, so the final colors come from temperature differences.
    yy, xx = np.mgrid[0:ih, 0:iw].astype(np.float32)
    nx = (xx / max(1, iw - 1)) * 2.0 - 1.0
    ny = (yy / max(1, ih - 1)) * 2.0 - 1.0
    tilt_gradient = -7.0 * ny + 3.4 * nx
    lowfreq = (_noise_field((ih, iw), seed_for(location_id, selected_cell_id, "close-thermal-low"), iw * 0.34, 4) - 0.5) * 14.0
    island = np.zeros((ih, iw), np.float32)
    for idx in range(4):
        cx = iw * float(rng.uniform(0.12, 0.88))
        cy = ih * float(rng.uniform(0.12, 0.88))
        sx = iw * float(rng.uniform(0.14, 0.34))
        sy = ih * float(rng.uniform(0.12, 0.30))
        amp = float(rng.uniform(-7.0, 9.0))
        island += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
    edge_cool = -5.2 * np.clip((np.abs(nx) - 0.72) / 0.28, 0, 1)
    edge_cool += -4.2 * np.clip((np.abs(ny) - 0.74) / 0.26, 0, 1)
    temp[iy1:iy2, ix1:ix2] += tilt_gradient + lowfreq + island + edge_cool
    _add_gaussian(temp, (ix1 + ix2) / 2, iy1 + ih * 0.12, iw * 0.07, ih * 0.09, 12.0)

    rows_per_sub = max(1, geo["rows"] // 3)

    for record in fault_records or []:
        tid = record.get("target_cell_id")
        if tid not in rects:
            continue
        x1, y1, x2, y2 = rects[tid]
        cw, ch = x2 - x1, y2 - y1
        ft = canonical_fault_type(record["fault_type"])
        sev = clamp(record.get("fault_scale", 8), 1, 10) / 10
        row_idx = int(cell_map[tid].get("module_cell_col") or 1) - 1
        sub_idx = clamp(row_idx // rows_per_sub, 0, 2)

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
            for _ in range(spots):
                cx = x1 + cw * rnd.uniform(0.34, 0.66)
                cy = y1 + ch * rnd.uniform(0.34, 0.66)
                s = 0.17 if spots == 1 else 0.11
                _add_gaussian(temp, cx, cy, cw * s, ch * s, 60 + 50 * sev)
            temp[y1:y2, x1:x2] += 6 + 8 * sev
            _add_gaussian(temp, (x1 + x2) / 2, (y1 + y2) / 2, pitch * 0.6, pitch * 0.6, 8 + 10 * sev)

    # Camera model: optics blur, sensor noise, banding, vignette, dithering
    temp = cv2.GaussianBlur(temp, (0, 0), 1.3)
    temp += cv2.GaussianBlur(rng.normal(0, 4.5, temp.shape).astype(np.float32), (0, 0), 0.8)
    temp += rng.normal(0, 0.8, (height, 1)).astype(np.float32)
    temp += rng.normal(0, 0.6, (1, width)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
    temp -= r2 * 5.0
    temp += rng.uniform(-0.5, 0.5, temp.shape).astype(np.float32)

    lut = np.array(thermal_lut(), dtype=np.uint8)
    rgb = lut[np.clip(temp, 0, 255).astype(np.uint8)]
    return Image.fromarray(rgb, "RGB")


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
    fault_cell_ids = {record.get("target_cell_id") for record in (fault_records or [])}
    num_font = grid_font(12)

    # FIX: no thick boxes on every cell. Corner brackets mark selection / faults; grid is opt-in.
    for cell in module_cells:
        rect = rects[cell["id"]]
        selected = cell["id"] == selected_cell_id
        faulted = cell["id"] in fault_cell_ids
        if show_grid:
            draw.rectangle(rect, outline=(224, 242, 254, 70), width=1)
            label = str(cell.get("pv_cell_number_in_row", ""))
            if label:
                draw.text((rect[0] + 4, rect[1] + 3), label, fill=(255, 255, 255, 190), font=num_font)
        if faulted and not selected:
            draw_corner_marks(draw, rect, (56, 189, 248, 235), width=2)
        if selected:
            draw_corner_marks(draw, rect, (250, 204, 21, 255), width=3)

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


@st.cache_data(show_spinner=False, max_entries=512)
def cached_module_preview_face(location_id, view, selected_row_id, selected_cell_id, fault_key):
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
        module_img = draw_close_thermal_module(module_cells, selected_cell_id, records, size, location_id)
    else:
        module_img = draw_close_rgb_module(module_cells, selected_cell_id, records, size, location_id)
    return crop_module_face(module_img, margin=5)


def solar_panel_tile_image(location_id, view, selected_row_id, panel_number, selected_cell_id, fault_records=None):
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


def module_field_image(location_id, view, selected_row_id, selected_cell_id, fault_records=None, show_grid=True):
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
    field = module_field_background(location_id, view, (field_w, field_h), selected_row_id).convert("RGBA")
    draw = ImageDraw.Draw(field, "RGBA")

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
            tile = solar_panel_tile_image(location_id, view, row["id"], panel_number, tile_selected_cell_id, records)
            tile = tile.resize((tile_w, tile_h), Image.Resampling.BICUBIC)
            row_offset_x = 0
            if location_id == "agri_field_new":
                row_offset_x = [70, 70, 70, 70, 70][row_idx % 5]
            x = pad_x + row_offset_x + col_idx * (tile_w + gap_x)
            y = pad_y + row_idx * (tile_h + gap_y)
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
            if is_selected:
                selected_rect = (x, y, x + tile_w, y + tile_h)

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


def main():
    st.set_page_config(page_title="PV Array Fault Tool", layout="wide")
    st.title("PV Array Fault Tool")
    st.caption("Select row, panel, module, and cell. The main view shows solar panels made from four 40-cell modules.")

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

    main_tab, output_tab, big_screen_tab = st.tabs(["Main", "Output", "Big Screen Test"])

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
        target_col, panel_fault_col = st.columns([1, 1])
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
                    "Panel in selected row",
                    panel_numbers,
                    index=panel_numbers.index(selected_panel_number),
                    format_func=lambda panel_number: f"Panel {panel_number}",
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
                    "Module inside selected panel (1 of 4)",
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
            st.subheader("Rows and Panels")
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
                "Panels per row",
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
            st.caption(
                f"{PV_PANEL_MODEL_NAME}: {int(desired_panels)} panels per row x "
                f"{SOLAR_PANEL_MODULES} modules/panel x {cells_per_module} cells/module "
                f"= {int(desired_panels) * cells_per_panel} selectable cells per row."
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
            f"panels/row: {main_metadata['panels_per_row']} | modules/row: {main_metadata['modules_per_row']} | "
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
            f"Row {selected_row['label']} | selected panel {selected_cell.get('panel_number')} | "
            f"selected module {selected_cell.get('module_number_in_panel', selected_module_number)}. "
            f"Each visible tile is one solar panel: {SOLAR_PANEL_MODULES} modules x "
            f"{CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS} cells/module."
        )
    with output_tab:
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
            f"panels/row: {metadata['panels_per_row']} | modules/row: {metadata['modules_per_row']} | "
            f"faults: {metadata['fault_count']}"
        )
        if big_screen_thermal is not None:
            st.image(big_screen_thermal, use_container_width=True)
        else:
            st.caption("Thermal preview is generated when Camera view is set to Thermal.")
