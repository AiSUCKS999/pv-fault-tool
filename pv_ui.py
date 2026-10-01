# Streamlit UI, maps, tabs, controls, and app main().
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
        "Real-photo compositor: default mode uses realistic generated PV rows with controllable strings/modules; prepared source layouts are still available."
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
        st.caption("Use the Main tab for layout, target selection, random faults, and image review.")

    main_tab, output_tab, big_screen_tab = st.tabs(["Main", "Output", "Big Screen Test"])

    fault_scale = DEFAULT_FAULT_SCALE
    fault_type = "PartialShading"
    manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
    lock_manual_target = True

    with main_tab:
        layout_col, target_col = st.columns([1, 1])
        with layout_col:
            st.subheader("Rows and Modules")
            current_custom_layout = custom_layout_enabled()
            custom_choice = st.checkbox("Realistic generated PV layout", value=current_custom_layout)
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
                f"{PV_MODULE_MODEL_NAME}: {int(desired_panels)} modules per row x "
                f"{cells_per_module} selectable PV cells/module "
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
            available_fault_types = selectable_fault_types_for_location(location["id"])
            default_random_faults = ["PartialShading"] if "PartialShading" in available_fault_types else [available_fault_types[0]]
            if "PartialShading" not in available_fault_types:
                st.info(
                    "Vegetation / partial shading is not available in this scene. "
                    "It is limited to the grass-field and agrivoltaic scenes for realism."
                )
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

            fault_type = st.selectbox(
                "Fault type",
                available_fault_types,
                format_func=fault_display_name,
            )
            manual_record = fault_record(location, view, selected_row, selected_cell, fault_type, fault_scale)
            lock_manual_target = st.checkbox("Lock manual target into generated fault set", value=True)
            st.write(FAULT_TYPES[fault_type])
            st.caption(
                "Hotspot is treated as the thermal symptom, not the selectable root-cause fault. "
                "Choose the suspected cause that would create the hot region."
            )
            st.metric("Selected row", selected_row["label"])
            st.metric("Selected PV cell", selected_cell["id"])
            if selected_cell.get("module_number"):
                st.caption(
                    f"{PV_MODULE_MODEL_NAME} module {selected_cell['module_number']} - "
                    f"cell {selected_cell['module_cell_number']} "
                    f"({CUSTOM_MODULE_CELL_ROWS}x{CUSTOM_MODULE_CELL_COLS} module-cell grid)"
                )

        st.divider()
        rng_col, status_col = st.columns([1, 1])
        with rng_col:
            st.subheader("Random Fault Generator")
            if "random_fault_types" in st.session_state:
                st.session_state["random_fault_types"] = [
                    canonical_fault_type(item)
                    for item in st.session_state["random_fault_types"]
                    if fault_available_in_location(location["id"], canonical_fault_type(item))
                ] or default_random_faults
            random_types = st.multiselect(
                "Random fault types",
                available_fault_types,
                default=default_random_faults,
                format_func=fault_display_name,
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
            counts_by_patch = fault_patch_counts()
            direct_count = counts_by_patch.get(fault_type, 0)
            fallback_class = next((name for name in PATCH_FALLBACKS.get(fault_type, []) if counts_by_patch.get(name, 0)), "")
            if direct_count:
                st.caption(f"Patch bank: {direct_count} direct starter patch(es) for {fault_display_name(fault_type)}.")
            elif fallback_class:
                st.caption(
                    f"Patch bank: using {fault_display_name(fallback_class)} starter patch fallback "
                    f"for {fault_display_name(fault_type)}."
                )
            else:
                st.caption("Patch bank: no local patch for this class yet; procedural fallback is used.")
            with st.expander("Local patch bank status", expanded=False):
                manifest = patch_bank_manifest()
                if manifest:
                    st.caption(manifest.get("quality", "local patch bank"))
                    st.table(patch_bank_status_rows())
                    if manifest.get("limitations"):
                        st.caption("Limitations: " + " ".join(manifest["limitations"][:2]))
                else:
                    st.caption("No local patch bank found yet. Run tools/build_starter_patch_bank.py.")

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

    with main_tab:
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
            render_photo_map(pv_scene, location["id"], view, selected_row_id, selected_cell_id, True, show_bounding_boxes, display_records)
            st.caption("Click a row in this image. The selected row opens in the output tab for cell-level inspection.")
        with st.expander("AI classifier check on generated thermal scene", expanded=True):
            render_classifier_check(big_screen_thermal)
        with st.expander("Scene metadata", expanded=False):
            st.json(metadata)

    with output_tab:
        st.subheader("Fault Summary")
        summary_rows = fault_summary_rows(display_records)
        if summary_rows:
            st.table(summary_rows)
        else:
            st.caption("No faults selected.")

        st.subheader("AI Classifier Check")
        render_classifier_check(big_screen_thermal)

        st.subheader("Fault Table")
        st.table(fault_table_rows(display_records))

        st.subheader(f"Cropped row: {selected_row['label']} - selected PV cell {selected_cell['id']}")
        render_zoom_map(pv_scene, location["id"], view, selected_row_id, selected_cell_id)
        st.caption("Click an individual PV cell in the crop, or use the target selector in the Main tab.")

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
