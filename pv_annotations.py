# Fault records, bounding boxes, tables, metadata, exports, and stored records.
def fault_record(location, view, row, cell, fault_type, fault_scale):
    fault_type = canonical_fault_type(fault_type)
    row_id = row.get("id", row.get("row"))
    row_label = row.get("label", f"Row {row_id}")
    cx = round((cell["x1"] + cell["x2"]) / 2, 2)
    cy = round((cell["y1"] + cell["y2"]) / 2, 2)
    module_bbox = module_bbox_for_cell(location["id"], row_id, cell)
    visible_bbox = fault_visible_bbox_for_cell(cell_with_fault_bbox(cell, {"fault_bbox_px": module_bbox}), fault_type)
    return {
        "location_id": location["id"],
        "location_name": location["name"],
        "view": view.upper(),
        "row_id": row_id,
        "row_label": row_label,
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
        "affected_cell_ids": [cell["id"]],
        "affected_cell_count": 1,
        "scope_note": fault_scope_note(fault_type),
        "layout": {
            "custom": custom_layout_enabled(),
            "row_count": len(panels(location["id"])),
            "panels_per_row": custom_panels_per_row(location["id"]) if custom_layout_enabled() else len(cells_for_row(location["id"], row_id)),
            "pv_cells_per_row": len(cells_for_row(location["id"], row_id)),
            "cells_per_module": CUSTOM_MODULE_CELL_COLS * CUSTOM_MODULE_CELL_ROWS if custom_layout_enabled() else None,
        },
        "bbox_px": {
            "x1": cell["x1"],
            "y1": cell["y1"],
            "x2": cell["x2"],
            "y2": cell["y2"],
        },
        "fault_bbox_px": module_bbox,
        "visible_fault_bbox_px": visible_bbox,
        "center_px": {"x": cx, "y": cy},
        "center_normalized": {
            "x": round(cx / BASE_SIZE[0], 4),
            "y": round(cy / BASE_SIZE[1], 4),
        },
    }


def fault_record_csv(record):
    affected_cell_ids = record.get("affected_cell_ids") or [record["target_cell_id"]]
    visible_bbox = record.get("visible_fault_bbox_px") or record["bbox_px"]
    flat = {
        "location_id": record["location_id"],
        "location_name": record["location_name"],
        "view": record["view"],
        "row_id": record["row_id"],
        "target_cell_id": record["target_cell_id"],
        "affected_cell_ids": ";".join(affected_cell_ids),
        "affected_cell_count": record.get("affected_cell_count", len(affected_cell_ids)),
        "pv_cell_number_in_row": record["pv_cell_number_in_row"],
        "pv_cells_in_selected_row": record["pv_cells_in_selected_row"],
        "module_number_in_row": record["module_number_in_row"],
        "module_cell_number": record["module_cell_number"],
        "fault_type": record["fault_type"],
        "fault_label": fault_display_name(record["fault_type"]),
        "fault_scale": record["fault_scale"],
        "scope_note": record.get("scope_note", fault_scope_note(record["fault_type"])),
        "x1": record["bbox_px"]["x1"],
        "y1": record["bbox_px"]["y1"],
        "x2": record["bbox_px"]["x2"],
        "y2": record["bbox_px"]["y2"],
        "visible_fault_x1": visible_bbox["x1"],
        "visible_fault_y1": visible_bbox["y1"],
        "visible_fault_x2": visible_bbox["x2"],
        "visible_fault_y2": visible_bbox["y2"],
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
        affected_cell_ids = record.get("affected_cell_ids") or [record["target_cell_id"]]
        rows.append(
            {
                "#": idx,
                "location": record["location_name"],
                "view": record["view"],
                "row": record["row_label"],
                "pv_cell_id": record["target_cell_id"],
                "affected_cells": ", ".join(affected_cell_ids),
                "affected_count": record.get("affected_cell_count", len(affected_cell_ids)),
                "cell_number": record["pv_cell_number_in_row"],
                "module": record["module_number_in_row"],
                "module_cell": record["module_cell_number"],
                "fault_type": fault_display_name(record["fault_type"]),
                "scope_note": record.get("scope_note", fault_scope_note(record["fault_type"])),
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
    return [{"fault_type": fault_display_name(key), "count": value} for key, value in fault_type_counts(records).items()]


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
    fault_types = [
        canonical_fault_type(fault_type)
        for fault_type in fault_types
        if fault_available_in_location(location["id"], canonical_fault_type(fault_type))
    ]
    if not fault_types:
        return []
    shuffled_types = list(fault_types)
    rng.shuffle(shuffled_types)
    candidates_by_module = {}
    for row in panels(location["id"]):
        for cell in cells_for_row(location["id"], row["id"]):
            key = (row["id"], cell.get("module_number"), cell.get("module_cell_row"))
            candidates_by_module.setdefault(key, []).append((row, cell))

    module_keys = list(candidates_by_module)
    rng.shuffle(module_keys)
    selected = []
    used_cells = set()

    # First pass: spread random faults across different modules/half-modules.
    for key in module_keys:
        choices = candidates_by_module[key]
        rng.shuffle(choices)
        for row, cell in choices:
            if cell["id"] not in used_cells:
                selected.append((row, cell))
                used_cells.add(cell["id"])
                break
        if len(selected) >= fault_count:
            break

    # Second pass only if the user asks for more faults than we can spread cleanly.
    if len(selected) < fault_count:
        remaining = [
            (row, cell)
            for choices in candidates_by_module.values()
            for row, cell in choices
            if cell["id"] not in used_cells
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, fault_count - len(selected))])

    return [
        fault_record(location, view, row, cell, shuffled_types[idx % len(shuffled_types)], fault_scale)
        for idx, (row, cell) in enumerate(selected)
    ]


def normalized_fault_record(record):
    fault_type = canonical_fault_type(record.get("fault_type"))
    if fault_type in HIDDEN_SYMPTOM_FAULTS:
        fault_type = "PartialShading"
    if fault_type in LOW_CONFIDENCE_LEGACY_FAULTS:
        return None
    if fault_type not in FAULT_TYPES:
        return None
    if not fault_available_in_location(record.get("location_id"), fault_type):
        return None
    normalized = dict(record)
    normalized["fault_type"] = fault_type
    normalized.setdefault("affected_cell_ids", [normalized.get("target_cell_id")])
    normalized["affected_cell_ids"] = [
        cell_id for cell_id in normalized.get("affected_cell_ids", [])
        if cell_id
    ] or [normalized.get("target_cell_id")]
    normalized["affected_cell_count"] = len(normalized["affected_cell_ids"])
    normalized["scope_note"] = fault_scope_note(fault_type)
    if "fault_bbox_px" not in normalized:
        try:
            row = row_by_id(normalized["location_id"], normalized["row_id"])
            cell = cell_by_id(normalized["location_id"], row["id"], normalized["target_cell_id"])
            normalized["fault_bbox_px"] = module_bbox_for_cell(normalized["location_id"], row["id"], cell)
        except (KeyError, TypeError):
            pass
    if "visible_fault_bbox_px" not in normalized:
        try:
            row = row_by_id(normalized["location_id"], normalized["row_id"])
            cell = cell_by_id(normalized["location_id"], row["id"], normalized["target_cell_id"])
            normalized["visible_fault_bbox_px"] = fault_visible_bbox_for_cell(
                cell_with_fault_bbox(cell, normalized),
                fault_type,
            )
        except (KeyError, TypeError):
            pass
    return normalized


def stored_fault_records():
    source_records = st.session_state.get("random_fault_records", [])
    records = []
    for record in source_records:
        normalized = normalized_fault_record(record)
        if normalized is not None:
            records.append(normalized)
    if len(records) != len(source_records) or any(
        left.get("fault_type") != right.get("fault_type")
        or left.get("fault_bbox_px") != right.get("fault_bbox_px")
        for left, right in zip(source_records, records)
    ):
        st.session_state["random_fault_records"] = records
    return records


def bounded_cell_mask(width, height, pad_ratio=0.05):
    mask = np.ones((height, width), dtype=np.float32)
    if width <= 2 or height <= 2:
        return mask
    pad_x = max(1, int(width * pad_ratio))
    pad_y = max(1, int(height * pad_ratio))
    mask[:pad_y, :] = 0
    mask[-pad_y:, :] = 0
    mask[:, :pad_x] = 0
    mask[:, -pad_x:] = 0
    return cv2.GaussianBlur(mask, (0, 0), max(0.45, min(width, height) * 0.04))


def bounded_pil_mask(size, pad_ratio=0.05):
    width, height = size
    if width <= 2 or height <= 2:
        return Image.new("L", size, 255)
    mask = Image.new("L", size, 0)
    pad_x = max(1, int(width * pad_ratio))
    pad_y = max(1, int(height * pad_ratio))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((pad_x, pad_y, max(pad_x, width - pad_x), max(pad_y, height - pad_y)), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(0.45, min(width, height) * 0.04)))


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
