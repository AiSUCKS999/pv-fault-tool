import contextlib
import io
import logging
import random
import runpy

import numpy as np


def quiet_streamlit_logs():
    for name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.caching",
        "streamlit.runtime.scriptrunner_utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_query(st, location_id, rows, arrays):
    st.query_params.clear()
    st.query_params["custom_layout"] = "1"
    st.query_params["row_count"] = str(rows)
    st.query_params["panels_per_row"] = str(arrays)
    st.query_params["location"] = location_id


def assert_scene(namespace, location_id, rows, arrays):
    st = namespace["st"]
    configure_query(st, location_id, rows, arrays)

    panels = namespace["panels"](location_id)
    assert len(panels) == rows, f"{location_id} {rows}x{arrays}: row count mismatch"

    first_row = panels[0]
    first_cell = namespace["cells_for_row"](location_id, first_row["id"])[0]
    rgb = namespace["module_field_image"](location_id, "rgb", first_row["id"], first_cell["id"], [], False).convert("RGB")
    regions = namespace["array_regions"](location_id, rgb)
    assert len(regions) == rows * arrays, f"{location_id} {rows}x{arrays}: array region count mismatch"

    for region in regions:
        array_crop, crop_origin, modules, _ = namespace["crop_array_from_module_union"](rgb, region, 0.14)
        ox1, oy1, ox2, oy2 = crop_origin
        assert array_crop.width > 0 and array_crop.height > 0, f"{location_id} {rows}x{arrays}: empty array crop"
        for point in namespace["array_module_union_points"](region):
            assert ox1 <= point[0] <= ox2 and oy1 <= point[1] <= oy2, (
                f"{location_id} {rows}x{arrays}: array crop does not contain full module union"
            )
        for module_number, module in modules.items():
            cells = namespace["module_cell_polygons"](module)
            assert len(cells) == 40, f"{location_id} {rows}x{arrays}: module {module_number} does not have 40 cells"
            outer_polygon = namespace["module_outer_polygon"](module)
            active_polygon = namespace["module_active_polygon"](module)
            for point in active_polygon:
                assert namespace["point_in_polygon"](
                    point,
                    outer_polygon,
                ), f"{location_id} {rows}x{arrays}: active PV polygon outside outer module"
            for cell_number, geometry in cells.items():
                assert namespace["point_in_polygon"](
                    geometry["centroid"],
                    active_polygon,
                ), f"{location_id} {rows}x{arrays}: cell {cell_number} centroid outside active module"
                assert namespace["point_in_polygon"](
                    geometry["safe_centroid"],
                    active_polygon,
                ), f"{location_id} {rows}x{arrays}: safe fault center outside active module"

    location = {"id": location_id, "name": location_id}
    fault_record = namespace["fault_record"](location, "thermal", first_row, first_cell, "SingleHotSpot", 8)
    thermal = namespace["module_field_image"](location_id, "thermal", first_row["id"], first_cell["id"], [fault_record], False).convert("RGB")
    assert thermal.size == rgb.size, f"{location_id} {rows}x{arrays}: RGB/Thermal size mismatch"

    crop, *_ = namespace["crop_natural_module_image"](thermal, rgb.size, regions[0], 1, 0.62)
    assert crop.width > 0 and crop.height > 0, f"{location_id} {rows}x{arrays}: empty module crop"

    if rows == 6 and arrays == 6:
        assert_random_fault_origins(namespace, location_id, location, regions)
        assert_fault_matrix(namespace, location_id, location, regions)


def assert_random_fault_origins(namespace, location_id, location, regions):
    rng = random.Random(93817)
    array_geometries = {
        (int(region["row"]), int(region["col"])): {
            "box": region["box"],
            "polygon": region.get("polygon") or namespace["box_polygon"](region["box"]),
        }
        for region in regions
    }
    rows = namespace["panels"](location_id)
    candidates = [
        (row, cell)
        for row in rows
        for cell in namespace["cells_for_row"](location_id, row["id"])
    ]
    sample = rng.sample(candidates, min(100, len(candidates)))
    for index, (row, cell) in enumerate(sample):
        record = namespace["fault_record"](location, "thermal", row, cell, "SingleHotSpot", 8)
        rendered = namespace["rendered_fault_cell_from_record"](
            location_id,
            record,
            row["id"],
            cell["id"],
            array_geometries,
        )
        assert rendered is not None, f"{location_id}: missing rendered fault cell for {cell['id']}"
        origin = (rendered["fault_origin_px"]["x"], rendered["fault_origin_px"]["y"])
        assert namespace["point_in_polygon"](
            origin,
            rendered["source_polygon"],
        ), f"{location_id}: fault origin outside selected cell polygon for {cell['id']} sample {index}"

        row_num = namespace["row_number_from_id"](location_id, row["id"])
        array_col = int(cell.get("panel_number") or 1)
        module_number = int(cell.get("module_number_in_panel") or 1)
        region = next(item for item in regions if int(item["row"]) == row_num and int(item["col"]) == array_col)
        module = (region.get("modules") or namespace["array_module_polygons"](region))[module_number]
        active_polygon = namespace["module_active_polygon"](module)
        assert namespace["point_in_polygon"](
            origin,
            active_polygon,
        ), f"{location_id}: fault origin outside active PV area for {cell['id']} sample {index}"


def assert_fault_matrix(namespace, location_id, location, regions):
    fault_types = list(namespace.get("CLOSE_DIAGNOSTIC_FAULT_TYPES") or namespace["selectable_fault_types_for_location"](location_id))
    far_types = list(namespace.get("FAR_SCOUT_FAULT_TYPES") or [])
    assert "PartialShading" in far_types, "Partial Shading missing from far-view thermal observations"
    assert "PartialShading" not in fault_types, "Partial Shading must not appear in close diagnostic faults"
    assert "SurfaceObstruction" in fault_types, "Surface Obstruction missing from close diagnostic faults"
    assert any("Diode" in fault_type for fault_type in fault_types), "Diode faults missing from diagnostic faults"
    assert any("String" in fault_type for fault_type in fault_types), "String faults missing from diagnostic faults"

    array_geometries = {
        (int(region["row"]), int(region["col"])): {
            "box": region["box"],
            "polygon": region.get("polygon") or namespace["box_polygon"](region["box"]),
        }
        for region in regions
    }
    probe_regions = [regions[0], regions[len(regions) // 2], regions[-1]]
    temp_shape = (720, 1080)
    for region in probe_regions:
        for module_number in (1, 2, 3, 4):
            for cell_number in (1, 5, 18, 36, 40):
                row_id, cell = namespace["default_cell_for_array"](
                    location_id,
                    int(region["row"]),
                    int(region["col"]),
                    module_number,
                    cell_number,
                )
                for fault_type in fault_types:
                    record = namespace["fault_record"](location, "thermal", namespace["row_by_id"](location_id, row_id), cell, fault_type, 8)
                    rendered = namespace["rendered_fault_cell_from_record"](
                        location_id,
                        record,
                        row_id,
                        cell["id"],
                        array_geometries,
                    )
                    assert rendered is not None, f"{location_id}: no rendered fault cell for {fault_type} {cell['id']}"
                    temp = np.full(temp_shape, 156.0, dtype=np.float32)
                    try:
                        namespace["apply_thermal_fault"](temp, rendered, fault_type, 8)
                    except Exception as exc:
                        raise AssertionError(
                            f"{location_id}: fault matrix failed fault_type={fault_type} "
                            f"row={row_id} array={region['key']} module={module_number} cell={cell['id']}: {exc}"
                        ) from exc


def main():
    quiet_streamlit_logs()
    with contextlib.redirect_stderr(io.StringIO()):
        namespace = runpy.run_path("app.py", run_name="pv_regression_smoke")

    for location_id in ("grass_open", "desert_track", "floating_water", "rooftop_warehouse"):
        assert_scene(namespace, location_id, 1, 1)
        assert_scene(namespace, location_id, 6, 6)

    print("regression_smoke: ok")


if __name__ == "__main__":
    main()
