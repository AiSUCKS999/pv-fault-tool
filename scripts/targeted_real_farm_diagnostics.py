"""Targeted visual QA for Real Farm Close View diagnostics and calibration."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "real_farm" / "targeted_diagnostics"


def make_scene(namespace):
    image = Image.new("RGB", (960, 600), (98, 132, 64))
    modules = []
    positions = [(95, 80), (365, 72), (640, 86), (88, 340), (360, 332), (635, 346)]
    for index, (x, y) in enumerate(positions, 1):
        module = namespace["real_farm_module_record"](
            f"M{index:03d}",
            [[x, y], [x + 190, y + 12], [x + 178, y + 160], [x - 10, y + 145]],
            image.size, "QA", 0.025, "targeted_diagnostic_qa",
        )
        modules.append(module)
    draw = ImageDraw.Draw(image)
    for module in modules:
        polygon = [tuple(point) for point in module["active_polygon_rgb"]]
        draw.polygon(polygon, fill=(37, 63, 96))
        x1 = min(point[0] for point in polygon)
        x2 = max(point[0] for point in polygon)
        y1 = min(point[1] for point in polygon)
        y2 = max(point[1] for point in polygon)
        for step in range(1, 7):
            x = x1 + (x2 - x1) * step / 7
            draw.line((x, y1, x, y2), fill=(120, 143, 167), width=1)
        for step in range(1, 10):
            y = y1 + (y2 - y1) * step / 10
            draw.line((x1, y, x2, y), fill=(120, 143, 167), width=1)
    return image, modules


def labelled(image, label, size=(470, 310)):
    tile = Image.new("RGB", size, (7, 12, 21))
    copy = image.copy()
    copy.thumbnail((size[0] - 12, size[1] - 36), Image.Resampling.LANCZOS)
    tile.paste(copy, ((size[0] - copy.width) // 2, 30))
    ImageDraw.Draw(tile).text((10, 9), label, fill="white")
    return tile


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        namespace = runpy.run_path(str(ROOT / "app.py"), run_name="targeted_real_farm_diagnostics")

    image, modules = make_scene(namespace)
    layers = namespace["real_farm_build_thermal_master"](image, modules)
    selected = modules[1]
    observation = {
        "observation_id": "RF-QA-001",
        "module_id": selected["module_id"],
        "observation_type": "SingleHotSpot",
        "local_uv": [0.54, 0.51],
        "severity": 7,
        "production_reference_ids": {},
    }
    healthy_scalar = layers["master_scalar"].copy()
    healthy_sensor = namespace["apply_sensor_model"](healthy_scalar, "real_farm", "far")
    healthy_far = namespace["real_farm_palette_image"](
        healthy_sensor, layers["display_range"], "far"
    )
    healthy_close = namespace["real_farm_close_context_image"](
        healthy_scalar, selected, modules, display_range=layers["display_range"]
    )
    x1, y1, x2, y2 = namespace["real_farm_close_context_bounds"](
        selected, modules, healthy_scalar.shape
    )
    expected_close_sensor = namespace["apply_sensor_model"](
        healthy_scalar, "real_farm", "close"
    )[y1:y2, x1:x2]
    expected_close = namespace["real_farm_palette_image"](
        expected_close_sensor, layers["display_range"], "close"
    )
    context_calibration_error = np.abs(
        np.asarray(healthy_close, dtype=np.int16)
        - np.asarray(expected_close, dtype=np.int16)
    )
    # The cyan selected-module outline is the only intended pixel difference.
    assert float(np.percentile(context_calibration_error, 95)) <= 1.0, (
        "Close View applied crop-relative camera processing"
    )
    # With a fixed scene range, identical scalar samples map to the same palette
    # neighborhood even when rendered as a crop. Only deterministic sub-pixel
    # dither may differ.
    calibration_crop = healthy_scalar[70:520, 70:890]
    mapped_full = np.asarray(namespace["real_farm_palette_image"](
        healthy_scalar, layers["display_range"], "far"
    ), dtype=np.int16)[70:520, 70:890]
    mapped_crop = np.asarray(namespace["real_farm_palette_image"](
        calibration_crop, layers["display_range"], "close"
    ), dtype=np.int16)
    calibration_error = np.abs(mapped_full - mapped_crop)
    assert float(np.percentile(calibration_error, 95)) <= 6.0, "Close View palette recalibrated its crop"
    healthy_far.save(OUT / "00_healthy_far.png")
    healthy_close.save(OUT / "01_healthy_close.png")

    tiles = [labelled(healthy_far, "Healthy Far View"), labelled(healthy_close, "Healthy Close View")]
    records = []
    outputs = {}
    for index, fault_type in enumerate(namespace["CLOSE_DIAGNOSTIC_FAULT_TYPES"], 2):
        candidate = json.loads(json.dumps(observation))
        candidate["diagnostic_fault_type"] = fault_type
        scalar = namespace["real_farm_diagnostic_master"](layers, candidate, fault_type, 7)
        assert np.any(scalar != healthy_scalar), f"{fault_type}: no scalar anomaly"
        assert np.array_equal(healthy_scalar, layers["master_scalar"]), f"{fault_type}: healthy master mutated"
        image_close = namespace["real_farm_close_context_image"](
            scalar, selected, modules, display_range=layers["display_range"]
        )
        path = OUT / f"{index:02d}_{fault_type}.png"
        image_close.save(path)
        outputs[fault_type] = scalar
        tiles.append(labelled(image_close, namespace["real_farm_diagnostic_label"](fault_type)))

        committed = json.loads(json.dumps(candidate))
        namespace["real_farm_faulted_master"](layers, [committed])
        metadata = committed["production_fault_debug_by_type"][fault_type]
        records.append({
            "fault_type": fault_type,
            "renderer_class": metadata.get("renderer_class"),
            "production_model": metadata.get("production_model"),
            "reference_id": metadata.get("reference_id"),
            "fallback_used": metadata.get("fallback_used"),
            "scalar_min_delta": round(float(np.min(scalar - healthy_scalar)), 4),
            "scalar_max_delta": round(float(np.max(scalar - healthy_scalar)), 4),
            "image": path.name,
        })

    # Exercise the actual selector order. Each preview and committed metadata
    # rebuilds from the healthy master and retains only the active observation
    # plus the currently selected diagnosis.
    switch_sequence = [
        "SurfaceObstruction", "SoilingDust", "CellCracking", "SingleDiode",
        "SingleByPassed", "StringOpenCircuit", "StringReversedPolarity", "SingleDiode",
    ]
    switching_record = json.loads(json.dumps(observation))
    diode_reference_id = None
    switching_results = []
    for fault_type in switch_sequence:
        switching_record["diagnostic_fault_type"] = fault_type
        switched = namespace["real_farm_diagnostic_master"](
            layers, switching_record, fault_type, 7
        )
        assert np.array_equal(switched, outputs[fault_type]), (
            f"{fault_type}: previous morphology remained under the active diagnosis"
        )
        namespace["real_farm_faulted_master"](layers, [switching_record])
        active_metadata = namespace["real_farm_fault_metadata"]([switching_record])
        active_types = {record["fault_type"] for record in active_metadata}
        assert active_types == {"SingleHotSpot", fault_type}, (
            f"{fault_type}: stale export metadata remained active"
        )
        metadata = switching_record["production_fault_debug_by_type"][fault_type]
        if fault_type == "SingleDiode":
            current_reference_id = metadata.get("reference_id")
            if diode_reference_id is None:
                diode_reference_id = current_reference_id
            else:
                assert current_reference_id == diode_reference_id, (
                    "Bypass Diode production reference changed after switching back"
                )
        switching_results.append({
            "fault_type": fault_type,
            "active_metadata_types": sorted(active_types),
            "reference_id": metadata.get("reference_id"),
        })
    hashes = {np.asarray(value, dtype=np.float32).tobytes() for value in outputs.values()}
    assert len(hashes) == len(outputs), "two diagnoses collapsed to identical scalar morphology"
    assert np.array_equal(healthy_scalar, layers["master_scalar"]), "healthy master mutated during switching"

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 470, rows * 310), (4, 8, 15))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 470, (index // columns) * 310))
    sheet.save(OUT / "diagnostic_contact_sheet.jpg", quality=94)
    (OUT / "diagnostic_metadata.json").write_text(json.dumps({
        "display_range": layers["display_range"],
        "palette_calibration_error_p95": float(np.percentile(calibration_error, 95)),
        "context_calibration_error_p95": float(np.percentile(context_calibration_error, 95)),
        "switching_sequence": switching_results,
        "diagnostics": records,
    }, indent=2), encoding="utf-8")
    print(f"real-farm targeted diagnostics: PASS ({len(outputs)} diagnoses)")
    print(OUT / "diagnostic_contact_sheet.jpg")


if __name__ == "__main__":
    main()
