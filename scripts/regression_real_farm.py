"""Regression checks for the real-image thermal-mask and fault workflow."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        namespace = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_regression")

    image = Image.new("RGB", (640, 420), (96, 132, 61))
    modules = []
    for index, (x, y) in enumerate(((70, 60), (250, 60), (430, 60), (70, 235), (250, 235), (430, 235)), 1):
        modules.append(namespace["real_farm_module_record"](
            f"M{index:03d}",
            [[x, y], [x + 120, y + 10], [x + 115, y + 110], [x - 5, y + 100]],
            image.size,
            "A1",
            0.035,
            "regression",
        ))
    draw = ImageDraw.Draw(image)
    for module in modules:
        polygon = [tuple(point) for point in module["active_polygon_rgb"]]
        draw.polygon(polygon, fill=(38, 66, 102))
        x1 = min(point[0] for point in polygon); x2 = max(point[0] for point in polygon)
        y1 = min(point[1] for point in polygon); y2 = max(point[1] for point in polygon)
        for step in range(1, 5):
            x = x1 + (x2 - x1) * step / 5
            draw.line((x, y1, x, y2), fill=(118, 142, 168), width=1)
        for step in range(1, 8):
            y = y1 + (y2 - y1) * step / 8
            draw.line((x1, y, x2, y), fill=(118, 142, 168), width=1)

    state = namespace["real_farm_default_state"](None, image, "regression.png", "regression")
    state["modules"] = modules
    state["thermal_mask_ready"] = True
    state["thermal_source"] = namespace["REAL_FARM_SYNTHETIC_THERMAL"]
    state["thermal_geometry_signature"] = namespace["real_farm_geometry_signature"](modules)
    observations = namespace["real_farm_generate_observations"](modules, 10, 42)
    observations[0]["diagnostic_fault_type"] = "SoilingDust"
    observations[1]["diagnostic_fault_type"] = "CellCracking"
    observations[2]["diagnostic_fault_type"] = "SurfaceObstruction"
    state["observations"] = observations

    mask = namespace["real_farm_module_mask"](image.size, modules)
    layers = namespace["real_farm_build_thermal_master"](image, modules)
    thermal = namespace["real_farm_thermal_image"](image, modules, observations, master_layers=layers)
    dataset = namespace["real_farm_dataset_zip"](state, image, thermal)
    with zipfile.ZipFile(io.BytesIO(dataset)) as package:
        names = set(package.namelist())
        report = json.loads(package.read("hotspot_fault_report.json"))

    assert mask.max() == 255 and int((mask > 0).sum()) > 1000
    assert thermal.size == image.size
    assert len(layers["accepted_modules"]) == len(modules)
    assert not layers["rejected_modules"]
    assert np.count_nonzero(layers["pv_scalar"][mask == 0]) == 0
    low_texture = Image.new("RGB", (320, 220), (110, 140, 110))
    low_texture_module = namespace["real_farm_module_record"](
        "MLOW", [[42, 44], [276, 62], [251, 184], [29, 169]], low_texture.size, "QA", 0.02, "low_texture_regression"
    )
    low_accepted, low_rejected = namespace["real_farm_thermal_modules"](low_texture, [low_texture_module])
    assert len(low_accepted) == 1 and not low_rejected
    assert low_accepted[0]["mechanically_valid"] is True
    assert low_accepted[0]["visually_supported"] is False
    assert any(item["code"] == "weak_pv_texture_confidence" for item in low_accepted[0]["warnings"])
    invalid = namespace["real_farm_module_record"](
        "MBAD", [[20, 20], [20, 20], [21, 21], [20, 21]], low_texture.size, "QA", 0.0, "invalid_regression"
    )
    invalid_accepted, invalid_rejected = namespace["real_farm_thermal_modules"](low_texture, [invalid])
    assert not invalid_accepted and len(invalid_rejected) == 1
    assert invalid_rejected[0]["mechanically_valid"] is False
    assert len(observations) == 10
    assert len({item["observation_id"] for item in observations}) == 10
    assert len(report) == 10
    assert report[1]["diagnostic_fault_type"] == "CellCracking"
    assert {
        "project.json",
        "modules.csv",
        "hotspot_fault_report.csv",
        "hotspot_fault_report.json",
        "thermal_mask.png",
        "thermal_scene.png",
        "source_rgb.png",
    } <= names
    print("real-farm regression: PASS")


if __name__ == "__main__":
    main()
