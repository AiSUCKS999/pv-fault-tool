"""Generate the Real Farm thermal-proxy before/after and QA artifact bundle."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def reference_array_polygons():
    # Actual PV-array boundaries in the saved 4000x2250 screenshot source.
    return [
        [[10, 0], [294, 0], [300, 781], [18, 778]],
        [[616, 0], [880, 0], [884, 790], [625, 786]],
        [[1215, 0], [1458, 0], [1459, 793], [1221, 789]],
        [[1799, 0], [2040, 0], [2039, 799], [1805, 794]],
        [[2378, 0], [2635, 0], [2631, 807], [2384, 801]],
        [[2933, 0], [3159, 0], [3155, 813], [2937, 807]],
        [[3502, 0], [3744, 0], [3738, 815], [3507, 809]],
        [[18, 866], [302, 869], [309, 2249], [30, 2249]],
        [[619, 870], [886, 873], [899, 2249], [632, 2249]],
        [[1217, 872], [1461, 874], [1470, 2249], [1228, 2249]],
        [[1798, 874], [2039, 876], [2045, 2249], [1808, 2249]],
        [[2370, 876], [2629, 879], [2638, 2249], [2382, 2249]],
        [[2925, 878], [3154, 881], [3165, 2249], [2937, 2249]],
        [[3488, 880], [3738, 884], [3749, 2249], [3501, 2249]],
    ]


def scalar_preview(array, display_range):
    low, high = display_range
    normalized = np.clip((np.asarray(array, dtype=np.float32) - low) / max(1e-6, high - low), 0, 1)
    return Image.fromarray(np.uint8(normalized * 255), "L")


def comparison(before, after):
    width = 1600
    half = width // 2
    left = before.copy(); left.thumbnail((half, 700), Image.Resampling.LANCZOS)
    right = after.copy(); right.thumbnail((half, 700), Image.Resampling.LANCZOS)
    height = max(left.height, right.height) + 54
    canvas = Image.new("RGB", (width, height), (6, 11, 20))
    canvas.paste(left, ((half - left.width) // 2, 48))
    canvas.paste(right, (half + (half - right.width) // 2, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 17), "BEFORE — starter/ROI rectangle treated as a hot module", fill=(255, 170, 80), font=font)
    draw.text((half + 18, 17), "AFTER — full-scene scalar master + actual PV polygon union", fill=(150, 240, 190), font=font)
    return canvas


def load_namespace():
    with contextlib.redirect_stderr(io.StringIO()):
        return runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_thermal_diagnostic")


def diagnostic_modules(namespace, image):
    if image.size != (4000, 2250):
        raise ValueError("The screenshot reproduction expects the saved 4000x2250 Real Farm upload.")
    modules = []
    for index, polygon in enumerate(reference_array_polygons(), 1):
        modules.append(namespace["real_farm_module_record"](
            f"M{index:03d}", polygon, image.size, f"ARRAY{index:02d}", 0.008, "screenshot_failure_qa",
        ))
    return modules


def diagnostic_fault_module(namespace, image):
    quad = np.asarray(reference_array_polygons()[8], dtype=np.float32)

    def point(u, v):
        top = quad[0] * (1.0 - u) + quad[1] * u
        bottom = quad[3] * (1.0 - u) + quad[2] * u
        return top * (1.0 - v) + bottom * v

    row, rows, column, columns = 3, 14, 0, 2
    u1, u2 = column / columns, (column + 1) / columns
    v1, v2 = row / rows, (row + 1) / rows
    polygon = [point(u1, v1), point(u2, v1), point(u2, v2), point(u1, v2)]
    return namespace["real_farm_module_record"]("FAULT_MODULE", polygon, image.size, "ARRAY09", 0.018, "fault_engine_qa")


def run_geometry_matrix(namespace):
    results = []
    for geometry_path in sorted((ROOT / "artifacts" / "real_farm").glob("*_geometry.json")):
        data = json.loads(geometry_path.read_text(encoding="utf-8"))
        source = Path(data["source_rgb_path"])
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        layers = namespace["real_farm_build_thermal_master"](image, data.get("modules", []))
        mask = layers["pv_module_union_mask"] > 0
        leakage = int(np.count_nonzero(layers["pv_scalar"][~mask]))
        results.append({
            "case": geometry_path.stem,
            "modules": len(data.get("modules", [])),
            "accepted": len(layers["accepted_modules"]),
            "rejected": len(layers["rejected_modules"]),
            "mask_pixels": int(np.count_nonzero(mask)),
            "pv_scalar_pixels_outside_mask": leakage,
        })
        assert leakage == 0, geometry_path.name
        assert len(layers["accepted_modules"]) == len(data.get("modules", [])), geometry_path.name
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "real_farm" / "thermal_proxy_fix")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    namespace = load_namespace()
    image = Image.open(args.source).convert("RGB")

    # Reproduce the old default starter rectangle visible in the supplied screenshot.
    default_width = image.width * 0.12
    default_height = image.height * 0.08
    center_x, center_y = image.width * 0.5, image.height * 0.5
    starter = [[center_x-default_width/2, center_y-default_height/2], [center_x+default_width/2, center_y-default_height/2], [center_x+default_width/2, center_y+default_height/2], [center_x-default_width/2, center_y+default_height/2]]
    old_module = namespace["real_farm_module_record"]("M001", starter, image.size, "", 0.035, "legacy_failure")
    before = namespace["real_farm_legacy_thermal_image"](image, [old_module], [])

    modules = diagnostic_modules(namespace, image)
    layers = namespace["real_farm_build_thermal_master"](image, modules)
    base_accepted_count = len(layers["accepted_modules"])
    after = namespace["real_farm_thermal_image"](image, modules, [], master_layers=layers)
    fault_module = diagnostic_fault_module(namespace, image)
    layers["accepted_modules"].append(fault_module)
    observation = namespace["real_farm_generate_observations"]([fault_module], 1, 42)[0]
    with_hotspot = namespace["real_farm_thermal_image"](image, modules, [observation], master_layers=layers)
    diode = dict(observation)
    diode["observation_type"] = None
    diode["diagnostic_fault_type"] = "SingleDiode"
    with_diode = namespace["real_farm_thermal_image"](image, modules, [diode], master_layers=layers)

    before.save(args.output / "before_current_mask.png")
    Image.fromarray(layers["pv_module_union_mask"], "L").save(args.output / "pv_module_union_mask.png")
    scalar_preview(layers["background_scalar"], layers["display_range"]).save(args.output / "scalar_background.png")
    pv_display = np.where(layers["pv_module_union_mask"] > 0, layers["pv_scalar"], layers["display_range"][0])
    scalar_preview(pv_display, layers["display_range"]).save(args.output / "scalar_pv_layer.png")
    scalar_preview(layers["master_scalar"], layers["display_range"]).save(args.output / "scalar_combined.png")
    after.save(args.output / "after_fixed_thermal_proxy.png")
    comparison(before, after).save(args.output / "before_after_comparison.jpg", quality=94)
    with_hotspot.save(args.output / "after_fixed_with_hotspot.png")
    with_diode.save(args.output / "after_fixed_with_bypass_diode.png")

    mask = layers["pv_module_union_mask"] > 0
    hotspot_scalar = namespace["real_farm_faulted_master"](layers, [observation])
    diode_scalar = namespace["real_farm_faulted_master"](layers, [diode])
    substring = dict(observation)
    substring["observation_type"] = None
    substring["diagnostic_fault_type"] = "SingleByPassed"
    substring_scalar = namespace["real_farm_faulted_master"](layers, [substring])
    hotspot_delta = np.abs(hotspot_scalar - layers["master_scalar"])
    diode_delta = np.abs(diode_scalar - layers["master_scalar"])
    substring_delta = np.abs(substring_scalar - layers["master_scalar"])
    matrix = run_geometry_matrix(namespace)
    report = {
        "thermal_source": namespace["REAL_FARM_SYNTHETIC_THERMAL"],
        "thermal_is_real": False,
        "thermal_is_radiometric": False,
        "source": str(args.source.resolve()),
        "accepted_modules": base_accepted_count,
        "rejected_modules": len(layers["rejected_modules"]),
        "module_union_pixels": int(np.count_nonzero(mask)),
        "module_union_coverage_fraction": round(float(np.mean(mask)), 8),
        "pv_scalar_leakage_pixels": int(np.count_nonzero(layers["pv_scalar"][~mask])),
        "hotspot_delta_outside_pv_pixels": int(np.count_nonzero(hotspot_delta[~mask] > 1e-4)),
        "diode_delta_outside_pv_pixels": int(np.count_nonzero(diode_delta[~mask] > 1e-4)),
        "substring_delta_outside_pv_pixels": int(np.count_nonzero(substring_delta[~mask] > 1e-4)),
        "production_diode_engine_enabled": bool(namespace["production_submodule_enabled"]("SingleDiode")),
        "production_substring_engine_enabled": bool(namespace["production_submodule_enabled"]("SingleByPassed")),
        "display_range": layers["display_range"],
        "geometry_matrix": matrix,
    }
    assert report["pv_scalar_leakage_pixels"] == 0
    assert report["hotspot_delta_outside_pv_pixels"] == 0
    assert report["diode_delta_outside_pv_pixels"] == 0
    assert report["substring_delta_outside_pv_pixels"] == 0
    assert report["production_diode_engine_enabled"]
    assert report["production_substring_engine_enabled"]
    assert namespace["real_farm_module_visual_score"](image, old_module) < 0.43
    (args.output / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
