"""Visual QA for the common Generated/Real Farm scalar thermal pipeline."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"
ARCH = ROOT / "artifacts" / "real_farm" / "thermal_v2_architecture"
COMPARE = ROOT / "artifacts" / "real_farm" / "generated_vs_real_v2"
ROI = [[1760, 1035], [2240, 1035], [2240, 1215], [1760, 1215]]
FAULTS = ["SingleHotSpot", "SingleDiode", "SingleByPassed"]


def caption(image, text):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 42), fill=(6, 12, 22))
    draw.text((12, 14), text, fill="white", font=ImageFont.load_default())
    return result


def fit(image, size=(720, 560)):
    image = image.convert("RGB").copy()
    scale = min(size[0] / max(1, image.width), size[1] / max(1, image.height))
    image = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (3, 7, 13))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def compare_pair(left, right, left_text, right_text):
    left = caption(fit(left), left_text)
    right = caption(fit(right), right_text)
    canvas = Image.new("RGB", (left.width + right.width, left.height), "black")
    canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
    return canvas


def scalar_image(values, signed=False):
    values = np.asarray(values, dtype=np.float32)
    if signed:
        values = np.maximum(values, 0)
        low, high = 0.0, max(1.0, float(np.percentile(values, 99.8)))
    else:
        low, high = [float(item) for item in np.percentile(values, (0.6, 99.4))]
    normalized = np.clip((values - low) / max(1e-5, high - low), 0, 1)
    return Image.fromarray(np.rint(normalized * 255).astype(np.uint8), "L")


def instance_mask_preview(image_size, modules):
    width, height = image_size
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for index, module in enumerate(modules):
        polygon = np.rint(np.asarray(module["active_polygon_rgb"], dtype=np.float32)).astype(np.int32)
        color = np.random.default_rng(701 + index * 7919).integers(58, 246, size=3, dtype=np.uint8)
        cv2.fillConvexPoly(result, polygon, tuple(int(v) for v in color), lineType=cv2.LINE_AA)
    return Image.fromarray(result, "RGB")


def rectified_sheet(ns, image, modules, limit=16):
    tiles = []
    for module in modules[:limit]:
        crop = ns["real_farm_rectified_crop"](image, module).convert("RGB")
        crop.thumbnail((220, 170), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (240, 205), (5, 10, 18))
        tile.paste(crop, ((240 - crop.width) // 2, 28))
        ImageDraw.Draw(tile).text((8, 8), module["module_id"], fill="white", font=ImageFont.load_default())
        tiles.append(tile)
    columns = 4
    rows = max(1, int(np.ceil(len(tiles) / columns)))
    sheet = Image.new("RGB", (columns * 240, rows * 205), (3, 7, 13))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 240, (index // columns) * 205))
    return sheet


def crop_polygon(image, polygon, pad=0.08):
    points = np.asarray(polygon, dtype=np.float32)
    x1, y1 = points.min(axis=0); x2, y2 = points.max(axis=0)
    margin = max(x2 - x1, y2 - y1) * pad
    box = (max(0, int(x1 - margin)), max(0, int(y1 - margin)), min(image.width, int(x2 + margin)), min(image.height, int(y2 + margin)))
    return image.crop(box)


def roi_overlay(image):
    result = image.copy()
    ImageDraw.Draw(result).line([tuple(point) for point in ROI] + [tuple(ROI[0])], fill=(0, 188, 255), width=8)
    return result


def generated_views(ns, fault_type=None):
    st = ns["st"]
    st.query_params.clear()
    st.query_params["custom_layout"] = "1"
    st.query_params["row_count"] = "1"
    st.query_params["panels_per_row"] = "1"
    st.query_params["location"] = "grass_open"
    row = ns["panels"]("grass_open")[0]
    cell = ns["cells_for_row"]("grass_open", row["id"])[0]
    records = []
    if fault_type:
        records = [ns["fault_record"]({"id": "grass_open", "name": "Grass"}, "thermal", row, cell, fault_type, 8)]
    scene = ns["module_field_image"]("grass_open", "thermal", row["id"], cell["id"], records, False, "far").convert("RGB")
    region = ns["array_regions"]("grass_open", scene)[0]
    crop, *_ = ns["crop_natural_module_image"](scene, scene.size, region, 1, 0.10)
    return scene, crop


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    ARCH.mkdir(parents=True, exist_ok=True); COMPARE.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_thermal_v2_qa")
    image = Image.open(SOURCE).convert("RGB")
    parent = ns["real_farm_module_record"]("M001", ROI, image.size, "LIVE", 0.0, "manual_add")
    refinement = ns["real_farm_refine_pv_support"](image, parent)
    if not refinement.get("success"):
        raise AssertionError(refinement.get("message"))
    ns["real_farm_accept_refinement"](parent, refinement)
    layers = ns["real_farm_build_thermal_master"](image, [parent])
    modules = layers["accepted_modules"]
    if len(modules) < 2:
        raise AssertionError("Module-instance refinement did not recover multiple modules")
    center = np.asarray([2000.0, 1125.0], dtype=np.float32)
    target = min(modules, key=lambda module: float(np.linalg.norm(np.asarray(module["active_polygon_rgb"], dtype=np.float32).mean(axis=0) - center)))
    observation = {"observation_id": "RFV2QA", "module_id": target["module_id"], "local_uv": [0.53, 0.50], "severity": 8}

    healthy_scalar = layers["master_scalar"]
    healthy_sensor = ns["apply_sensor_model"](healthy_scalar, "real_farm", "far")
    healthy_final = ns["thermal_display_rgb_from_scalar"](healthy_sensor, "real_farm", "far")
    hotspot_scalar = ns["real_farm_apply_production_fault"](healthy_scalar, target, observation, "SingleHotSpot")
    hotspot_delta = hotspot_scalar - healthy_scalar
    hotspot_final = ns["thermal_display_rgb_from_scalar"](ns["apply_sensor_model"](hotspot_scalar, "real_farm", "far"), "real_farm", "far")

    image.save(ARCH / "01_rgb.png")
    roi_overlay(image).save(ARCH / "02_coarse_roi.png")
    Image.fromarray(layers["pv_module_union_mask"], "L").save(ARCH / "03_pv_union_mask.png")
    ns["real_farm_instance_overlay"](image, parent).save(ARCH / "04_module_instance_overlay.png")
    instance_mask_preview(image.size, modules).save(ARCH / "05_module_instance_masks.png")
    rectified_sheet(ns, image, modules).save(ARCH / "06_rectified_modules.png")
    scalar_image(healthy_scalar).save(ARCH / "07_healthy_thermal_scalar.png")
    healthy_final.save(ARCH / "08_healthy_final_thermal.png")
    scalar_image(hotspot_delta, signed=True).save(ARCH / "09_hotspot_scalar_delta.png")
    hotspot_final.save(ARCH / "10_final_hotspot_thermal.png")
    material_tiles = [caption(Image.fromarray(mask, "L").convert("RGB"), name) for name, mask in layers["material_masks"].items()]
    material_rows = int(np.ceil(len(material_tiles) / 2))
    material_sheet = Image.new("RGB", (image.width * 2, image.height * material_rows), "black")
    for index, tile in enumerate(material_tiles): material_sheet.paste(tile, ((index % 2) * image.width, (index // 2) * image.height))
    material_sheet.thumbnail((1800, 1500), Image.Resampling.LANCZOS)
    material_sheet.save(ARCH / "material_class_masks.jpg", quality=91)

    old = ROOT / "artifacts" / "real_farm" / "live_mask_diagnostic" / "11_final_thermal_clean.png"
    if old.exists(): shutil.copyfile(old, ARCH / "old_purple_real_farm_output.png")
    generated_healthy_scene, generated_healthy_crop = generated_views(ns)
    real_healthy_crop = crop_polygon(healthy_final, target["active_polygon_rgb"])
    compare_pair(generated_healthy_crop, real_healthy_crop, "GENERATED FARM — HEALTHY", "REAL FARM V2 — HEALTHY").save(COMPARE / "generated_real_healthy.jpg", quality=94)
    comparisons = [compare_pair(generated_healthy_crop, real_healthy_crop, "GENERATED HEALTHY", "REAL V2 HEALTHY")]
    fault_files = {}
    for fault_type in FAULTS:
        _, generated_crop = generated_views(ns, fault_type)
        real_scalar = ns["real_farm_apply_production_fault"](healthy_scalar, target, observation, fault_type)
        outside = layers["pv_module_union_mask"] == 0
        leakage = int(np.count_nonzero(np.abs(real_scalar - healthy_scalar)[outside] > 1e-4))
        if leakage:
            raise AssertionError(f"{fault_type} leaked to {leakage} non-PV pixels")
        real_final = ns["thermal_display_rgb_from_scalar"](ns["apply_sensor_model"](real_scalar, "real_farm", "far"), "real_farm", "far")
        real_crop = crop_polygon(real_final, target["active_polygon_rgb"])
        pair = compare_pair(generated_crop, real_crop, f"GENERATED — {fault_type}", f"REAL V2 — {fault_type}")
        filename = {"SingleHotSpot": "generated_real_hotspot.jpg", "SingleDiode": "generated_real_bypass_diode.jpg", "SingleByPassed": "generated_real_bypassed_substring.jpg"}[fault_type]
        pair.save(COMPARE / filename, quality=94)
        comparisons.append(pair)
        fault_files[fault_type] = {"outside_pv_leakage_pixels": leakage}
    sheet = Image.new("RGB", (1440, 560 * len(comparisons)), "black")
    for index, item in enumerate(comparisons): sheet.paste(item, (0, index * 560))
    sheet.save(COMPARE / "generated_vs_real_fault_comparison.jpg", quality=92)
    healthy_four = [
        caption(fit(image), "REAL RGB"),
        caption(fit(Image.open(old).convert("RGB") if old.exists() else image), "OLD REAL OUTPUT"),
        caption(fit(healthy_final), "NEW REAL V2 THERMAL"),
        caption(fit(generated_healthy_scene), "GENERATED THERMAL REFERENCE"),
    ]
    overview = Image.new("RGB", (1440, 1120), "black")
    for index, item in enumerate(healthy_four): overview.paste(item, ((index % 2) * 720, (index // 2) * 560))
    overview.save(COMPARE / "healthy_architecture_before_after.jpg", quality=92)

    report = {
        "result": "PASS", "source": str(SOURCE), "module_instance_count": len(modules),
        "selected_module": target["module_id"], "thermal_master_dtype": str(healthy_scalar.dtype),
        "thermal_source": layers["mode"], "shared_sensor_function": "apply_sensor_model",
        "shared_palette_function": "thermal_display_rgb_from_scalar", "faults": fault_files,
    }
    (ARCH / "thermal_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARCH / "thermal_architecture_audit.md").write_text(
        "# Real Farm Thermal V2 audit\n\n"
        "## Generated Farm trace\n\n"
        "1. `solar_render_thermal_base_field` creates the full-scene float32 scalar master.\n"
        "2. `thermal_noise` and `pv_environment_multiplier` create broad environment and illumination variation.\n"
        "3. `smooth_thermal_array_field` supplies the PV material response inside generated geometry.\n"
        "4. `pv_cell_temperature_field` adds module/cell-scale variation.\n"
        "5. The base builder combines broad gradients, detail, edge response, and mottling before faults.\n"
        "6. `apply_thermal_fault` selects the approved Thermal Hotspot residual and adds it to the scalar field.\n"
        "7. The same `apply_thermal_fault` entry point selects the approved Bypass Diode production signal.\n"
        "8. The same entry point selects the approved Bypassed Substring production signal.\n"
        "9. `apply_sensor_model` performs sensor blur, resampling, noise, banding, and vignette once.\n"
        "10. `thermal_display_rgb_from_scalar` performs stable scene-level normalization and the final palette map.\n\n"
        "## Real Farm V2 reuse\n\n"
        "Real Farm V2 creates material masks and a float32 scalar scene, then reuses `thermal_noise`, "
        "`pv_environment_multiplier`, `smooth_thermal_array_field`, `pv_cell_temperature_field`, "
        "`apply_thermal_fault`, `apply_sensor_model`, and `thermal_display_rgb_from_scalar`. RGB supplies material "
        "classification, edges, and low-amplitude structure only; RGB hue is never used as thermal color.\n\n"
        "## Previous root cause\n\n"
        "The former `real_farm_background_scalar` was dominated by source luminance (`low`, `mid`, and `detail` "
        "terms). `real_farm_build_thermal_master` added its own private sensor noise, and "
        "`real_farm_palette_image` normalized and called `temperature_to_rgb` on a separate path. That preserved the "
        "photograph's brightness structure so strongly that the result looked like a purple-tinted RGB image.\n\n"
        f"Recovered module instances: **{len(modules)}**. Production fault banks were not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps({"architecture_output": str(ARCH), "comparison_output": str(COMPARE), **report}, indent=2))


if __name__ == "__main__":
    main()
