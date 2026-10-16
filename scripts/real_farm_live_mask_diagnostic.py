"""Trace the exact Real Farm starter-ROI path on the uploaded vertical-row image."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "real_farm" / "live_mask_diagnostic"
IMAGE_NAME = "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"


def scalar_preview(values, support=None):
    array = np.asarray(values, dtype=np.float32)
    selected = array[support > 0] if support is not None and np.any(support > 0) else array.reshape(-1)
    low, high = np.percentile(selected, [1, 99])
    normalized = np.clip((array - low) * 255.0 / max(1e-6, high - low), 0, 255).astype(np.uint8)
    if support is not None:
        normalized[support == 0] = 0
    return Image.fromarray(normalized, "L")


def caption(image, text, color):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((20, 20, 1120, 88), fill=color)
    draw.text((35, 42), text, fill="white", font=ImageFont.load_default())
    return result


def main():
    image_path = ROOT / "artifacts" / "real_farm" / "uploads" / IMAGE_NAME
    if not image_path.exists():
        raise FileNotFoundError(f"Exact uploaded Real Farm image is missing: {image_path}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_live_mask_diagnostic")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    roi_width = max(20, width * 0.12)
    roi_height = max(20, height * 0.08)
    cx, cy = width * 0.5, height * 0.5
    roi = [[cx-roi_width/2, cy-roi_height/2], [cx+roi_width/2, cy-roi_height/2], [cx+roi_width/2, cy+roi_height/2], [cx-roi_width/2, cy+roi_height/2]]

    coarse = ns["real_farm_module_record"]("M001", roi, image.size, "LIVE", 0.0, "diagnostic_manual_final")
    starter = ns["real_farm_module_record"]("M001", roi, image.size, "LIVE", 0.0, "manual_add")
    result = ns["real_farm_refine_pv_support"](image, starter)
    if not result.get("success"):
        raise AssertionError(result.get("message"))
    ns["real_farm_accept_refinement"](starter, result)

    before_layers = ns["real_farm_build_thermal_master"](image, [coarse])
    coarse_mask = ns["real_farm_module_mask"](image.size, [coarse], active=True)
    legacy_rectangle_scalar = before_layers["background_scalar"].copy()
    legacy_rectangle_scalar[coarse_mask > 0] = 151.0
    legacy_range = ns["real_farm_display_range"](legacy_rectangle_scalar)
    before = ns["real_farm_palette_image"](legacy_rectangle_scalar, legacy_range)
    layers = ns["real_farm_build_thermal_master"](image, [starter])
    final_mask = layers["pv_module_union_mask"]
    roi_mask = ns["real_farm_roi_mask"](image.size, starter)
    after = ns["real_farm_palette_image"](layers["master_scalar"], layers["display_range"])

    roi_points = np.asarray(roi, dtype=int)
    x1, y1 = roi_points.min(axis=0); x2, y2 = roi_points.max(axis=0)
    grass_slice = final_mask[y1:y2, int(cx+roi_width*0.12):x2]
    grass_fraction = float(np.mean(grass_slice > 0)) if grass_slice.size else 1.0
    mask_in_roi_fraction = float(np.mean(final_mask[roi_mask > 0] > 0))
    orientation = starter.get("orientation_label")
    if orientation != "portrait":
        raise AssertionError(f"Expected refined vertical/portrait geometry, got {orientation}")
    if grass_fraction > 0.02:
        raise AssertionError(f"Grass leakage inside coarse ROI is too high: {grass_fraction:.3%}")
    if mask_in_roi_fraction >= 0.85:
        raise AssertionError("Final support still resembles the filled coarse ROI")

    observation_base = {"observation_id": "LIVE001", "module_id": "M001", "local_uv": [0.5, 0.76], "center_rgb_px": [0, 0], "severity": 8}
    hotspot_observation = [{**observation_base, "observation_type": "SingleHotSpot", "diagnostic_fault_type": None}]
    diode_observation = [{**observation_base, "observation_type": None, "diagnostic_fault_type": "SingleDiode"}]
    substring_observation = [{**observation_base, "observation_type": None, "diagnostic_fault_type": "SingleByPassed"}]
    fault_outputs = {}
    for name, observations in (("hotspot", hotspot_observation), ("bypass_diode", diode_observation), ("bypassed_substring", substring_observation)):
        scalar = ns["real_farm_faulted_master"](layers, observations)
        outside_delta = float(np.max(np.abs(scalar[final_mask == 0] - layers["master_scalar"][final_mask == 0])))
        if outside_delta > 1e-4:
            raise AssertionError(f"{name} leaked outside refined PV support")
        fault_outputs[name] = ns["real_farm_palette_image"](scalar, layers["display_range"])

    image.save(OUTPUT / "01_rgb_original.png")
    ns["real_farm_refined_overlay"](image, starter, {"refined_mask_contours_rgb": []}).save(OUTPUT / "02_user_roi_overlay.png")
    image.crop((x1, y1, x2, y2)).save(OUTPUT / "03_roi_crop_rgb.png")
    Image.fromarray(result["raw_mask"], "L").save(OUTPUT / "04_candidate_pv_segmentation_raw.png")
    Image.fromarray(result["cleaned_mask"], "L").save(OUTPUT / "05_candidate_pv_segmentation_cleaned.png")
    ns["real_farm_refined_overlay"](image, starter).save(OUTPUT / "06_refined_pv_polygon_overlay.png")
    Image.fromarray(final_mask, "L").save(OUTPUT / "07_FINAL_BINARY_PV_MASK.png")
    scalar_preview(layers["background_scalar"]).save(OUTPUT / "08_scalar_background.png")
    scalar_preview(layers["pv_scalar"], final_mask).save(OUTPUT / "09_scalar_pv_only.png")
    scalar_preview(layers["master_scalar"]).save(OUTPUT / "10_scalar_combined.png")
    after.save(OUTPUT / "11_final_thermal_clean.png")
    before_labeled = caption(before, "BEFORE: coarse starter ROI became thermal support", (105, 20, 28))
    after_labeled = caption(after, "AFTER: RGB-refined vertical PV support; grass remains background", (12, 86, 52))
    comparison = Image.new("RGB", (width*2, height), "black")
    comparison.paste(before_labeled, (0, 0)); comparison.paste(after_labeled, (width, 0))
    comparison.save(OUTPUT / "12_before_after.jpg", quality=92)
    fault_outputs["hotspot"].save(OUTPUT / "13_final_with_hotspot.png")
    fault_outputs["bypass_diode"].save(OUTPUT / "14_final_with_bypass_diode.png")
    fault_outputs["bypassed_substring"].save(OUTPUT / "15_final_with_bypassed_substring.png")
    report = {
        "result": "PASS",
        "source_image": str(image_path),
        "coarse_roi_polygon_rgb": roi,
        "old_live_path": "outer_polygon_rgb -> active_polygon_rgb -> real_farm_module_mask -> cv2.fillConvexPoly -> real_farm_module_scalar_layer",
        "responsible_variable_after_cdcbc77": "active_polygon_rgb",
        "new_live_path": "coarse_roi_polygon_rgb -> real_farm_refine_pv_support -> accepted refined_mask_contours_rgb -> real_farm_module_mask -> thermal scalar layer",
        "refined_orientation": orientation,
        "coarse_roi_pv_coverage_fraction": round(mask_in_roi_fraction, 6),
        "grass_leakage_fraction": round(grass_fraction, 6),
        "final_mask_pixels": int(np.count_nonzero(final_mask)),
        "coarse_roi_pixels": int(np.count_nonzero(roi_mask)),
        "faults_verified_inside_refined_support": ["SingleHotSpot", "SingleDiode", "SingleByPassed"],
    }
    (OUTPUT / "trace_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **report}, indent=2))


if __name__ == "__main__":
    main()
