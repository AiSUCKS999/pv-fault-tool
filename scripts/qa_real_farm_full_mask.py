"""End-to-end QA for the recovered-module -> full-farm mask data path."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"
OUTPUT = ROOT / "artifacts" / "real_farm" / "full_farm_mask_fix"
# This is the exact starter ROI used by the former UI. It recovers 52 instances.
SEARCH_ROI = [[1760, 1035], [2240, 1035], [2240, 1215], [1760, 1215]]


def thermal_image(ns, scalar, distance="far"):
    sensor = ns["apply_sensor_model"](np.asarray(scalar, dtype=np.float32), "real_farm", distance)
    return ns["thermal_display_rgb_from_scalar"](sensor, "real_farm", distance)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="qa_real_farm_full_mask")
    image = Image.open(SOURCE).convert("RGB")
    parent = ns["real_farm_module_record"](
        "M001", SEARCH_ROI, image.size, "LIVE_52", 0.0, "manual_add"
    )
    refinement = ns["real_farm_refine_pv_support"](image, parent)
    if not refinement.get("success"):
        raise AssertionError(refinement.get("message"))
    ns["real_farm_accept_refinement"](parent, refinement)
    parents = [parent]
    recovered = ns["real_farm_recovered_modules"](parents, image_size=image.size)
    summary = ns["real_farm_geometry_qa_summary"](image, parents)
    final_mask = ns["real_farm_module_mask"](image.size, recovered, active=True)

    per_module_pixels = []
    union_from_instances = np.zeros_like(final_mask)
    for module in recovered:
        instance_mask = ns["real_farm_module_mask"](image.size, [module], active=True)
        pixels = int(np.count_nonzero(instance_mask))
        if pixels <= 0:
            raise AssertionError(f"{module.get('module_id')} contributes no PV support")
        per_module_pixels.append(pixels)
        union_from_instances = np.maximum(union_from_instances, instance_mask)
    if not np.array_equal(final_mask, union_from_instances):
        raise AssertionError("Final PV mask differs from the OR of accepted instance masks")

    parent_support = (np.asarray(refinement["final_mask"]) > 0).astype(np.uint8) * 255
    leakage = int(np.count_nonzero((final_mask > 0) & (parent_support == 0)))
    if leakage:
        raise AssertionError(f"Final mask leaks {leakage} pixels outside RGB-refined PV support")

    layers = ns["real_farm_build_thermal_master"](image, parents)
    if not np.array_equal(final_mask, layers["pv_module_union_mask"]):
        raise AssertionError("Thermal master did not consume the canonical final PV mask")
    if len(layers["accepted_modules"]) != len(recovered):
        raise AssertionError("Thermal master accepted-module collection differs from counters/mask")
    if np.count_nonzero(layers["pv_scalar"][final_mask == 0]):
        raise AssertionError("PV scalar response exists outside the final PV mask")

    target = recovered[len(recovered) // 2]
    base_observation = ns["real_farm_add_module_observation"](
        recovered, [], target["module_id"], 7, 520052
    )
    fault_results = {}
    for name, diagnostic in (
        ("hotspot", None), ("bypass_diode", "SingleDiode"),
        ("bypassed_substring", "SingleByPassed"),
    ):
        observation = json.loads(json.dumps(base_observation))
        if diagnostic:
            observation["diagnostic_fault_type"] = diagnostic
        faulted = ns["real_farm_faulted_master"](layers, [observation])
        delta = np.abs(faulted - layers["master_scalar"])
        if float(delta.max(initial=0.0)) <= 0.1:
            raise AssertionError(f"{name} produced no visible scalar delta")
        if int(np.count_nonzero((delta > 1e-4) & (final_mask == 0))) > 0:
            raise AssertionError(f"{name} leaked outside final PV support")
        debug = observation.get("production_fault_debug", {})
        if debug.get("fallback_used") or debug.get("production_fault_render_failed"):
            raise AssertionError(f"{name} did not use its approved production renderer")
        fault_results[name] = {
            "scalar_peak": float(delta.max()),
            "reference_id": debug.get("reference_id"),
            "fallback_used": bool(debug.get("fallback_used")),
        }
        if diagnostic:
            ns["real_farm_close_thermal_image"](faulted, target).save(OUTPUT / f"07_{name}_close.png")
        else:
            thermal_image(ns, faulted).save(OUTPUT / "06_hotspot_full_scene.png")

    rgb_roi = image.copy()
    ImageDraw.Draw(rgb_roi).line(SEARCH_ROI + [SEARCH_ROI[0]], fill=(56, 189, 248), width=8)
    rgb_roi.save(OUTPUT / "01_rgb_search_hint.png")
    ns["real_farm_annotated_preview"](image, recovered, target["module_id"], max_width=4000).save(
        OUTPUT / "02_all_recovered_module_outlines.png"
    )
    Image.fromarray(final_mask, "L").save(OUTPUT / "03_final_full_farm_pv_mask.png")
    thermal_image(ns, layers["master_scalar"]).save(OUTPUT / "04_healthy_real_v2_thermal.png")

    first_only = ns["real_farm_module_mask"](image.size, [recovered[0]], active=True)
    before_after = Image.new("RGB", (image.width * 2, image.height), "black")
    before_after.paste(Image.fromarray(first_only, "L").convert("RGB"), (0, 0))
    before_after.paste(Image.fromarray(final_mask, "L").convert("RGB"), (image.width, 0))
    draw = ImageDraw.Draw(before_after)
    draw.rectangle((0, 0, 850, 80), fill=(0, 0, 0)); draw.text((20, 25), "BEFORE: selected module only", fill="white")
    draw.rectangle((image.width, 0, image.width + 1100, 80), fill=(0, 0, 0)); draw.text((image.width + 20, 25), "AFTER: union of all recovered modules", fill="white")
    before_after.save(OUTPUT / "05_before_after_mask.jpg", quality=92)

    # Fresh-upload normal path: the full image is only a search hint. Its
    # recovered children, rather than SEARCH001, must drive the scene mask.
    full_state = ns["real_farm_default_state"](None, image, SOURCE.name, "full-farm-qa")
    full_recovery = ns["real_farm_recover_full_farm"](full_state, image)
    if not full_recovery.get("success"):
        raise AssertionError(full_recovery.get("message"))
    full_modules = ns["real_farm_recovered_modules"](
        full_state["modules"], image_size=image.size
    )
    full_layers = ns["real_farm_build_thermal_master"](image, full_state["modules"])
    full_mask = full_layers["pv_module_union_mask"]
    if len(full_modules) != len(full_layers["accepted_modules"]):
        raise AssertionError("Fresh recovery counters and thermal collection differ")
    if not np.array_equal(
        full_mask, ns["real_farm_module_mask"](image.size, full_modules, active=True)
    ):
        raise AssertionError("Fresh recovery thermal mask differs from recovered-module union")
    ns["real_farm_annotated_preview"](image, full_modules, max_width=4000).save(
        OUTPUT / "08_fresh_full_image_recovered_outlines.png"
    )
    Image.fromarray(full_mask, "L").save(OUTPUT / "09_fresh_full_image_final_mask.png")
    thermal_image(ns, full_layers["master_scalar"]).save(OUTPUT / "10_fresh_full_image_thermal.png")

    report = {
        "result": "PASS",
        "source": str(SOURCE),
        "search_roi_is_final_mask": False,
        "detected_modules": summary["detected"],
        "accepted_modules": summary["accepted"],
        "warning_modules": summary["warnings"],
        "rejected_geometry": summary["rejected"],
        "thermal_master_accepted_modules": len(layers["accepted_modules"]),
        "modules_with_nonzero_mask": sum(pixels > 0 for pixels in per_module_pixels),
        "final_mask_pixels": int(np.count_nonzero(final_mask)),
        "leakage_outside_refined_support_pixels": leakage,
        "selected_module": target["module_id"],
        "faults": fault_results,
        "fresh_full_image_recovery": {
            "detected_modules": len(full_modules),
            "accepted_modules": len(full_layers["accepted_modules"]),
            "mask_pixels": int(np.count_nonzero(full_mask)),
            "search_roi_is_final_mask": False,
        },
    }
    (OUTPUT / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
