"""Compare raw production fault deltas with the Real Farm realism compositor."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "real_farm" / "fault_realism_debug"
SOURCE = ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"
ROI = [[1760, 1035], [2240, 1035], [2240, 1215], [1760, 1215]]


def caption(image, text, color):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((16, 16, min(result.width - 16, 1100), 78), fill=color)
    draw.text((30, 37), text, fill="white", font=ImageFont.load_default())
    return result


def compare(left, right, left_text, right_text):
    left = caption(left, left_text, (105, 20, 28)); right = caption(right, right_text, (12, 86, 52))
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "black")
    canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
    return canvas


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_fault_realism_debug")
    image = Image.open(SOURCE).convert("RGB")
    module = ns["real_farm_module_record"]("M001", ROI, image.size, "LIVE", 0.0, "manual_add")
    refinement = ns["real_farm_refine_pv_support"](image, module)
    ns["real_farm_accept_refinement"](module, refinement)
    layers = ns["real_farm_build_thermal_master"](image, [module])
    mask = layers["pv_module_union_mask"]
    healthy = ns["real_farm_palette_image"](layers["master_scalar"], layers["display_range"])
    healthy.save(OUTPUT / "healthy_thermal_proxy_after.png")
    previous_healthy_path = ROOT / "artifacts" / "real_farm" / "live_mask_diagnostic" / "11_final_thermal_clean.png"
    if previous_healthy_path.exists():
        previous_healthy = Image.open(previous_healthy_path).convert("RGB")
        previous_healthy.save(OUTPUT / "healthy_thermal_proxy_before.png")
        compare(
            previous_healthy, healthy,
            "BEFORE: coarse support", "AFTER: detailed support",
        ).save(OUTPUT / "healthy_thermal_proxy_before_after.jpg", quality=93)

    observation = {
        "observation_id": "REALISM001", "module_id": "M001", "local_uv": [0.5, 0.70],
        "center_rgb_px": [0, 0], "severity": 8,
    }
    fault_types = {
        "hotspot": "SingleHotSpot",
        "bypass_diode": "SingleDiode",
        "bypassed_substring": "SingleByPassed",
    }
    report = {"result": "PASS", "source": str(SOURCE), "faults": {}}
    for label, fault_type in fault_types.items():
        raw_scalar = ns["real_farm_apply_production_fault"](
            layers["master_scalar"], module, observation, fault_type, realism=False
        )
        improved_scalar = ns["real_farm_apply_production_fault"](
            layers["master_scalar"], module, observation, fault_type, realism=True
        )
        outside_delta = float(np.max(np.abs(improved_scalar[mask == 0] - layers["master_scalar"][mask == 0])))
        if outside_delta > 1e-4:
            raise AssertionError(f"{label} leaked outside refined support: {outside_delta}")
        raw = ns["real_farm_palette_image"](raw_scalar, layers["display_range"])
        improved = ns["real_farm_palette_image"](improved_scalar, layers["display_range"])
        raw.save(OUTPUT / f"{label}_before_raw.png")
        improved.save(OUTPUT / f"{label}_after_blended.png")
        compare(raw, improved, f"BEFORE: raw {label} delta", f"AFTER: blended {label}").save(
            OUTPUT / f"{label}_before_after.jpg", quality=93
        )
        focus = (1450, 300, 2550, 2200)
        compare(
            raw.crop(focus), improved.crop(focus),
            f"BEFORE: raw {label} (detail)", f"AFTER: blended {label} (detail)",
        ).save(OUTPUT / f"{label}_before_after_detail.jpg", quality=94)
        raw_delta = np.abs(raw_scalar - layers["master_scalar"])
        improved_delta = np.abs(improved_scalar - layers["master_scalar"])
        report["faults"][label] = {
            "raw_peak_delta": round(float(raw_delta.max()), 4),
            "blended_peak_delta": round(float(improved_delta.max()), 4),
            "outside_mask_peak_delta": round(outside_delta, 6),
        }
    (OUTPUT / "fault_realism_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **report}, indent=2))


if __name__ == "__main__":
    main()
