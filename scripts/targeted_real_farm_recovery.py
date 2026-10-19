"""Targeted Real Farm recovery check across the locally supplied farm photos."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "artifacts" / "real_farm" / "uploads"
OUTPUT = ROOT / "artifacts" / "real_farm" / "targeted_tightening"
DENSE_REFERENCE = "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"


def clean_module_overlay(image, modules, max_width=1800):
    """Draw one uncluttered cyan quadrilateral per recovered module."""
    scale = min(1.0, max_width / float(max(1, image.width)))
    preview = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(preview)
    for module in modules:
        polygon = [
            (round(float(x) * scale), round(float(y) * scale))
            for x, y in module.get("outer_polygon_rgb", [])
        ]
        if len(polygon) >= 3:
            draw.line(polygon + [polygon[0]], fill=(0, 235, 255), width=2, joint="curve")
    return preview


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        namespace = runpy.run_path(str(ROOT / "app.py"), run_name="targeted_real_farm_recovery")
    report = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    requested = set(sys.argv[1:])
    for path in sorted(UPLOADS.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        if requested and path.name not in requested:
            continue
        image = Image.open(path).convert("RGB")
        state = namespace["real_farm_default_state"](None, image, path.name, path.stem)
        stages = []
        result = namespace["real_farm_recover_full_farm"](
            state, image, progress_callback=stages.append
        )
        if not result.get("success"):
            raise AssertionError(f"{path.name}: {result.get('message')}")
        recovered = namespace["real_farm_recovered_modules"](
            state["modules"], image_size=image.size
        )
        union = namespace["real_farm_module_mask"](image.size, recovered, active=True)
        if len(recovered) != result.get("accepted_count"):
            raise AssertionError(f"{path.name}: accepted count differs from canonical collection")
        if not np.array_equal(union, result["final_pv_mask"]):
            raise AssertionError(f"{path.name}: final mask differs from canonical module union")
        if path.name == DENSE_REFERENCE and len(recovered) != 307:
            raise AssertionError(f"Dense reference changed from 307 to {len(recovered)} modules")
        namespace["real_farm_annotated_preview"](
            image, recovered, max_width=1800
        ).save(OUTPUT / f"{path.stem}_modules.jpg", quality=90)
        clean_module_overlay(image, recovered).save(
            OUTPUT / f"{path.stem}_module_instances.jpg", quality=92
        )
        Image.fromarray(union, "L").save(OUTPUT / f"{path.stem}_mask.png")
        areas = []
        for module in recovered:
            points = np.asarray(module["outer_polygon_rgb"], dtype=np.float32)
            areas.append(abs(float(cv2.contourArea(points))))
        median_area = float(np.median(areas)) if areas else 0.0
        oversized = sum(area > median_area * 5.0 for area in areas) if median_area else 0
        visual_scores = [
            (module.get("module_id"), module.get("visual_pv_score"))
            for module in recovered if module.get("visual_pv_score") is not None
        ]
        grid_scores = [
            (module.get("module_id"), module.get("grid_texture_score"))
            for module in recovered if module.get("grid_texture_score") is not None
        ]
        unsplit_ids = [
            module.get("module_id")
            for module in recovered
            if any(
                notice.get("code") == "unsplit_recovered_region"
                for notice in module.get("warnings", [])
            )
        ]
        report.append({
            "image": path.name,
            "size": list(image.size),
            "modules": len(recovered),
            "accepted": result.get("accepted_count"),
            "rejected": result.get("rejected_count"),
            "mask_pixels": int(np.count_nonzero(union)),
            "oversized_vs_median": oversized,
            "unsplit_recovered_regions": len(unsplit_ids),
            "unsplit_module_ids": unsplit_ids,
            "lowest_visual_scores": sorted(visual_scores, key=lambda item: item[1])[:8],
            "lowest_grid_scores": sorted(grid_scores, key=lambda item: item[1])[:8],
            "stages": stages,
        })
    if not report:
        raise AssertionError(f"No Real Farm images found in {UPLOADS}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
