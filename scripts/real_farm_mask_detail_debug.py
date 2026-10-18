"""Generate detailed segmentation QA for the exact Real Farm live case."""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "real_farm" / "detailed_mask_debug"
SOURCE = ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"
ROI = [[1760, 1035], [2240, 1035], [2240, 1215], [1760, 1215]]


def caption(image, text, color):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((16, 16, min(result.width - 16, 1130), 78), fill=color)
    draw.text((30, 37), text, fill="white", font=ImageFont.load_default())
    return result


def side_by_side(left, right, left_text, right_text):
    left = caption(left, left_text, (105, 20, 28))
    right = caption(right, right_text, (12, 86, 52))
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "black")
    canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
    return canvas


def component_preview(labels):
    labels = np.asarray(labels, dtype=np.int32)
    preview = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:
            continue
        rng = np.random.default_rng(int(label) * 7919)
        preview[labels == label] = rng.integers(48, 246, size=3, dtype=np.uint8)
    return Image.fromarray(preview, "RGB")


def outline_overlay(image, mask, color):
    result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, max(3, image.width // 1000), cv2.LINE_AA)
    return Image.fromarray(result, "RGB")


def instance_mask_preview(image_size, instances):
    width, height = image_size
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    for index, module in enumerate(instances):
        polygon = np.rint(np.asarray(module["active_polygon_rgb"], dtype=np.float32)).astype(np.int32)
        rng = np.random.default_rng(1709 + index * 7919)
        color = tuple(int(value) for value in rng.integers(58, 246, size=3))
        cv2.fillConvexPoly(preview, polygon, color, lineType=cv2.LINE_AA)
    return Image.fromarray(preview, "RGB")


def rectified_sheet(ns, image, instances, limit=12):
    tiles = []
    for module in instances[:limit]:
        crop = ns["real_farm_rectified_crop"](image, module).convert("RGB")
        crop.thumbnail((260, 260), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (280, 300), (7, 12, 20))
        tile.paste(crop, ((280 - crop.width) // 2, 28))
        ImageDraw.Draw(tile).text((10, 8), module["module_id"], fill="white", font=ImageFont.load_default())
        tiles.append(tile)
    columns = 4
    rows = max(1, int(np.ceil(len(tiles) / columns)))
    sheet = Image.new("RGB", (columns * 280, rows * 300), (3, 7, 13))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 280, (index // columns) * 300))
    return sheet


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_mask_detail_debug")
    image = Image.open(SOURCE).convert("RGB")
    module = ns["real_farm_module_record"]("M001", ROI, image.size, "LIVE", 0.0, "manual_add")
    result = ns["real_farm_refine_pv_support"](image, module)
    if not result.get("success"):
        raise AssertionError(result.get("message"))
    ns["real_farm_accept_refinement"](module, result)
    instances = module.get("module_instances", [])
    improved_mask = ns["real_farm_module_mask"](image.size, instances or [module], active=True)
    old_path = OUTPUT / "current_mask_before.png"
    if not old_path.exists():
        raise FileNotFoundError(f"Preserved pre-change mask is required: {old_path}")
    current_mask = np.asarray(Image.open(old_path).convert("L"), dtype=np.uint8)
    if current_mask.shape != improved_mask.shape:
        raise AssertionError("Preserved current mask has unexpected dimensions")

    x1, y1, x2, y2 = result["roi_bbox"]
    image.crop((x1, y1, x2, y2)).save(OUTPUT / "roi_crop_rgb.png")
    Image.fromarray(result["raw_mask"][y1:y2, x1:x2], "L").save(OUTPUT / "pv_candidates_raw.png")
    Image.fromarray(result["cleaned_mask"][y1:y2, x1:x2], "L").save(OUTPUT / "pv_candidates_cleaned.png")
    component_preview(result["connected_component_labels"]).save(OUTPUT / "connected_components_labeled.png")
    refined_overlay = ns["real_farm_refined_overlay"](image, module)
    refined_overlay.save(OUTPUT / "refined_outline_overlay.png")
    Image.fromarray(improved_mask, "L").save(OUTPUT / "final_binary_pv_mask.png")
    ns["real_farm_instance_overlay"](image, module).save(OUTPUT / "module_instance_overlay.png")
    instance_mask_preview(image.size, instances).save(OUTPUT / "module_instance_masks.png")
    rectified_sheet(ns, image, instances).save(OUTPUT / "rectified_example_modules.png")

    old_outline = outline_overlay(image, current_mask, (235, 65, 54))
    new_outline = outline_overlay(image, improved_mask, (34, 197, 94))
    side_by_side(old_outline, new_outline, "BEFORE: merged outer strip", "AFTER: detailed PV support and frame gaps").save(
        OUTPUT / "refined_outline_before_after.jpg", quality=93
    )
    side_by_side(
        Image.fromarray(current_mask, "L").convert("RGB"),
        Image.fromarray(improved_mask, "L").convert("RGB"),
        "CURRENT MASK", "IMPROVED MASK",
    ).save(OUTPUT / "current_vs_improved_mask.jpg", quality=93)
    focus = (1450, 180, 2550, 2190)
    side_by_side(
        Image.fromarray(current_mask, "L").convert("RGB").crop(focus),
        Image.fromarray(improved_mask, "L").convert("RGB").crop(focus),
        "CURRENT MASK (DETAIL)", "IMPROVED MASK (DETAIL)",
    ).save(OUTPUT / "current_vs_improved_mask_detail.jpg", quality=94)
    side_by_side(
        old_outline.crop(focus), new_outline.crop(focus),
        "BEFORE OUTLINE (DETAIL)", "AFTER OUTLINE (DETAIL)",
    ).save(OUTPUT / "refined_outline_before_after_detail.jpg", quality=94)

    old_pixels = int(np.count_nonzero(current_mask))
    new_pixels = int(np.count_nonzero(improved_mask))
    detail_gaps = int(np.count_nonzero((current_mask > 0) & (improved_mask == 0)))
    if new_pixels == 0 or result.get("selected_component_count", 0) < 1:
        raise AssertionError("Improved refinement did not produce usable support")
    report = {
        "result": "PASS",
        "source": str(SOURCE),
        "roi": ROI,
        "processing_bbox": [int(value) for value in result["roi_bbox"]],
        "candidate_components": result.get("candidate_component_count"),
        "selected_components": result.get("selected_component_count"),
        "module_instance_count": len(instances),
        "exterior_contours": len(result.get("refined_mask_contours_rgb", [])),
        "preserved_gap_contours": len(result.get("refined_mask_holes_rgb", [])),
        "old_mask_pixels": old_pixels,
        "improved_mask_pixels": new_pixels,
        "pixels_removed_from_old_slab_as_structure": detail_gaps,
    }
    (OUTPUT / "mask_detail_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **report}, indent=2))


if __name__ == "__main__":
    main()
