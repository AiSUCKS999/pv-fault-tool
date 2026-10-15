"""Generate Real Farm orientation/perspective robustness evidence."""
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
OUTPUT = ROOT / "artifacts" / "real_farm" / "robust_mask_generation"


CASES = [
    ("01_near_nadir_landscape", "clean near-nadir landscape modules", "grass", [
        [[90, 90], [310, 90], [310, 205], [90, 205]], [[360, 90], [580, 90], [580, 205], [360, 205]], [[630, 90], [850, 90], [850, 205], [630, 205]],
    ]),
    ("02_portrait", "portrait modules", "grass", [
        [[130, 70], [260, 70], [260, 330], [130, 330]], [[365, 90], [495, 90], [495, 350], [365, 350]], [[610, 110], [740, 110], [740, 370], [610, 370]],
    ]),
    ("03_skewed_perspective", "skewed and perspective modules", "earth", [
        [[75, 105], [340, 70], [310, 245], [105, 270]], [[390, 95], [710, 145], [655, 330], [430, 280]], [[735, 190], [930, 250], [875, 430], [700, 345]],
    ]),
    ("04_edge_trapezoids", "trapezoidal modules near image edges", "earth", [
        [[-18, 60], [245, 95], [205, 285], [5, 250]], [[745, 320], [971, 285], [945, 555], [700, 590]],
    ]),
    ("05_mixed_orientations", "mixed orientations in one image", "grass", [
        [[75, 80], [310, 105], [295, 225], [62, 195]], [[390, 55], [510, 70], [480, 335], [360, 310]], [[610, 110], [870, 65], [895, 190], [635, 230]], [[625, 300], [760, 275], [800, 560], [655, 580]],
    ]),
    ("06_incomplete_rows", "incomplete and ragged rows", "grass", [
        [[80, 80], [285, 80], [285, 190], [80, 190]], [[340, 80], [545, 80], [545, 190], [340, 190]], [[80, 285], [285, 285], [285, 395], [80, 395]], [[600, 285], [805, 285], [805, 395], [600, 395]],
    ]),
    ("07_large_spacing", "large spacing between arrays", "earth", [
        [[55, 60], [260, 75], [250, 185], [45, 165]], [[690, 90], [910, 110], [895, 230], [675, 205]], [[355, 430], [585, 410], [600, 535], [370, 560]],
    ]),
    ("08_rooftop_irregular", "rooftop irregular modules", "roof", [
        [[85, 105], [330, 65], [345, 185], [100, 225]], [[375, 80], [575, 105], [550, 300], [345, 270]], [[620, 135], [900, 100], [870, 245], [600, 285]],
    ]),
    ("09_floating_solar", "floating solar and unusual perspective", "water", [
        [[75, 155], [315, 105], [355, 230], [110, 290]], [[380, 115], [665, 150], [630, 300], [355, 255]], [[680, 185], [925, 245], [850, 390], [640, 320]],
    ]),
    ("10_low_texture_manual", "manual modules with low texture confidence", "low_texture", [
        [[100, 115], [405, 95], [390, 285], [85, 305]], [[510, 180], [830, 155], [815, 370], [500, 395]],
    ]),
]


def scene(kind: str, seed: int) -> Image.Image:
    h, w = 640, 960
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "water":
        base = np.dstack((48 + 8*np.sin(yy/27), 92 + 15*np.sin(xx/45), 122 + 20*np.sin((xx+yy)/35)))
    elif kind == "roof":
        base = np.dstack((118 + 12*np.sin(xx/55), 117 + 10*np.sin(yy/39), 114 + 8*np.sin((xx+yy)/60)))
    elif kind == "earth":
        base = np.dstack((132 + 22*np.sin(xx/67), 107 + 18*np.sin(yy/51), 72 + 10*np.sin((xx+yy)/43)))
    elif kind == "low_texture":
        base = np.dstack((110 + 3*np.sin(xx/90), 140 + 3*np.sin(yy/85), 110 + 2*np.sin((xx+yy)/110)))
    else:
        base = np.dstack((83 + 15*np.sin(xx/60), 139 + 20*np.sin(yy/50), 69 + 12*np.sin((xx+yy)/49)))
    base += rng.normal(0, 5 if kind != "low_texture" else 1.2, (h, w, 1))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def paint_modules(image: Image.Image, polygons, low_texture=False) -> Image.Image:
    rgb = np.asarray(image).copy()
    for index, polygon in enumerate(polygons):
        quad = np.asarray(polygon, dtype=np.float32)
        if low_texture:
            color, line = (108, 135, 108), (112, 140, 112)
        else:
            color, line = (31 + index * 3, 57 + index * 2, 88 + index * 4), (112, 137, 162)
        cv2.fillConvexPoly(rgb, np.rint(quad).astype(np.int32), color, lineType=cv2.LINE_AA)
        for u in np.linspace(0.15, 0.85, 5):
            top = quad[0] * (1-u) + quad[1] * u
            bottom = quad[3] * (1-u) + quad[2] * u
            cv2.line(rgb, tuple(np.rint(top).astype(int)), tuple(np.rint(bottom).astype(int)), line, 1, cv2.LINE_AA)
        for v in np.linspace(0.15, 0.85, 6):
            left = quad[0] * (1-v) + quad[3] * v
            right = quad[1] * (1-v) + quad[2] * v
            cv2.line(rgb, tuple(np.rint(left).astype(int)), tuple(np.rint(right).astype(int)), line, 1, cv2.LINE_AA)
    return Image.fromarray(rgb)


def observation(module_id, observation_type=None, diagnostic=None):
    return [{
        "observation_id": "RFQA001", "module_id": module_id, "local_uv": [0.42, 0.48],
        "center_rgb_px": [0, 0], "severity": 8, "observation_type": observation_type,
        "diagnostic_fault_type": diagnostic,
    }]


def normalize_scalar(array):
    values = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    return Image.fromarray(np.clip((values-low) * 255 / max(1e-6, high-low), 0, 255).astype(np.uint8), "L")


def sheet(items, path):
    cell_w, cell_h, label_h = 430, 275, 28
    canvas = Image.new("RGB", (cell_w*4, (cell_h+label_h)*2), (8, 13, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, item) in enumerate(items):
        preview = item.convert("RGB").copy()
        preview.thumbnail((cell_w-12, cell_h-12), Image.Resampling.LANCZOS)
        x = (index % 4)*cell_w + (cell_w-preview.width)//2
        y = (index // 4)*(cell_h+label_h) + label_h + (cell_h-preview.height)//2
        canvas.paste(preview, (x, y))
        draw.text(((index % 4)*cell_w+8, (index // 4)*(cell_h+label_h)+8), label, fill="white", font=font)
    canvas.save(path, quality=92)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_mask_robustness")
    totals = {"valid_polygons": 0, "blocked_polygons": 0, "warning_only_polygons": 0, "successful_thermal_masks": 0, "successful_fault_renderings": 0}
    case_rows, warning_rows = [], []
    first_low_images = None
    for case_index, (slug, title, background_kind, polygons) in enumerate(CASES, 1):
        folder = OUTPUT / slug
        folder.mkdir(parents=True, exist_ok=True)
        rgb = paint_modules(scene(background_kind, 800+case_index), polygons, background_kind == "low_texture")
        modules = [ns["real_farm_module_record"](f"M{i:03d}", points, rgb.size, f"A{case_index}", 0.02, f"robustness:{slug}") for i, points in enumerate(polygons, 1)]
        accepted, blocked = ns["real_farm_thermal_modules"](rgb, modules)
        if len(accepted) != len(modules):
            raise AssertionError(f"{slug}: mechanically valid manual geometry was blocked")
        layers = ns["real_farm_build_thermal_master"](rgb, modules)
        mask = Image.fromarray(layers["pv_module_union_mask"], "L")
        thermal = ns["real_farm_palette_image"](layers["master_scalar"], layers["display_range"])
        target = accepted[0]["module_id"]
        hotspot_observation = observation(target, "SingleHotSpot")
        diode_observation = observation(target, None, "SingleDiode")
        substring_observation = observation(target, None, "SingleByPassed")
        hotspot_scalar = ns["real_farm_faulted_master"](layers, hotspot_observation)
        diode_scalar = ns["real_farm_faulted_master"](layers, diode_observation)
        substring_scalar = ns["real_farm_faulted_master"](layers, substring_observation)
        outside = layers["pv_module_union_mask"] == 0
        for fault_scalar in (hotspot_scalar, diode_scalar, substring_scalar):
            if float(np.max(np.abs(fault_scalar[outside] - layers["master_scalar"][outside]))) > 1e-4:
                raise AssertionError(f"{slug}: production fault leaked outside module geometry")
        hotspot = ns["real_farm_palette_image"](hotspot_scalar, layers["display_range"])
        diode = ns["real_farm_palette_image"](diode_scalar, layers["display_range"])
        substring = ns["real_farm_palette_image"](substring_scalar, layers["display_range"])
        overlay = ns["real_farm_annotated_preview"](rgb, accepted)
        canonical = ns["real_farm_rectified_contact_sheet"](rgb, accepted)
        rgb.save(folder / "original_rgb.png")
        overlay.save(folder / "polygon_overlay.png")
        canonical.save(folder / "canonical_rectified_modules.png")
        mask.save(folder / "pv_union_mask.png")
        thermal.save(folder / "thermal_proxy.png")
        hotspot.save(folder / "hotspot.png")
        diode.save(folder / "bypass_diode.png")
        substring.save(folder / "bypassed_substring.png")
        sheet([
            ("Original RGB", rgb), ("Manual polygon overlay", overlay), ("Canonical rectification", canonical), ("PV union mask", mask),
            ("Thermal proxy", thermal), ("Thermal hotspot", hotspot), ("Bypass diode", diode), ("Bypassed substring", substring),
        ], folder / "qa_sheet.jpg")
        warning_modules = [item for item in accepted if item.get("warnings")]
        totals["valid_polygons"] += len(accepted)
        totals["blocked_polygons"] += len(blocked)
        totals["warning_only_polygons"] += len(warning_modules)
        totals["successful_thermal_masks"] += 1
        totals["successful_fault_renderings"] += 3
        case_rows.append((slug, title, len(accepted), len(blocked), len(warning_modules), "PASS"))
        for item in warning_modules:
            warning_rows.append({"case": slug, "module_id": item["module_id"], "visual_score": item["thermal_geometry_score"], "warnings": item["warnings"]})
        if slug == "10_low_texture_manual":
            if not all(item["visually_supported"] is False for item in accepted):
                raise AssertionError("low-texture case did not exercise advisory visual warnings")
            first_low_images = (overlay, thermal)

    invalid = ns["real_farm_module_record"]("INVALID", [[20, 20], [20, 20], [21, 21], [20, 21]], (960, 640), "QA", 0.0, "hard_invalid_control")
    invalid_accepted, invalid_blocked = ns["real_farm_thermal_modules"](scene("grass", 999), [invalid])
    assert not invalid_accepted and len(invalid_blocked) == 1
    totals["blocked_polygons"] += 1
    (OUTPUT / "warnings_summary.json").write_text(json.dumps(warning_rows, indent=2), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    if first_low_images:
        before = first_low_images[0].copy()
        draw = ImageDraw.Draw(before)
        draw.rectangle((15, 15, 510, 62), fill=(90, 18, 28))
        draw.text((25, 29), "BEFORE: weak material score blocked generation", fill="white", font=ImageFont.load_default())
        after = first_low_images[1].copy()
        draw = ImageDraw.Draw(after)
        draw.rectangle((15, 15, 530, 62), fill=(12, 78, 48))
        draw.text((25, 29), "AFTER: valid manual geometry generates thermal proxy", fill="white", font=ImageFont.load_default())
        comparison = Image.new("RGB", (1920, 640), "black")
        comparison.paste(before, (0, 0)); comparison.paste(after, (960, 0))
        comparison.save(OUTPUT / "before_after_low_texture.jpg", quality=92)
    lines = [
        "# Real Farm robust mask generation report", "",
        "Manual quadrilateral geometry is authoritative. Mechanical validity is the only generation gate; RGB material confidence is advisory.", "",
        "| Case | Category | Valid | Blocked | Warning-only | Result |", "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(f"| `{slug}` | {title} | {valid} | {blocked} | {warnings} | {result} |" for slug, title, valid, blocked, warnings, result in case_rows)
    lines.extend(["", "## Totals", ""])
    lines.extend(f"- {key.replace('_', ' ').title()}: **{value}**" for key, value in totals.items())
    lines.extend(["", "The blocked count is the deliberately degenerate hard-invalid control. All valid manual polygons produced one mask and all three requested production-fault renderings.", ""])
    (OUTPUT / "geometry_acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"result": "PASS", "output": str(OUTPUT), **totals}, indent=2))


if __name__ == "__main__":
    main()
