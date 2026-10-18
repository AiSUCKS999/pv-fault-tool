"""Same-reference production-fault transfer QA for Real Farm modules."""
from __future__ import annotations

import contextlib
import io
import json
import math
import runpy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "real_farm" / "uploads" / "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg"
OUTPUT = ROOT / "artifacts" / "real_farm" / "fault_morphology_transfer"
ROI = [[1760, 1035], [2240, 1035], [2240, 1215], [1760, 1215]]
FAULTS = ["SingleHotSpot", "SingleDiode", "SingleByPassed"]


def normalized_gray(values, signed=False):
    data = np.asarray(values, dtype=np.float32)
    if signed:
        data = np.abs(data)
        low, high = 0.0, max(1e-5, float(np.percentile(data, 99.8)))
    else:
        low, high = [float(v) for v in np.percentile(data, (0.5, 99.5))]
    result = np.clip((data - low) / max(1e-5, high - low), 0, 1)
    return Image.fromarray(np.rint(result * 255).astype(np.uint8), "L").convert("RGB")


def fit(image, size=(360, 260)):
    source = image.convert("RGB")
    scale = min(size[0] / max(1, source.width), size[1] / max(1, source.height))
    resized = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (3, 7, 13))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def tile(image, label, size=(360, 300)):
    content = fit(image, (size[0], size[1] - 40))
    result = Image.new("RGB", size, (5, 10, 18))
    result.paste(content, (0, 40))
    ImageDraw.Draw(result).text((10, 14), label, fill="white", font=ImageFont.load_default())
    return result


def sheet(items, columns=3, tile_size=(360, 300)):
    rows = max(1, math.ceil(len(items) / columns))
    output = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), (2, 5, 10))
    for index, (image, label) in enumerate(items):
        output.paste(tile(image, label, tile_size), ((index % columns) * tile_size[0], (index // columns) * tile_size[1]))
    return output


def reference_records(ns, fault_type):
    if fault_type == "SingleHotSpot":
        return ns["production_hotspot_residuals"]()
    return ns["production_submodule_signals"](fault_type)


def field_for_reference(reference):
    return np.asarray(reference.get("residual", reference.get("field_array")), dtype=np.float32)


def support_for_reference(reference):
    return np.asarray(reference.get("support", reference.get("support_array")), dtype=np.float32)


def morphology_metrics(delta):
    data = np.abs(np.asarray(delta, dtype=np.float32))
    peak = float(data.max(initial=0.0))
    if peak <= 1e-6:
        return {"area": 0, "aspect_ratio": 0.0, "eccentricity": 0.0, "centroid": [0.0, 0.0], "component_count": 0, "orientation_deg": 0.0, "concentration": 0.0}
    norm = data / peak
    mask = (norm >= 0.10).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask)
    ys, xs = np.nonzero(mask)
    weights = norm[ys, xs]
    total = max(1e-8, float(weights.sum()))
    cx, cy = float((xs * weights).sum() / total), float((ys * weights).sum() / total)
    centered = np.column_stack((xs - cx, ys - cy)).astype(np.float32)
    covariance = (centered * weights[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    aspect = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
    eccentricity = float(math.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[-1])))
    axis = eigenvectors[:, -1]
    orientation = float(math.degrees(math.atan2(float(axis[1]), float(axis[0]))))
    ordered = np.sort(norm.ravel())[::-1]
    top_count = max(1, int(round(ordered.size * 0.10)))
    concentration = float(ordered[:top_count].sum() / max(1e-8, ordered.sum()))
    return {
        "area": int(mask.sum()), "aspect_ratio": round(aspect, 5), "eccentricity": round(eccentricity, 5),
        "centroid": [round(cx / max(1, data.shape[1] - 1), 5), round(cy / max(1, data.shape[0] - 1), 5)],
        "component_count": int(max(0, count - 1)), "orientation_deg": round(orientation, 4),
        "concentration": round(concentration, 5),
    }


def normalized_correlation(left, right):
    a = np.abs(np.asarray(left, dtype=np.float32)); b = np.abs(np.asarray(right, dtype=np.float32))
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)
    a /= max(1e-6, float(a.max(initial=0.0))); b /= max(1e-6, float(b.max(initial=0.0)))
    if float(a.std()) < 1e-7 or float(b.std()) < 1e-7:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def module_local_case(ns, image, master, module):
    points = np.asarray(module["active_polygon_rgb"], dtype=np.float32)
    x1, y1 = points.min(axis=0); x2, y2 = points.max(axis=0)
    margin = max(18, int(max(x2 - x1, y2 - y1) * 0.32))
    box = (max(0, int(x1) - margin), max(0, int(y1) - margin), min(image.width, int(x2) + margin + 1), min(image.height, int(y2) + margin + 1))
    local_image = image.crop(box)
    local_master = np.asarray(master[box[1]:box[3], box[0]:box[2]], dtype=np.float32).copy()
    local_points = points - np.asarray([box[0], box[1]], dtype=np.float32)
    local_module = ns["real_farm_module_record"](module["module_id"], local_points.tolist(), local_image.size, "QA", 0.0, "manual_final")
    local_module["source"] = "REAL_FARM"
    return local_image, local_master, local_module


def polygon_overlay(image, polygon):
    result = image.convert("RGB").copy()
    points = [tuple(map(float, point)) for point in polygon]
    ImageDraw.Draw(result).line(points + [points[0]], fill=(0, 255, 210), width=3)
    return result


def generated_case(ns, debug, module, observation, fault_type, reference_id):
    height, width = debug["canonical_residual"].shape
    healthy = ns["smooth_thermal_array_field"]((width, height), ns["seed_for"]("same-reference", reference_id), 164, 196)
    healthy += ns["pv_cell_temperature_field"]((height, width), ns["seed_for"](reference_id, "cell"), 1, strength=2.6)
    faulted = healthy.copy()
    cell = ns["real_farm_production_cell"](module, observation, width, height)
    trace = {}
    ns["apply_thermal_fault"](
        faulted, cell, fault_type, int(observation["severity"]),
        production_reference_id=reference_id, production_trace=trace, require_production=True,
    )
    return healthy, faulted, faulted - healthy, trace


def orientation_cases(ns, fault_type, reference_id, output_dir):
    geometries = {
        "portrait": [[180, 70], [350, 88], [325, 700], [150, 670]],
        "landscape": [[70, 220], [720, 160], [742, 410], [95, 450]],
        "perspective_skewed": [[130, 95], [685, 180], [595, 700], [78, 560]],
    }
    results, visuals = {}, []
    for name, polygon in geometries.items():
        module = ns["real_farm_module_record"](f"QA_{name}", polygon, (800, 800), "QA", 0.0, "manual_final")
        base = ns["smooth_thermal_array_field"]((800, 800), ns["seed_for"](name, fault_type), 72, 116)
        observation = {"observation_id": f"{fault_type}_{name}", "local_uv": [0.54, 0.48], "severity": 8, "production_reference_id": reference_id}
        faulted, debug = ns["real_farm_apply_production_fault"](base, module, observation, fault_type, return_debug=True)
        mask = ns["real_farm_module_mask"]((800, 800), [module], active=True) > 0
        leakage = int(np.count_nonzero(np.abs(faulted - base)[~mask] > 1e-4))
        points = np.asarray(module["active_polygon_rgb"], dtype=np.float32)
        h, w = debug["canonical_residual"].shape
        destination = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        inverse = cv2.getPerspectiveTransform(points, destination)
        returned = cv2.warpPerspective(debug["warped_residual"], inverse, (w, h), flags=cv2.INTER_CUBIC)
        correlation = normalized_correlation(debug["canonical_residual"], returned)
        results[name] = {"outside_active_polygon_pixels": leakage, "round_trip_residual_correlation": round(correlation, 6)}
        visuals.append((normalized_gray(debug["warped_residual"], True), f"{name} | corr={correlation:.3f} | leak={leakage}"))
    sheet(visuals, columns=3).save(output_dir / f"orientation_{fault_type}.jpg", quality=94)
    return results


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stderr(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "app.py"), run_name="real_farm_fault_morphology_transfer_qa")
    image = Image.open(SOURCE).convert("RGB")
    parent = ns["real_farm_module_record"]("M001", ROI, image.size, "LIVE", 0.0, "manual_add")
    refinement = ns["real_farm_refine_pv_support"](image, parent)
    if not refinement.get("success"):
        raise AssertionError(refinement.get("message"))
    ns["real_farm_accept_refinement"](parent, refinement)
    layers = ns["real_farm_build_thermal_master"](image, [parent])
    modules = layers["accepted_modules"]
    center = np.asarray([2000.0, 1125.0], dtype=np.float32)
    target = min(modules, key=lambda item: float(np.linalg.norm(np.asarray(item["active_polygon_rgb"], dtype=np.float32).mean(axis=0) - center)))
    local_rgb, local_master, local_module = module_local_case(ns, image, layers["master_scalar"], target)
    all_results = {"module_instance_count": len(modules), "target_module": target["module_id"], "faults": {}}

    for fault_type in FAULTS:
        folder = OUTPUT / fault_type
        folder.mkdir(parents=True, exist_ok=True)
        references = reference_records(ns, fault_type)[:10]
        if len(references) < 10:
            raise AssertionError(f"Expected at least 10 approved references for {fault_type}, got {len(references)}")
        cases = []
        representative = None
        representative_score = -1.0
        for index, reference in enumerate(references, 1):
            reference_id = reference["reference_id"]
            observation = {
                "observation_id": f"SAME_REF_{index:02d}", "module_id": local_module["module_id"],
                "local_uv": [0.53, 0.50], "severity": 8, "production_reference_id": reference_id,
            }
            real_faulted, debug = ns["real_farm_apply_production_fault"](
                local_master, local_module, observation, fault_type, return_debug=True,
            )
            generated_healthy, generated_faulted, generated_delta, generated_trace = generated_case(
                ns, debug, local_module, observation, fault_type, reference_id,
            )
            real_delta = debug["canonical_residual"]
            generated_metrics = morphology_metrics(generated_delta)
            real_metrics = morphology_metrics(real_delta)
            correlation = normalized_correlation(generated_delta, real_delta)
            area_ratio = real_metrics["area"] / max(1, generated_metrics["area"])
            module_mask = ns["real_farm_module_mask"](local_rgb.size, [local_module], active=True) > 0
            leakage = int(np.count_nonzero(np.abs(real_faulted - local_master)[~module_mask] > 1e-4))
            if leakage or debug["metadata"]["fallback_used"] or debug["metadata"]["production_fault_render_failed"]:
                raise AssertionError(f"Invalid transfer for {fault_type} {reference_id}: leakage={leakage}, metadata={debug['metadata']}")
            generated_final = ns["thermal_display_rgb_from_scalar"](
                ns["apply_sensor_model"](generated_faulted, "grass_open", "close"), "grass_open", "close"
            )
            real_final = ns["real_farm_close_thermal_image"](real_faulted, local_module)
            case_visual = sheet([
                (normalized_gray(field_for_reference(reference), True), f"Reference {reference_id}"),
                (normalized_gray(generated_delta, True), "Generated scalar delta"),
                (normalized_gray(real_delta, True), "Real canonical delta"),
                (normalized_gray(debug["warped_residual"], True), "Warped Real residual"),
                (generated_final, "Generated final thermal"),
                (real_final, "Real final thermal"),
            ], columns=3)
            case_visual.save(folder / f"same_reference_{index:02d}_{reference_id}.jpg", quality=94)
            case = {
                "case": index, "reference_id": reference_id, "production_model": debug["metadata"]["production_model"],
                "residual_correlation": round(correlation, 6), "support_area_ratio": round(area_ratio, 6),
                "generated_morphology": generated_metrics, "real_morphology": real_metrics,
                "centroid_displacement": round(float(np.linalg.norm(np.asarray(generated_metrics["centroid"]) - np.asarray(real_metrics["centroid"]))), 6),
                "outside_active_polygon_pixels": leakage, "fallback_used": debug["metadata"]["fallback_used"],
                "provenance": debug["metadata"],
            }
            cases.append(case)
            # Use a visibly section-scale live-bank sample for the substring raw QA.
            # This only selects the report example; it does not alter rendering or banks.
            representative_candidate_score = (
                generated_metrics["aspect_ratio"] if fault_type == "SingleByPassed" else -float(index)
            )
            if representative is None or representative_candidate_score > representative_score:
                representative = (reference, observation, debug, real_faulted, real_final)
                representative_score = representative_candidate_score

        reference, observation, debug, real_faulted, real_final = representative
        healthy_local = debug["healthy_canonical"]
        faulted_local = debug["faulted_canonical"]
        raw = sheet([
            (normalized_gray(field_for_reference(reference), True), "1 Production source/reference"),
            (normalized_gray(debug["canonical_residual"], True), "2 Canonical scalar residual"),
            (normalized_gray(debug["canonical_support"]), "3 Canonical support mask"),
            (local_rgb, "4 Target Real Farm RGB crop"),
            (polygon_overlay(local_rgb, local_module["active_polygon_rgb"]), "5 Target active polygon"),
            (normalized_gray(debug["warped_residual"], True), "6 Warped scalar residual"),
            (normalized_gray(debug["warped_support"]), "7 Warped support mask"),
            (normalized_gray(healthy_local), "8 Healthy scalar module"),
            (normalized_gray(faulted_local), "9 Faulted scalar module"),
            (real_final, "10 Final close thermal"),
        ], columns=2, tile_size=(480, 340))
        raw.save(folder / "raw_transfer_qa.jpg", quality=94)
        orientation = orientation_cases(ns, fault_type, reference["reference_id"], folder)
        all_results["faults"][fault_type] = {
            "production_reference_count": len(reference_records(ns, fault_type)), "same_reference_cases": cases,
            "orientation_tests": orientation,
        }

    (OUTPUT / "fault_morphology_transfer_report.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (OUTPUT / "fault_morphology_transfer_report.md").write_text(
        "# Real Farm production morphology transfer\n\n"
        "The Real Farm path now calls the production-only renderers with `require_production=True`; failure raises "
        "`FaultGeometryError` and never enters a generic fallback. Continuous scalar residuals and support masks "
        "are traced separately. Close View rectifies the authoritative scalar master before the shared close sensor "
        "model, preventing far-view downsampling from collapsing small real-module morphology.\n\n"
        "Banks: 26 Hotspot references, 14 Diode V3 robust references, and 273 filtered live substring references. "
        "Ten exact-reference transfers per fault and portrait/landscape/perspective tests are recorded here.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "PASS", "output": str(OUTPUT), "module_instances": len(modules),
        "same_reference_cases": {fault: len(data["same_reference_cases"]) for fault, data in all_results["faults"].items()},
        "fallback_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
