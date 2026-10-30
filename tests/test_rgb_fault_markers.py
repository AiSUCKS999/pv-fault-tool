"""Location annotations are distinct from physical RGB surface rendering."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pv_real_farm_rgb_view as rgb_view


@pytest.fixture
def scene(monkeypatch):
    image = Image.new("RGB", (160, 120), (35, 65, 95))
    source = hashlib.sha256(image.tobytes()).hexdigest()
    module = {"module_id": "M001", "active_polygon_rgb": [[40, 25], [130, 25], [130, 100], [40, 100]]}
    observation = {"observation_id": "RFH001", "module_id": "M001", "observation_type": "SingleHotSpot",
                   "diagnostic_fault_type": None, "center_rgb_px": [82, 62], "local_uv": [.5, .5],
                   "source_rgb_pixel_sha256": source}

    def build(observations=None, modules=None):
        observations = deepcopy(observations if observations is not None else [observation])
        modules = deepcopy(modules if modules is not None else [module])
        base = image.copy()
        base.paste((120, 50, 15), (5, 5, 20, 15))
        report = {"source_rgb_pixel_sha256": source, "simulation_only": True,
                  "physical_ground_truth_verified": False, "rendered": [], "unavailable": [],
                  "palette": "original natural RGB; existing RGB surface effects"}
        for item in observations:
            fault = item.get("diagnostic_fault_type") or item.get("observation_type", "SingleHotSpot")
            entry = {"observation_id": item["observation_id"], "fault_type": fault}
            if fault in rgb_view.VISIBLE_RGB_EFFECTS:
                report["rendered"].append({**entry, "changed_pixels": 150, "engine": "existing surface renderer"})
            else:
                report["unavailable"].append({**entry, "reason": "No established daylight RGB surface effect"})
        session = {"real_farm_rgb_preview_cache": ("natural", base, report), "unrelated": {"keep": True}}

        def mechanical(polygon, size):
            q = np.asarray(polygon, dtype=np.float32)
            valid = q.shape == (4, 2) and np.isfinite(q).all() and abs(cv2.contourArea(q)) > 1
            return {"mechanically_valid": bool(valid)}

        ns = {"real_farm_source_identity": lambda value: hashlib.sha256(value.tobytes()).hexdigest(),
              "real_farm_mechanical_validation": mechanical, "st": SimpleNamespace(session_state=session)}
        monkeypatch.setattr(rgb_view, "rgb_fault_preview", lambda *args: (base, report))
        return SimpleNamespace(image=image, source=source, module=module, observation=observation,
                               observations=observations, modules=modules, base=base, report=report, ns=ns)
    return build


def preview(s):
    return rgb_view.annotated_rgb_fault_preview(s.ns, s.image, s.modules, s.observations)


def test_marker_uses_actual_center_and_preserves_source_geometry_and_cached_surface_image(scene):
    s = scene()
    image_bytes, base_bytes = s.image.tobytes(), s.base.tobytes()
    observations, modules, report = deepcopy(s.observations), deepcopy(s.modules), deepcopy(s.report)
    output, metadata = preview(s)
    assert output.size == s.image.size
    changed = np.any(np.asarray(output) != np.asarray(s.base), axis=2)
    assert changed[54:71, 74:91].any(), "Marker must be at the actual [82, 62] observation position"
    assert s.image.tobytes() == image_bytes and s.base.tobytes() == base_bytes
    assert s.observations == observations and s.modules == modules and s.report == report
    assert s.ns["st"].session_state["unrelated"] == {"keep": True}
    assert s.ns["st"].session_state["real_farm_rgb_preview_cache"][1] is s.base
    assert metadata["annotated"][0]["observation_id"] == "RFH001"
    assert metadata["annotated"][0]["center_rgb_px"] == [82, 62]
    assert metadata["annotated"][0]["marker_kind"] == "location_annotation"
    assert metadata["annotated"][0]["label"]
    assert metadata["rendered"] == [] and metadata["unavailable"] == []
    assert not metadata["physical_ground_truth_verified"]


def test_real_surface_effects_are_retained_and_never_replaced_by_a_marker(scene):
    s = scene()
    surface = deepcopy(s.observation)
    surface.update(observation_id="RFH002", diagnostic_fault_type="SoilingDust")
    s = scene([s.observation, surface])
    output, report = preview(s)
    assert np.array_equal(np.asarray(output)[5:15, 5:20], np.asarray(s.base)[5:15, 5:20])
    assert [entry["observation_id"] for entry in report["rendered"]] == ["RFH002"]
    assert [entry["observation_id"] for entry in report["annotated"]] == ["RFH001"]
    surface_only = scene([surface])
    output, report = preview(surface_only)
    assert output.tobytes() == surface_only.base.tobytes()
    assert report["annotated"] == []


@pytest.mark.parametrize("invalid", ["source", "module", "polygon", "outside", "unknown"])
def test_invalid_or_unsupported_observations_remain_unavailable(scene, invalid):
    s = scene()
    if invalid == "source":
        s.observations[0]["source_rgb_pixel_sha256"] = "different photograph"
    elif invalid == "module":
        s.observations[0]["module_id"] = "missing"
    elif invalid == "polygon":
        s.modules[0]["active_polygon_rgb"] = [[40, 25]] * 4
    elif invalid == "outside":
        s.observations[0]["center_rgb_px"] = [150, 110]
    else:
        s.observations[0]["diagnostic_fault_type"] = "UnrecognizedFault"
    output, report = preview(s)
    assert output.tobytes() == s.base.tobytes()
    assert report["annotated"] == []
    assert [entry["observation_id"] for entry in report["unavailable"]] == ["RFH001"]


def test_missing_center_uses_actual_module_geometry_and_local_coordinates(scene):
    s = scene()
    s.observations[0].pop("center_rgb_px")
    s.observations[0]["local_uv"] = [.25, .75]
    _, report = preview(s)
    np.testing.assert_allclose(report["annotated"][0]["center_rgb_px"], [62.5, 81.25], atol=.01)


@pytest.mark.parametrize("center", [[0, 0], [159, 119]])
def test_markers_at_image_edges_are_clipped_without_moving_the_location(scene, center):
    s = scene()
    s.modules[0]["active_polygon_rgb"] = [[0, 0], [159, 0], [159, 119], [0, 119]]
    s.observations[0]["center_rgb_px"] = center
    output, report = preview(s)
    assert output.size == s.image.size
    assert report["annotated"][0]["center_rgb_px"] == center
    assert np.any(np.asarray(output) != np.asarray(s.base))
