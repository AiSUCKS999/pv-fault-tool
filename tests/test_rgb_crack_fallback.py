"""Empty crack references still produce the established physical RGB fallback."""
import ast
import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_functions(namespace, filename, names):
    path = ROOT / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def renderer():
    ns = {"np": np, "cv2": cv2, "random": random, "Image": Image,
          "ImageDraw": ImageDraw, "ImageFilter": ImageFilter, "BASE_SIZE": (160, 120),
          "canonical_fault_type": lambda name: name,
          "HOTSPOT_FAULTS": {"SingleHotSpot", "MultiHotSpot"},
          "DIODE_FAULTS": {"SingleDiode"}, "BYPASSED_FAULTS": {"SingleByPassed"},
          "CRACKING_FAULTS": {"CellCracking"}, "SOILING_FAULTS": {"SoilingDust"},
          "SURFACE_OBSTRUCTION_FAULTS": {"SurfaceObstruction"},
          "STRING_FAULTS": {"StringOpenCircuit", "StringReversedPolarity"},
          "LEGACY_LINE_FAULTS": set(), "THERMAL_BLOCK_FAULTS": set(),
          "panel_segmentation_preserve_mask": lambda shape, cell: np.zeros(shape[:2], np.float32)}
    load_functions(ns, "pv_render_rgb.py", {
        "clamp", "smooth_unit_field", "seed_for", "paste_color_with_mask", "draw_realistic_fault"})
    load_functions(ns, "pv_render_thermal.py", {"organic_fault_mask", "cell_edge_feather"})
    organic = ns["organic_fault_mask"]
    fallback_calls = []

    def tracked_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return organic(*args, **kwargs)

    ns["organic_fault_mask"] = tracked_fallback
    source = Image.new("RGB", ns["BASE_SIZE"], (33, 61, 89))
    cell = {"id": "test-cell", "x1": 20, "y1": 15, "x2": 140, "y2": 105}

    def render(kind, view="rgb"):
        ns["selected_fault_patch"] = lambda *args: None if kind == "missing" else np.zeros((12, 12, 4), np.uint8)

        def texture(patch, size, seed):
            shape = (size[1], size[0])
            if kind == "missing-mask":
                return None, None
            mask = np.ones(shape, np.float32) if kind == "nonempty" else np.zeros(shape, np.float32)
            return np.zeros(shape, np.float32), mask

        ns["crack_patch_texture_mask"] = texture
        return ns["draw_realistic_fault"](source, cell, "CellCracking", 5, view)

    return SimpleNamespace(ns=ns, source=source, render=render, fallback_calls=fallback_calls)


@pytest.mark.parametrize("kind", ["missing", "missing-mask", "zero"])
def test_unusable_rgb_reference_uses_visible_organic_fallback_without_source_mutation(renderer, kind):
    before = renderer.source.tobytes()
    output = renderer.render(kind)
    assert len(renderer.fallback_calls) == 1
    assert renderer.fallback_calls[0][0][2] == "CellCracking"
    assert output.size == renderer.source.size and output.mode == "RGB"
    assert output.tobytes() != before
    assert renderer.source.tobytes() == before
    changed = np.any(np.asarray(output) != np.asarray(renderer.source), axis=2)
    assert np.count_nonzero(changed[15:105, 20:140]) > 100
    assert not changed[:15].any() and not changed[105:].any()
    assert not changed[:, :20].any() and not changed[:, 140:].any()


def test_zero_rgb_mask_matches_existing_missing_reference_fallback(renderer):
    assert renderer.render("zero").tobytes() == renderer.render("missing").tobytes()


def test_nonempty_reference_preserves_exact_compositor_without_fallback(renderer):
    before = renderer.source.tobytes()
    output = renderer.render("nonempty")
    assert renderer.fallback_calls == []
    assert output.tobytes() != before
    # Fixed nonempty mask: retain the pre-fix compositor's complete pixel output.
    assert hashlib.sha256(output.tobytes()).hexdigest() == "c935f599a6eadbd84f02b724919c21ce58992b4f9e33b5a5172ebe037d20e72b"
    assert renderer.source.tobytes() == before


def test_zero_thermal_mask_retains_existing_no_effect_behavior(renderer):
    before = renderer.source.tobytes()
    output = renderer.render("zero", view="thermal")
    assert renderer.fallback_calls == []
    assert output.tobytes() == before
    assert renderer.source.tobytes() == before
