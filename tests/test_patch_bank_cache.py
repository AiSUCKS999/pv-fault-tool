"""Patch qualification is reused without changing selection or stale pixels."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bank(tmp_path):
    root = tmp_path / "faults"
    root.mkdir()
    ns = {"__file__": str(ROOT / "pv_patch_bank.py"), "__name__": "test_patch_bank_namespace",
          "st": st, "Path": Path, "Image": Image, "np": np, "json": json,
          "FAULT_PATCH_DIR": root, "PATCH_BANK_DIR": tmp_path, "ASSET_DIR": tmp_path,
          "PATCH_FALLBACKS": {}, "canonical_fault_type": lambda name: name,
          "CRACKING_FAULTS": {"CellCracking"}, "HOTSPOT_FAULTS": {"SingleHotSpot", "MultiHotSpot"},
          "PREFERRED_CELL_CRACKING_PATCH_SUFFIXES": {"001", "002"}}
    exec(compile((ROOT / "pv_patch_bank.py").read_text(), str(ROOT / "pv_patch_bank.py"), "exec"), ns)
    ns["alpha_fault_patch_paths"].clear()
    ns["load_fault_patch"].clear()

    def write(name, value, mode="RGBA", fault="SingleHotSpot", size=(5, 4)):
        folder = root / fault
        folder.mkdir(exist_ok=True)
        path = folder / name
        color = (value, 30, 60, 200) if mode == "RGBA" else (value, 30, 60)
        previous = path.stat().st_mtime_ns if path.exists() else None
        Image.new(mode, size, color).save(path)
        if previous is not None:
            changed = max(path.stat().st_mtime_ns, previous + 1_000_000_000)
            os.utime(path, ns=(changed, changed))
        return path
    return SimpleNamespace(ns=ns, write=write, select=ns["selected_fault_patch"], root=root)


def red(image):
    return int(image[0, 0, 0])


def track_opens(monkeypatch):
    original = Image.open
    calls = []
    def opened(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", opened)
    return calls


def test_repeated_seeds_reuse_alpha_qualification_and_decoded_patches(bank, monkeypatch):
    bank.write("a_opaque.png", 10, mode="RGB")
    bank.write("b_alpha.png", 20)
    bank.write("c_alpha.png", 30)
    calls = track_opens(monkeypatch)
    assert red(bank.select("SingleHotSpot", 0)) == 20
    assert len(calls) == 4, "Three headers plus the selected pixel image are read once"
    calls.clear()
    assert red(bank.select("SingleHotSpot", 1)) == 30
    assert len(calls) == 1, "A new seed must not reopen every candidate header"
    calls.clear()
    assert red(bank.select("SingleHotSpot", 0)) == 20
    assert calls == [], "An unchanged already decoded selection should require no image opens"


def test_cracking_selection_preserves_preferred_suffix_and_glass_priority(bank):
    bank.write("a_999.png", 10, fault="CellCracking")
    bank.write("b_001.png", 20, fault="CellCracking")
    bank.write("c_002.png", 30, fault="CellCracking")
    bank.write("d_001.png", 40, mode="RGB", fault="CellCracking")
    assert [red(bank.select("CellCracking", seed)) for seed in range(4)] == [20, 30, 20, 30]
    bank.write("z_pv_v1_glass_breakage_007.png", 70, fault="CellCracking")
    assert [red(bank.select("CellCracking", seed)) for seed in range(3)] == [70, 70, 70]


def test_changed_mode_requalifies_path_and_changed_pixels_invalidate_decode(bank):
    bank.write("a.png", 10, mode="RGB")
    bank.write("b.png", 20)
    assert red(bank.select("SingleHotSpot", 0)) == 20
    bank.write("a.png", 80, mode="RGBA")
    assert red(bank.select("SingleHotSpot", 0)) == 80, "The new alpha-capable file must enter ordered selection"
    bank.write("a.png", 90, mode="RGBA", size=(7, 6))
    changed = bank.select("SingleHotSpot", 0)
    assert red(changed) == 90 and changed.shape[:2] == (6, 7)


def test_added_and_removed_paths_change_selection_without_stale_listing(bank):
    bank.write("b.png", 20)
    assert red(bank.select("SingleHotSpot", 0)) == 20
    added = bank.write("a.png", 10)
    assert red(bank.select("SingleHotSpot", 0)) == 10
    added.unlink()
    assert red(bank.select("SingleHotSpot", 0)) == 20


def test_all_nonalpha_candidates_retain_ordered_fallback(bank, monkeypatch):
    bank.write("a.png", 10, mode="RGB")
    bank.write("b.png", 20, mode="RGB")
    calls = track_opens(monkeypatch)
    assert red(bank.select("SingleHotSpot", 0)) == 10
    assert len(calls) == 3
    calls.clear()
    assert red(bank.select("SingleHotSpot", 1)) == 20
    assert len(calls) == 1
    calls.clear()
    assert red(bank.select("SingleHotSpot", 2)) == 10
    assert not calls


def old_cracking_selection(records, seed):
    """Previous policy: qualify all paths, then choose glass/preferred/all."""
    ordered = sorted(records, key=lambda item: str(item[0]))
    alpha = [item for item in ordered if item[2] in {"RGBA", "LA"}]
    eligible = alpha or ordered
    glass = [item for item in eligible if "pv_v1_glass_breakage" in item[0].stem]
    preferred = [item for item in eligible if item[0].stem.rsplit("_", 1)[-1] in {"001", "002"}]
    chosen = glass or preferred or eligible
    return chosen[seed % len(chosen)][1]


def write_crack_records(bank, entries):
    return [(bank.write(name, value, mode=mode, fault="CellCracking"), value, mode)
            for name, value, mode in entries]


def test_alpha_glass_group_skips_all_lower_priority_image_reads(bank, monkeypatch):
    entries = [
        ("a_001.png", 10, "RGBA"),
        ("b_other_999.png", 20, "RGBA"),
        ("y_pv_v1_glass_breakage_001.png", 30, "RGBA"),
        ("z_pv_v1_glass_breakage_900.png", 40, "RGB"),
    ]
    records = write_crack_records(bank, entries)
    calls = track_opens(monkeypatch)
    assert red(bank.select("CellCracking", 7)) == old_cracking_selection(records, 7) == 30
    assert len(calls) == 3, "Two glass headers and one selected decode are sufficient"
    assert all("pv_v1_glass_breakage" in Path(path).stem for path in calls)


def test_opaque_glass_yields_to_preferred_alpha_without_reading_remaining_group(bank, monkeypatch):
    entries = [
        ("a_remaining_999.png", 10, "RGBA"),
        ("b_preferred_001.png", 20, "RGBA"),
        ("c_preferred_002.png", 30, "RGB"),
        ("d_pv_v1_glass_breakage_001.png", 40, "RGB"),
    ]
    records = write_crack_records(bank, entries)
    calls = track_opens(monkeypatch)
    assert red(bank.select("CellCracking", 2)) == old_cracking_selection(records, 2) == 20
    assert len(calls) == 4, "Glass and preferred headers are inspected once, then the chosen patch is decoded"
    assert str(records[0][0]) not in calls
    assert calls.count(str(records[3][0])) == 1, "A glass path with a preferred suffix must not be rescanned"


def test_remaining_alpha_wins_over_opaque_glass_and_preferred_groups(bank, monkeypatch):
    entries = [
        ("a_remaining_999.png", 10, "RGBA"),
        ("b_remaining_888.png", 20, "RGBA"),
        ("c_preferred_001.png", 30, "RGB"),
        ("d_pv_v1_glass_breakage_900.png", 40, "RGB"),
    ]
    records = write_crack_records(bank, entries)
    calls = track_opens(monkeypatch)
    assert red(bank.select("CellCracking", 1)) == old_cracking_selection(records, 1) == 20
    assert len(calls) == len(records) + 1
    calls.clear()
    assert red(bank.select("CellCracking", 0)) == old_cracking_selection(records, 0) == 10
    assert calls == [str(records[0][0])], "Subsequent seeds reuse all group qualifications"


@pytest.mark.parametrize("entries", [
    [("a_other_999.png", 10, "RGB"), ("b_001.png", 20, "RGB"),
     ("c_pv_v1_glass_breakage_900.png", 30, "RGB"), ("d_pv_v1_glass_breakage_901.png", 40, "RGB")],
    [("a_other_999.png", 10, "RGB"), ("b_001.png", 20, "RGB"), ("c_002.png", 30, "RGB")],
    [("a_other_999.png", 10, "RGB"), ("b_other_888.png", 20, "RGB")],
    [("a_other_999.png", 10, "RGBA"), ("b_001.png", 20, "RGBA"),
     ("c_002.png", 30, "RGB"), ("d_pv_v1_glass_breakage_900.png", 40, "RGBA"),
     ("e_pv_v1_glass_breakage_901.png", 50, "RGBA")],
    [("a_other_999.png", 10, "RGBA"), ("b_001.png", 20, "RGBA"),
     ("c_002.png", 30, "RGBA"), ("d_pv_v1_glass_breakage_900.png", 40, "RGB")],
    [("a_other_999.png", 10, "RGBA"), ("b_other_888.png", 20, "RGBA"),
     ("c_001.png", 30, "RGB"), ("d_pv_v1_glass_breakage_900.png", 40, "RGB")],
], ids=["opaque-glass", "opaque-preferred", "opaque-remaining", "mixed-glass", "mixed-preferred", "mixed-remaining"])
def test_lazy_cracking_selection_matches_old_policy_for_all_seeded_groups(bank, entries):
    records = write_crack_records(bank, entries)
    seeds = [-3, 0, 1, 2, 5, 17, 101]
    assert [red(bank.select("CellCracking", seed)) for seed in seeds] == [
        old_cracking_selection(records, seed) for seed in seeds]
