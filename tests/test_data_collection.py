"""Catalog and original prepared-asset regression checks."""
import hashlib
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pv_data_collection as dc

def test_catalog_exact_cases_priorities_and_preparation_limits():
    assert hashlib.sha256(dc.DATA_FILE.read_bytes()).hexdigest() == "03d82a4f958dcbfd9c91864d776fcb72677797caf87fc465689a87c10bcee0a4"
    cases = dc._load_cases()
    assert len(cases) == len({c["Case ID"] for c in cases}) == 929
    assert {c["Case ID"] for c in cases} == {f"TC{i:04}" for i in range(1, 930)}
    assert Counter(c["Source mode"] for c in cases) == {"Generated": 789, "Real Farm": 140}
    assert Counter((c["Source mode"], c["Priority"][:2]) for c in cases) == {
        ("Generated", "P0"): 22, ("Generated", "P1"): 86, ("Generated", "P2"): 681,
        ("Real Farm", "P0"): 20, ("Real Farm", "P1"): 50, ("Real Farm", "P2"): 70,
    }
    generated = [c for c in cases if c["Source mode"] == "Generated"]
    assert sum(dc.generated_preparation_issue(c) is not None for c in generated) == 99
    assert all(dc.generated_preparation_issue(c) is None for c in generated if c["Priority"].startswith("P0"))
    for c in generated:
        if dc.generated_preparation_issue(c) is None:
            for scene in ("grass_open", "agri_field_new", "desert_track", "rooftop_warehouse", "floating_water"):
                asset = ROOT / "assets/solar-renders" / scene / f"{c['Rows']}x{c['Columns / row']}.webp"
                legacy = ROOT / "assets/solar-renders" / f"{c['Rows']}x{c['Columns / row']}.webp"
                assert asset.is_file() or (scene == "grass_open" and legacy.is_file())


def test_all_14_prepared_sources_identity_assets_geometry_and_loader():
    ns = runpy.run_path(str(ROOT / "app.py"), run_name="pv_dc_catalog_test")
    entries = ns["real_farm_catalog_entries"]()
    assert len(entries) == 14
    session = {"pv_dc_records_v1": {"keep": "log"}, "real_farm_section": "DATA COLLECTION"}
    ns["real_farm_catalog_selected"].__globals__["st"] = SimpleNamespace(session_state=session)
    for entry in entries:
        for field, digest in (("rgb", "rgb_file_sha256"), ("project", "project_sha256"), ("pv_mask", "pv_mask_sha256"),
                              ("thermal", "thermal_sha256"), ("ignore_mask", "ignore_mask_sha256")):
            path = ns["real_farm_catalog_asset"](entry[field], entry[digest])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[digest]
        image = ImageOps.exif_transpose(Image.open(ns["real_farm_catalog_asset"](entry["rgb"]))).convert("RGB")
        assert ns["real_farm_source_identity"](image) == entry["rgb_pixel_sha256"]
        session["real_farm_prepared_picker"] = entry["id"]
        ns["real_farm_catalog_selected"]()
        state = session["real_farm_state"]
        assert state["prepared_catalog_id"] == entry["id"]
        assert len(state["modules"]) == entry["module_count"]
        assert not state["camera_observations"]
        assert session["pv_dc_records_v1"] == {"keep": "log"}
