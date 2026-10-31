"""Original scenario catalog and entry points for automatic experiment recording."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import streamlit as st

DATA_FILE = Path(__file__).with_name("pv_data_collection_cases.csv")
CASE_FIELDS = [
    "Case ID", "Priority", "Source mode", "Scene / layout ID", "Rows",
    "Columns / row", "Topology", "Target row", "Target array col",
    "Module slot (1–4)", "Cell index (1–40)", "Planned scenario",
    "Scenario stage", "FAR observable code", "CLOSE diagnosis code",
    "Severity (1–10)", "Seed", "FAR required?", "CLOSE required?", "Condition ID",
]

@st.cache_data(show_spinner=False)
def _load_cases():
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not set(CASE_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Scenario catalog is missing required columns")
        cases = list(reader)
    ids = [c["Case ID"] for c in cases]
    if len(cases) != 929 or len(set(ids)) != 929:
        raise ValueError("Scenario catalog integrity check failed: expected 929 unique cases")
    if Counter(c["Source mode"] for c in cases) != {"Generated": 789, "Real Farm": 140}:
        raise ValueError("Scenario catalog mode counts must be 789 Generated and 140 Real Farm")
    return cases


def generated_preparation_issue(case, max_rows=6, max_columns=6):
    if case["Topology"] not in {"Full rectangular", "Position sweep", "Severity sweep"}:
        return "This planned topology has no exact preparation in the approved renderer. Keep it for manual review; no substitute layout is opened."
    rows, columns = int(case["Rows"]), int(case["Columns / row"])
    if not (1 <= rows <= max_rows and 1 <= columns <= max_columns):
        return "This planned size exceeds the approved layout limits. No smaller layout will be substituted."
    if not (1 <= int(case["Target row"]) <= rows and 1 <= int(case["Target array col"]) <= columns):
        return "The planned target is outside this layout. Review it before acquisition."
    return None


def active_case_banner(mode):
    from pv_experiment_ui import banner
    banner(mode)


def render_data_collection(mode, on_prepare=None, prepared_sources=None, generated_scenes=None,
                           selected_scene=None, target_options=None, on_restore_simulator=None,
                           on_restore_original=None, simulator_ns=None):
    from pv_experiment_ui import render
    if simulator_ns is None:
        raise ValueError("Actual simulator namespace is required for experiment recording")
    collection_mode = st.radio("Collection mode", ["Scene experiments", "929 controlled cases"], horizontal=True, key="collection_mode_"+mode)
    if collection_mode == "Scene experiments":
        from pv_scene_ui import render_scene_workflow
        render_scene_workflow(simulator_ns,mode)
        return
    render(mode, simulator_ns, prepared_sources=prepared_sources, generated_scenes=generated_scenes,
           selected_scene=selected_scene, on_restore_simulator=on_restore_simulator,
           on_restore_original=on_restore_original)
