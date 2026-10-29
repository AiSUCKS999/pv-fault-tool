"""Isolated DATA COLLECTION workflow for the approved PV Fault Tool.

This module does not alter thermal rendering, geometry, fault state, or exports.
It uses a versioned scenario plan and user-confirmed acquisition records.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
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
TRACK_FIELDS = [
    "Status", "FAR captured?", "CLOSE captured?", "Exports saved?",
    "GT verified?", "Recorded observation IDs", "Prepared scene identity",
    "FAR video filename", "CLOSE video filename", "FAR clean reference",
    "CLOSE clean reference", "Dataset / metadata ZIP", "Screen / device",
    "Capture distance / angle", "Notes / exceptions", "Updated UTC",
]
STATUSES = ["Not started", "Prepared", "FAR captured", "CLOSE captured", "Captured", "Needs review", "Complete", "Skipped"]


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


def _records():
    return st.session_state.setdefault("pv_dc_records_v1", {})


def _blank_record():
    blank = {key: "" for key in TRACK_FIELDS}
    blank.update({"Status": "Not started", "FAR captured?": "No",
                  "CLOSE captured?": "No", "Exports saved?": "No", "GT verified?": "No"})
    return blank


def _current_record(case_id):
    return {**_blank_record(), **_records().get(case_id, {})}


def _record_error(case, record):
    if record["Status"] not in STATUSES:
        return "Unknown test status"
    if any(record[key] not in {"Yes", "No"} for key in TRACK_FIELDS[1:5]):
        return "Capture and verification flags must be Yes or No"
    if record["Status"] == "Complete" and (
        record["Exports saved?"] != "Yes" or record["GT verified?"] != "Yes"
        or any(case[f"{view} required?"] == "Yes" and record[f"{view} captured?"] != "Yes"
               for view in ("FAR", "CLOSE"))
    ):
        return "Complete requires all required views captured, exports saved, and independent ground-truth verification."
    return None


def _backup_json():
    return json.dumps({"schema": "pv_dc_records_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                       "records": _records()}, indent=2).encode("utf-8")


def _as_csv(rows, headers):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def _restore_json(upload, allowed_ids):
    data = json.loads(upload.getvalue().decode("utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema") != "pv_dc_records_v1":
        raise ValueError("Not a PV data collection progress backup")
    records = data.get("records")
    if not isinstance(records, dict):
        raise ValueError("Invalid records object")
    recovered = {}
    cases = {c["Case ID"]: c for c in _load_cases()}
    for cid, item in records.items():
        if cid not in allowed_ids:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Invalid record for {cid}")
        record = _blank_record()
        for key, value in item.items():
            if key in TRACK_FIELDS:
                if not isinstance(value, str):
                    raise ValueError(f"Invalid text field for {cid}: {key}")
                record[key] = value
        error = _record_error(cases[cid], record)
        if error:
            raise ValueError(f"{cid}: {error}")
        recovered[cid] = record
    st.session_state["pv_dc_records_v1"] = recovered
    st.session_state["pv_dc_form_revision"] = st.session_state.get("pv_dc_form_revision", 0) + 1
    return len(recovered)


def generated_preparation_issue(case, max_rows=6, max_columns=6):
    if case["Topology"] not in {"Full rectangular", "Position sweep", "Severity sweep"}:
        return "This planned topology has no exact preparation in the approved renderer. Keep it for manual review; no substitute layout is opened."
    rows, columns = int(case["Rows"]), int(case["Columns / row"])
    if not (1 <= rows <= max_rows and 1 <= columns <= max_columns):
        return "This planned size exceeds the approved layout limits. No smaller layout will be substituted."
    if not (1 <= int(case["Target row"]) <= rows and 1 <= int(case["Target array col"]) <= columns):
        return "The planned target is outside this layout. Review it before acquisition."
    return None


def _prepare_case(case, source_id, on_prepare):
    """A button callback runs before widgets, so existing workflow keys can change safely."""
    try:
        identity = on_prepare(case, source_id)
    except (ValueError, OSError, StopIteration) as exc:
        st.session_state["pv_dc_prepare_error"] = str(exc)
        return
    record = _current_record(case["Case ID"])
    if record["Status"] == "Not started":
        record["Status"] = "Prepared"
    elif record["Prepared scene identity"] and record["Prepared scene identity"] != identity:
        record["Status"] = "Needs review"
        record["GT verified?"] = "No"
    record["Prepared scene identity"] = identity
    record["Updated UTC"] = datetime.now(timezone.utc).isoformat()
    _records()[case["Case ID"]] = record


def render_data_collection(mode: str, on_prepare=None, prepared_sources=None):
    """Render one tab. on_prepare(case, selected_source_id=None) is optional."""
    if mode not in {"Generated", "Real Farm"}:
        raise ValueError("Unknown PV data collection mode")
    cases = [c for c in _load_cases() if c["Source mode"] == mode]
    st.title("DATA COLLECTION")
    st.caption("Structured experimental acquisition plan: generated reference vs drone recording, FAR observation and CLOSE diagnosis. No synthetic labels are automatically treated as verified real measurements.")
    if error := st.session_state.pop("pv_dc_prepare_error", None):
        st.error(f"Preparation declined: {error}")

    records = _records()
    complete = sum(_current_record(c["Case ID"])["Status"] == "Complete" for c in cases)
    started = sum(_current_record(c["Case ID"])["Status"] != "Not started" for c in cases)
    essential = [c for c in cases if c["Priority"].startswith("P0")]
    essential_done = sum(_current_record(c["Case ID"])["Status"] == "Complete" for c in essential)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Planned cases", len(cases))
    c2.metric("Started", started)
    c3.metric("Completed", complete)
    c4.metric("Essential complete", f"{essential_done}/{len(essential)}")
    st.progress(complete / len(cases) if cases else 0)

    with st.expander("Progress backup / restore — keep results across sessions", expanded=False):
        st.download_button("Download progress JSON", _backup_json(),
                           file_name="pv_data_collection_progress.json", mime="application/json",
                           key=f"pv_dc_backup_{mode}")
        uploaded = st.file_uploader("Restore a previously downloaded progress JSON", type=["json"], key=f"pv_dc_restore_{mode}")
        if uploaded is not None and st.button("Restore progress (replace current session)", key=f"pv_dc_restore_button_{mode}"):
            try:
                n = _restore_json(uploaded, {c["Case ID"] for c in _load_cases()})
                st.success(f"Restored {n} case records")
                st.rerun()
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"Restore declined: {exc}")
        st.caption("Progress is stored in the current browser session only. Download your JSON after a recording session; uploads restore it after reconnection. The approved app's fault/session state is not overwritten.")

    filt1, filt2, filt3 = st.columns(3)
    priorities = ["All priorities", "P0 – Essential", "P1 – Recommended", "P2 – Extended"]
    priority = filt1.selectbox("Priority", priorities, key=f"pv_dc_priority_{mode}")
    scenes = ["All scenes"] + sorted(set(c["Scene / layout ID"] for c in cases))
    scene = filt2.selectbox("Layout / photo slot", scenes, key=f"pv_dc_scene_{mode}")
    types = ["All scenarios"] + sorted(set(c["Planned scenario"] for c in cases))
    scenario = filt3.selectbox("Fault scenario", types, key=f"pv_dc_type_{mode}")
    f4, f5 = st.columns(2)
    state_filter = f4.selectbox("Completion status", ["All statuses"] + STATUSES, key=f"pv_dc_status_{mode}")
    search = f5.text_input("Search case ID / fault / topology", key=f"pv_dc_search_{mode}")
    filtered = []
    for c in cases:
        status = _current_record(c["Case ID"])["Status"]
        if priority != "All priorities" and not c["Priority"].startswith(priority[:2]):continue
        if scene != "All scenes" and c["Scene / layout ID"] != scene:continue
        if scenario != "All scenarios" and c["Planned scenario"] != scenario:continue
        if state_filter != "All statuses" and status != state_filter:continue
        if search and search.lower() not in " ".join(str(c.get(k, "")) for k in CASE_FIELDS).lower():continue
        filtered.append(c)
    st.caption(f"{len(filtered)} matching cases • display limited to 150 rows for browser responsiveness")
    columns = ["Case ID", "Priority", "Scene / layout ID", "Topology", "Target row", "Target array col", "Planned scenario", "Severity (1–10)", "Seed"]
    shown = [{**{k: c.get(k, "") for k in columns}, "Status": _current_record(c["Case ID"])["Status"]} for c in filtered[:150]]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("Export entire plan + tracking CSV (this mode)",
                       _as_csv([{**c, **_current_record(c["Case ID"])} for c in cases], CASE_FIELDS + TRACK_FIELDS),
                       file_name=f"pv_data_collection_{'generated' if mode == 'Generated' else 'real_farm'}.csv",
                       mime="text/csv", key=f"pv_dc_export_{mode}")
    if not filtered:
        st.info("No cases match these filters. Adjust a filter to select a test.")
        return
    selected_id = st.selectbox("Select case to prepare / log", [c["Case ID"] for c in filtered],
                               key=f"pv_dc_case_{mode}", format_func=lambda cid: next(f"{c['Case ID']} · {c['Planned scenario']} · {c['Scene / layout ID']}" for c in filtered if c["Case ID"] == cid))
    selected = next(c for c in filtered if c["Case ID"] == selected_id)
    st.markdown(f"#### {selected_id}: {selected['Planned scenario']}")
    top1, top2 = st.columns(2)
    with top1:
        st.write("**Acquisition target**")
        st.write(f"Scene: `{selected['Scene / layout ID']}` · topology: `{selected['Topology']}`")
        st.write(f"Row: `{selected['Target row'] or 'select manually'}` · array: `{selected['Target array col'] or 'select manually'}` · module: `{selected['Module slot (1–4)'] or 'select manually'}` · cell: `{selected['Cell index (1–40)'] or 'select manually'}`")
        st.write(f"Seed: `{selected['Seed']}` · severity: `{selected['Severity (1–10)'] or 'healthy'}`")
    with top2:
        st.write("**Expected imaging**")
        st.write(f"FAR: `{selected['FAR observable code']}` · required: `{selected['FAR required?']}`")
        st.write(f"CLOSE: `{selected['CLOSE diagnosis code']}` · required: `{selected['CLOSE required?']}`")
        st.write(f"Condition: `{selected['Condition ID']}`")
    preparation_issue = generated_preparation_issue(selected) if mode == "Generated" else None
    if preparation_issue:
        st.warning(preparation_issue)
    if mode == "Real Farm":
        st.info("RF01–RF14 are workbook photo slots, not guaranteed catalog IDs. Choose and record the actual prepared photograph; do not assume that catalog order matches these slots.")
    st.caption("Preparing a case opens the existing simulator with Current Thermal. It does NOT silently place faults or change approved thermal/geometry rendering. Use the existing Faults controls to apply the specified scenario, then confirm the result.")
    source_id = None
    if mode == "Real Farm" and prepared_sources:
        choices = {item["id"]: item.get("label", item["id"]) for item in prepared_sources}
        source_id = st.selectbox("Actual prepared photograph for this test", list(choices), format_func=lambda x:choices[x], key=f"pv_dc_real_source_{mode}")
    if mode == "Real Farm" and not prepared_sources:
        preparation_issue = "No prepared photographs are available. Restore the approved catalog before preparing a case."
        st.warning(preparation_issue)
    if on_prepare is not None:
        st.button("Prepare this case in the existing simulator", type="primary", key=f"pv_dc_prepare_{mode}",
                  disabled=bool(preparation_issue), on_click=_prepare_case, args=(selected, source_id, on_prepare))

    st.divider()
    st.markdown("#### Log the experiment and matching references")
    current = _current_record(selected_id)
    with st.form(f"pv_dc_form_{mode}_{selected_id}_{st.session_state.get('pv_dc_form_revision', 0)}"):
        status = st.selectbox("Test status", STATUSES, index=STATUSES.index(current["Status"]) if current["Status"] in STATUSES else 0)
        flags = st.columns(4)
        far = flags[0].checkbox("FAR recorded", value=current["FAR captured?"] == "Yes")
        close = flags[1].checkbox("CLOSE recorded", value=current["CLOSE captured?"] == "Yes")
        exports = flags[2].checkbox("Clean exports saved", value=current["Exports saved?"] == "Yes")
        gt = flags[3].checkbox("Ground truth verified", value=current["GT verified?"] == "Yes")
        fields = {}
        for label in TRACK_FIELDS[5:-1]:
            fields[label] = st.text_input(label, value=current.get(label, "")) if label != "Notes / exceptions" else st.text_area(label, value=current.get(label, ""), height=70)
        submitted = st.form_submit_button("Save case record", type="primary")
    if submitted:
        # File names/IDs are a log, not a proof that the expected image or fault is rendered.
        rec = {"Status": status, "FAR captured?": "Yes" if far else "No", "CLOSE captured?": "Yes" if close else "No",
               "Exports saved?": "Yes" if exports else "No", "GT verified?": "Yes" if gt else "No", **fields,
               "Updated UTC": datetime.now(timezone.utc).isoformat()}
        if error := _record_error(selected, rec):
            st.error(error)
        else:
            records[selected_id] = rec
            st.success(f"Saved {selected_id}. Download a progress backup before leaving this session.")
            st.rerun()
