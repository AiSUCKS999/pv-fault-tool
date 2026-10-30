"""Button-driven acquisition UI. Normal completion has no text fields or save form."""
from pathlib import Path
import json
import zipfile

import streamlit as st
import streamlit.components.v1 as components

from pv_experiment_log import Journal, utc
from pv_experiment_capture import actual_state, collect_automatically, ensure_reference, prepare

_video_picker = components.declare_component("pv_drone_video_metadata", path=str(Path(__file__).with_name("pv_video_metadata")))


def banner(mode):
    exp = Journal(st.session_state).active(mode)
    if exp:
        st.info(f"DATA COLLECTION · {exp['case_id']} · {exp['status']}. Return to DATA COLLECTION for your automatic experiment log and reference images.")


def action(ns, mode, eid, operation, view=None):
    journal = Journal(st.session_state)
    exp = journal.get(eid)
    try:
        if operation in {"save", "confirm"} and ns["st"].session_state.get("pv_dc_live_experiment") != eid:
            raise ValueError("Prepare this test's stage before saving or confirming; the simulator is on another context")
        if operation == "save":
            ensure_reference(ns, journal, exp, view, force=True)
            exp["status"] = f"{view} reference ready in this session"
        elif operation == "confirm":
            journal.confirm_capture(exp, view, actual_state(ns, mode))
        elif operation == "download":
            journal.request_download(exp, view)
        elif operation == "file_saved":
            journal.confirm_file_saved(exp, view)
        elif operation == "archived_capture":
            journal.confirm_archived_capture(exp, view)
        elif operation == "exception":
            exp["stages"][view]["exception_accepted_utc"] = utc()
            exp["stages"][view]["exception_accepted_fingerprint"] = exp["stages"][view]["preparations"][-1]["state_fingerprint"]
            journal.event(exp, "actual_setup_exception_accepted_by_operator", view=view)
        elif operation == "notes":
            exp["notes"] = st.session_state.get(f"pv_dc_notes_{eid}", "")
            journal.event(exp, "optional_notes_updated")
        else:
            raise ValueError("Unknown experiment action")
    except Exception as exc:
        journal.fail(exp, operation, exc)
        st.session_state[f"pv_dc_notice_{mode}"] = str(exc)


def prepare_action(ns, case, options, view):
    journal = Journal(st.session_state)
    exp = journal.active(case["Source mode"], case["Case ID"])
    prepare(ns, journal, case, options, view, exp)


def collect_action(ns, case, options):
    journal = Journal(st.session_state)
    exp = collect_automatically(ns, journal, case, options)
    if exp.get("finished_utc"):
        st.session_state[f"pv_dc_finished_{case['Source mode']}"] = f"{case['Case ID']} collected automatically. Your log and reference images are ready to download."


def next_action(mode, next_id):
    if next_id:
        st.session_state[f"pv_dc_case_{mode}"] = next_id
        Journal(st.session_state).data.setdefault("selected_case_by_mode", {})[mode] = next_id


def finish_action(mode, eid, next_id):
    journal = Journal(st.session_state)
    exp = journal.get(eid)
    try:
        journal.finish(exp)
        if next_id:
            st.session_state[f"pv_dc_case_{mode}"] = next_id
            journal.data.setdefault("selected_case_by_mode", {})[mode] = next_id
        st.session_state[f"pv_dc_finished_{mode}"] = f"{exp['case_id']} saved. {next_id or 'No further matching scenario'} is ready."
    except (ValueError, KeyError) as exc:
        journal.fail(exp, "finish", exc)


def inspect_action(ns, mode, view):
    if mode == "Generated":
        ns["st"].query_params["section"] = "Faults"
    else:
        ns["st"].session_state["real_farm_section"] = "Faults"
        ns["st"].session_state["real_farm_camera_distance"] = "Close View" if view == "CLOSE" else "Far View"


def restore_simulator(callback):
    callback()
    st.session_state.pop("pv_dc_live_experiment", None)


def restore_progress(mode):
    from pv_data_collection import _load_cases
    uploaded = st.session_state.get(f"pv_dc_restore_{mode}")
    if uploaded is None:
        return
    try:
        journal = Journal(st.session_state)
        total = journal.restore(uploaded.getvalue(), _load_cases(), uploaded.name.lower().endswith(".zip"))
        exp = journal.active(mode)
        selected = journal.data.get("selected_case_by_mode", {}).get(mode)
        if selected in {c["Case ID"] for c in _load_cases() if c["Source mode"] == mode}:
            st.session_state[f"pv_dc_case_{mode}"] = selected
        elif exp:
            st.session_state[f"pv_dc_case_{mode}"] = exp["case_id"]
        st.session_state[f"pv_dc_finished_{mode}"] = f"Restored {total} experiment records. Simulator state was left unchanged."
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile, UnicodeError) as exc:
        st.session_state[f"pv_dc_notice_{mode}"] = f"Restore declined: {exc}"


def render(mode, ns, generated_scenes=None, selected_scene=None, prepared_sources=None,
           on_restore_simulator=None, on_restore_original=None, **unused):
    from pv_data_collection import _load_cases
    from pv_data_collection_prepare import generated_block
    journal = Journal(st.session_state)
    if not journal.data["experiments"] and st.session_state.get("pv_dc_records_v1"):
        journal.restore(json.dumps({"schema": "pv_dc_records_v1", "records": st.session_state["pv_dc_records_v1"]}).encode(), _load_cases())
    cases = [c for c in _load_cases() if c["Source mode"] == mode]
    st.title("DATA COLLECTION")
    st.caption("Select a test. The app prepares it and collects its simulator images and data automatically.")
    if message := st.session_state.pop(f"pv_dc_finished_{mode}", None):
        st.success(message)
    if message := st.session_state.pop(f"pv_dc_notice_{mode}", None):
        st.error(message)
    done = {e["case_id"] for e in journal.data["experiments"].values() if e["mode"] == mode and e.get("finished_utc")}
    essentials = [c for c in cases if c["Priority"].startswith("P0")]
    left, right = st.columns(2)
    left.metric("Simulator tests collected", f"{len(done)}/{len(cases)}")
    right.metric("P0 Essential", f"{sum(c['Case ID'] in done for c in essentials)}/{len(essentials)}")
    priority = st.selectbox("Priority", ["All priorities", "P0 – Essential", "P1 – Recommended", "P2 – Extended"], key=f"pv_dc_priority_{mode}")
    filtered = [c for c in cases if priority == "All priorities" or c["Priority"].startswith(priority[:2])]
    lookup = {c["Case ID"]: c for c in filtered}
    if st.session_state.get(f"pv_dc_case_{mode}") not in lookup:
        st.session_state[f"pv_dc_case_{mode}"] = next(iter(lookup))
    cid = st.selectbox("Select test case", list(lookup), key=f"pv_dc_case_{mode}",
        format_func=lambda x: f"{x} · {lookup[x]['Priority'][:2]} · {lookup[x]['Planned scenario']} · {lookup[x]['Scene / layout ID']}")
    case = lookup[cid]
    journal.data.setdefault("selected_case_by_mode", {})[mode] = cid
    options = {}
    blocked = None
    if mode == "Generated":
        scenes = {item["id"]: item["name"] for item in generated_scenes}
        options["scene"] = st.selectbox("Scene", list(scenes), index=list(scenes).index(selected_scene),
            format_func=lambda x: scenes[x], key="pv_dc_generated_scene")
        blocked = generated_block(case, options["scene"])
    else:
        choices = {item["id"]: item["label"] for item in prepared_sources or []}
        slot = case["Scene / layout ID"]
        saved_source = journal.data.setdefault("photo_slots", {}).get(slot)
        options["source"] = st.selectbox("Prepared photograph", list(choices),
            index=list(choices).index(saved_source) if saved_source in choices else None,
            format_func=lambda x: choices[x], placeholder="Choose the actual photograph for this photo slot",
            key=f"pv_dc_photo_{slot}")
        if not options["source"]:
            blocked = "Choose a prepared photograph. Workbook photo slots are not assumed to match catalog order."
        else:
            journal.data["photo_slots"][slot] = options["source"]
    exp = journal.active(mode, cid)
    if mode == "Generated" and blocked and exp is None:
        exp = journal.start(case)
        journal.fail(exp, "unsupported/blocked setup", blocked)
    st.subheader(exp["status"] if exp else "Ready to collect")
    if blocked:
        st.warning(blocked)
    if exp and exp["blocked_reason"]:
        st.error(exp["blocked_reason"])
    st.button("COLLECT THIS TEST AUTOMATICALLY", type="primary", use_container_width=True, disabled=bool(blocked),
              key=f"pv_dc_collect_{mode}", on_click=collect_action, args=(ns, case, options))
    if exp:
        far_ready = journal.reference_complete(exp, "FAR")
        close_ready = journal.reference_complete(exp, "CLOSE")
        required_close = case["CLOSE required?"] == "Yes"
        st.progress((int(far_ready) + int(close_ready and required_close)) / (2 if required_close else 1))
        st.caption(f"FAR {'collected' if far_ready else 'pending'} · CLOSE {'collected' if close_ready else 'pending' if required_close else 'not required'}")
        columns = st.columns(2 if journal.current_reference(exp, "CLOSE") else 1)
        for column, view in zip(columns, ("FAR", "CLOSE")):
            ref = journal.current_reference(exp, view)
            if ref and ref["reference_id"] in journal.blobs:
                with column:
                    st.image(journal.blobs[ref["reference_id"]], caption=f"{view} clean thermal reference", use_container_width=True)
                    st.download_button(f"Download {view} image", journal.blobs[ref["reference_id"]], file_name=ref["filename"], mime="image/png",
                        key=f"pv_dc_download_{ref['reference_id']}", on_click=action,
                        args=(ns, mode, exp["experiment_id"], "download", view))
            elif ref:
                st.warning(f"{view} metadata is restored, but the image is missing. Restore the reference ZIP to recover it.")
        next_id = next((c["Case ID"] for c in filtered[list(lookup).index(cid) + 1:]
            if c["Case ID"] not in done and (mode != "Generated" or not generated_block(c, options["scene"]))), None)
        st.button("NEXT SCENARIO", use_container_width=True, disabled=not exp.get("finished_utc") or next_id is None,
                  key=f"pv_dc_next_{mode}", on_click=next_action, args=(mode, next_id))
        with st.expander("Advanced · optional notes, evidence and corrections", expanded=False):
            st.text_area("Optional notes", value=exp["notes"], key=f"pv_dc_notes_{exp['experiment_id']}",
                on_change=action, args=(ns, mode, exp["experiment_id"], "notes"))
            st.caption("Simulator collection does not claim that a drone video exists, a file was saved locally, physical ground truth was verified, or YOLO detected anything.")
            view = st.selectbox("Optional video stage", ["FAR", "CLOSE"], key=f"pv_dc_video_stage_{exp['experiment_id']}")
            metadata = _video_picker(experiment_id=exp["experiment_id"], view=view,
                key=f"pv_dc_video_{exp['experiment_id']}_{view}", default=None)
            st.caption("Optional: choose an existing drone video. Only filename and metadata are read in your browser; the video is not uploaded.")
            if metadata:
                try:
                    journal.attach_video(exp, view, metadata)
                except ValueError as exc:
                    st.error(str(exc))
            ref = journal.current_reference(exp, view)
            if ref:
                st.button(f"I CONFIRM AN EXTERNAL {view} RECORDING MATCHES THIS REFERENCE", key=f"pv_dc_optional_capture_{exp['experiment_id']}_{view}",
                    on_click=action, args=(ns, mode, exp["experiment_id"], "archived_capture", view))
                st.button(f"I CONFIRM THE {view} REFERENCE FILE WAS SAVED", disabled=bool(ref["file_saved_confirmed_utc"]),
                    key=f"pv_dc_file_saved_{ref['reference_id']}", on_click=action,
                    args=(ns, mode, exp["experiment_id"], "file_saved", view))
            st.button("Inspect current simulator view", key=f"pv_dc_inspect_{mode}",
                on_click=inspect_action, args=(ns, mode, exp["active_stage"]))
            st.json(exp)
    st.download_button("DOWNLOAD LOG + ALL REFERENCE IMAGES", journal.bundle(),
        file_name="pv_experiments_and_references.zip", mime="application/zip", key=f"pv_dc_bundle_{mode}",
        type="primary", use_container_width=True, disabled=not journal.data["experiments"])
    st.caption("Images and progress stay in this session until downloaded. Simulator data is collected automatically; external recording and physical verification are not claimed.")
    with st.expander("Progress backup / restore", expanded=False):
        st.warning("Session progress is not permanent. The ZIP includes images; JSON contains metadata only.")
        st.download_button("Download progress JSON", journal.backup(), file_name="pv_experiment_progress.json", mime="application/json", key=f"pv_dc_backup_{mode}")
        st.download_button("Download experiment CSV", journal.csv(), file_name="pv_experiment_log.csv", mime="text/csv", key=f"pv_dc_csv_{mode}")
        st.file_uploader("Restore progress JSON or reference ZIP", type=["json", "zip"], key=f"pv_dc_restore_{mode}",
            on_change=restore_progress, args=(mode,))
    with st.expander("Recover previous simulator state", expanded=False):
        if on_restore_simulator and "pv_dc_before_preparation" in st.session_state:
            st.button("Restore simulator from before the last preparation", on_click=restore_simulator, args=(on_restore_simulator,), key=f"pv_dc_restore_simulator_{mode}")
        if on_restore_original and "pv_dc_original_simulator" in st.session_state:
            st.button("Restore simulator from before DATA COLLECTION", on_click=restore_simulator, args=(on_restore_original,), key=f"pv_dc_restore_original_{mode}")
