# Automatic PV experiment collection

Branch: `codex/pv-auto-experiment-recording-20261029`.
Base: approved `origin/main` at `baf1f8f` (the merged DATA COLLECTION integration). This branch does not merge or deploy itself.

## Normal workflow

Select a case, then click **COLLECT THIS TEST AUTOMATICALLY**. The app prepares FAR, exports its clean simulated thermal reference, prepares/exports the required CLOSE diagnosis, reads the actual simulator parameters and saves the experiment journal. It finishes as **Simulator collection complete** and offers **NEXT SCENARIO**. No recording, filenames, observation IDs, coordinates, seed explanations, dates, status selection, checkbox checklist or Save Form is required.

**DOWNLOAD LOG + ALL REFERENCE IMAGES** exports a ZIP with the complete session's JSON and CSV journal, PNG references and a SHA-256 reference manifest. Each image has a unique filename containing its case, view, actual scene identity, acquisition revision and experiment/reference IDs. FAR and CLOSE stay under one experiment ID. Repeating a finished case creates a new experiment; switching cases can resume the same unfinished experiment.

The app collects simulator evidence. It never automatically claims a drone recorded a video, a browser saved a file locally, synthetic labels are verified physical ground truth, or YOLO detected the fault. Optional notes, video metadata and explicit operator evidence confirmations are hidden under Advanced. No drone recording or external evidence is required for simulator completion.

## Actual state and evidence

Actual fields are read from the live scene, effective fault records, selected cells/modules and source images, not populated from planned values. Preparation and export receipts retain the original plan alongside actual source identity, layout, target, fault type, severity, supported seed behavior, native observation IDs, timestamps and rendered changes. Generated engine records have no native observation ID: their automatically generated log identities are explicitly labelled collection-derived. Real Farm preserves its native RFH observation IDs.

The collector uses the existing fault and thermal/image export functions. It requires an actual PNG, verifies its integrity and records its hash. A requested fault with no rendered change, invalid geometry/source, failed preparation or failed exporter is blocked and is not marked complete. RGB and thermal references never become proof of a physical diagnosis. Changes between FAR and CLOSE are retained; a changed source cannot be silently paired with an earlier stage. Corrections through the existing simulator are read back from its effective state.

Generated fixed-target morphology does not accept an independent case seed. The requested seed remains in the original plan; the actual unsupported behavior is recorded automatically. Real Farm observation placement uses the actual stored seed and physically supported module selection. Missing credible targets block automatic collection rather than inventing a module.

## Natural-color Real Farm RGB fault view

In the existing Real Farm simulator choose **Image view → RGB faults**. The preview uses the original RGB photograph with the existing RGB effect function projected onto the selected module. It does not apply a thermal LUT or orange heat-map palette. **Original RGB** remains available for comparison, and **Close View** now crops the selected location and expands its display correctly.

The preview can work from prepared geometry without generating a thermal frame first. Existing RGB surface obstruction, dust/soiling and cracking effects are supported. Heat-only hotspots and electrical diagnoses without an established daylight surface appearance retain the original photo and explain that the diagnosis remains available in thermal view; a hotspot no longer invents RGB scorching. RGB changes are clipped to the existing active PV support, including its holes. Original source pixels, geometry, masks and approved RGB/thermal effect functions are unchanged. This preview is labelled simulated, not verified physical damage. Current camera downloads use the existing PNG exporter.

## Visible conditions in Far View

Both Generated and Real Farm **Faults → Far View** now offer **Visible Far conditions · RGB + thermal**. Choose a module, surface obstruction or dust/soiling, then click **GENERATE VISIBLE FAR CONDITION**. This prepares a visible surface condition with the existing thermal response on the same actual target. Changing views keeps the condition; Close View retains it for inspection. Pure thermal hotspot placement remains available separately. Small cracks and electrical-only diagnoses are not advertised as visible Far surface conditions.

Generated conditions reuse the exact approved electrical cell polygons: broad obstruction covers 12 adjacent cell targets and soiling covers all 40 cells of the selected module. Each cell contribution is retained as a real engine record under a surface group ID, including in actual-state collection logs. RGB uses the existing RGB effect function projected onto those same polygons; thermal uses the unchanged existing Generated pipeline. Geometry is resolved from the clean source, never from fault-altered RGB. The existing dataset exporter retains clean `source_rgb.png` alongside the faulted current scene, actual records and original support mask.

Real Farm uses the existing native observation and diagnostic fault engine, with no detector/refit needed for a prepared physical module. Provisional/unresolved points cannot host a new visible surface condition. Both modalities are rendered before committing observations; existing occupied modules are preserved. Cold Far View can prepare both images directly. No successful preparation is saved if either image lacks sufficient Far-scale change.

The display check tests a maximum 1024-pixel-wide Far image and requires at least 64 pixels with channel contrast of 8/255 in each modality. This is a reproducible simulation visibility check, not proof of human detection, drone capture or physical ground truth. Tiny or weak effects are rejected with a request for a closer view or larger visible module; they are not enlarged or painted with a heat color in RGB. The rooftop 6×6 soiling restriction remains blocked. Existing 929-case plans are unchanged rather than relabelled as visible surface experiments.

## Progress and simulator recovery

Progress and references are session data, not permanent storage. Download the ZIP before leaving. JSON backup is metadata-only; restore a ZIP to recover PNG bytes as well. Restore is validated before replacing collection data and never replaces current scene/fault/geometry state. Original v1 manual progress imports as a legacy record for review, without inventing actual evidence or verified ground truth.

The original simulator context remains recoverable across multiple preparations. A separate one-step snapshot restores the immediately previous context. Collection logs survive both restores; the live experiment binding is cleared so a restored scene is not mistaken for another case.

Optional drone-video selection uses a local browser File/video metadata picker. Only name, MIME, size, modification date, duration and dimensions are sent to the app. Video contents are not uploaded or loaded into server memory. Selecting a video does not confirm recording or detection.

## Preserved catalog and restrictions

The original CSV is unchanged, SHA-256 `03d82a4f958dcbfd9c91864d776fcb72677797caf87fc465689a87c10bcee0a4`.

| Cases | Generated | Real Farm | Total |
| --- | ---: | ---: | ---: |
| All | 789 | 140 | 929 |
| P0 Essential | 22 | 20 | 42 |
| P1 Recommended | 86 | 50 | 136 |
| P2 Extended | 681 | 70 | 751 |

All 14 prepared photographs and their source-matched geometry, masks and saved thermal assets remain intact. The 66 irregular and 33 oversized layouts remain unsupported. The existing hold on 40 otherwise supported rectangular rooftop 6×6 SoilingDust cases remains in force. No substitute layout or clamped target is generated. The known clean-main legacy regression failures remain documented in `tests/test_collection_acceptance.py`: geometry confirmation is required for verified-label export, and the rooftop legacy fault matrix has the inherited NumPy truth-value error. No engine repair is included.

## Validation, 2026-10-30

Bounded test groups cover the automatic one-click workflow in both modes without text entry, actual engine parameters and exact clean PNG bytes, same-experiment FAR/CLOSE association, no invented drone/GT/YOLO confirmations, failure handling, stale-state rejection, case resumption, atomic JSON/ZIP restore, original simulator recovery, all catalog priorities and 14 prepared sources, natural RGB effect containment, RGB Close View without thermal generation, and thermal pixels matching approved main.

Results: **79 Python tests passed**: 43 catalog, baseline acceptance and journal tests; 14 automatic workflow tests; 11 natural RGB tests; and 11 Far surface tests. The Far tests cover small and 6×6 layouts, obstruction/soiling at different severities, prepared Real Farm targets, cold RGB and thermal UI entry, no-typing creation, view switching, rejected weak effects, original-source dataset export, actual-state grouping, containment and rooftop blocking. The browser component's Node test also passed using a virtual 20 GiB video File: no contents are read or uploaded. JavaScript syntax, Python compilation and Git whitespace checks passed. Runtime dependencies are unchanged. Tests run in foreground with bounded subprocess timeouts; no training or background jobs are started.

This is representative simulation acceptance, not 929 completed physical experiments, certified ground truth or an end-to-end drone/YOLO evaluation. The complete diff was reviewed, including an AST comparison confirming only the routing functions and camera UI changed in approved simulator modules. Merge and deployment require user approval.
