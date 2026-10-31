# Experiment recovery acceptance

Baseline: `edb3737` (fetched 2026-10-31). Branch: `codex/experiment-recovery-20261031`.
No merge or deployment authorized for this redesign.

## Implementation sequence

- A: dedicated presentation document, native image payload, fit/fill, synchronized comparison, keyboard and second-window controls; browser evidence.
- B: scene-level generation through existing fault engines; validate targets/counts, preserve manual faults and historical cases.
- C: shared scene identities and Far/Close acquisitions.
- D: clean/annotated fault overlays and physically supported RGB effects.
- E: measured severity audit, explicit random distributions, advisory visibility.
- F: authoritative Review/Export and automatic scene journal with backup/restore.
- G: remove routine explanatory text; keep actionable failures and optional diagnostics.
- H: bounded functional and visual acceptance, exported examples, final diff review.

## Confirmed baseline problems

- Fixed-height iframe/SVG viewers and Streamlit columns constrain the image. The existing focus switch is not browser fullscreen.
- Real camera RGB rendering calls the annotated preview by default. Paired previews halve the available width.
- Far surface handlers roll back otherwise generated faults based on a 1024-pixel visibility heuristic.
- Real random classes cycle without replacement; operator distributions are unavailable.
- Camera observations and legacy observations have separate Review/Export paths.

Results and remaining limitations will be recorded as each stage is verified. Passing unit tests alone is not visual acceptance.

## Stage A results

Browser test passed (Edge): real fullscreen element, 1920×1080, 3840×2160 and 3440×1440; FIT/FILL preserve the source ratio, keyboard modality changes and laptop-to-display control verified. Screenshots in `artifacts/experiment_recovery/`; FIT visibly preserves all 36 arrays. The 3072×2048 source has a 3:2 ratio: ultrawide FIT therefore has expected side letterboxing; FILL intentionally crops vertically. No fabricated source detail.

## Verified checkpoints

- `7f7d060`: presentation browser checks at 1920×1080, 3840×2160, 3440×1440, FIT/FILL, fullscreen and second-window controls passed.
- `95d18e5`: scene tests passed: eight Generated classes on distinct modules across a 6×6 farm; Real Farm eight-class mix plus a preserved manual observation; healthy and rollback checks.
- `dce78df`: nine fault classes × five severity levels × both modes audited. Baseline measurements reproduced severity-dependent shape/reference selection; seed selection now retains severity-7 morphology across the sweep. All tested thermal responses and supported RGB effects increase across 1/3/5/7/10. The renderer models, palettes and geometry remain in use; random seed selection is the deliberate engine correction.
- `1c8d888`: automatic scene journal and current-camera Review/Export connected. Focused backup/restore, tampered archive, failed encoder and no-typing UI checks passed. 66 existing regression tests passed before the integration milestone.
- All 14 prepared photographs remain accessible; 70 original source/geometry/mask/thermal asset hashes match. No asset or production dependency changes.

The independent local patch in `D:/pv-fault-tool` was inspected read-only. It also addresses camera observations versus legacy observations. Its workspace remains untouched; this recovery already has a shared current-scene acquisition path.

## Limitations retained

- Browser screenshots verify pixel layout and controls, not the physical LED wall or drone acquisition quality. Native sources retain their original resolution.
- Boxes are labelled module-level; scalar footprints remain separate metadata, not precise physical ground truth.
- Synthetic intensity is relative signal, not measured temperature. Electrical/heat-only faults do not invent RGB damage.
- Placement is seeded. Existing Generated target-derived morphology does not expose an independent user morphology seed.
- Rooftop 6×6 SoilingDust and unsupported historical topologies remain blocked; no replacement layout is substituted.
- External drone capture is operator-confirmed; encoded PNGs do not prove browser files were saved. Session progress requires a downloaded backup.
- No actual interleaved recording was provided. The offline method cannot establish that recording's pairing or registration accuracy.

## Integration milestone results

The single complete integration run finished with 121 passed and 11 failed. Targeted reruns resolved all 11: seven passed after fixing Close-target confirmation and independent severity sampling, correcting the severity-7 baseline assertion, and repairing caption-cleanup syntax; the last four passed after updating obsolete AppTest image-element lookups to `imgs`. Total: 132 tests accounted for, with no unresolved failures. The complete suite was not repeated.

Actual browser checks cover fullscreen, source aspect ratio, clean/annotated Generated and Real Farm views, clickable fault boxes, Far/Close identity preservation, and the Streamlit component-to-journal selection bridge. The downloaded browser session ZIP contains the exact selected Close fault ID. Screenshots and sample ZIPs remain local under `artifacts/experiment_recovery/`; measurement JSON is committed. No physical LED-wall or drone-video acceptance has been performed.

23 routine technical paragraphs were removed from ordinary controls and collected in the collapsed Help/Diagnostics area. Actionable errors and geometry verification restrictions remain visible. The collection page no longer shows the legacy cell-contribution count as though it were the number of experiment fault instances.

Offline analysis is provided separately in `tools/analyze_interleaved_capture.py`. CLI and synthetic classifier/failed-registration smoke checks passed. No real recording was supplied or analyzed; alignment remains unverified.

### Main changed functions

- `pv_presentation.py`, `pv_presentation.js`, `pv_presentation_component/index.html`: standalone presentation, fullscreen, native source fit/fill, modalities, synchronized comparison, overlays, navigation and operator-state bridge.
- `pv_scene_generate.py`: explicit class/count/severity/placement distribution and transactional scene generation/restoration.
- `pv_scene_frames.py`: authoritative effective observations, native Far/Close acquisitions, module-level geometry and clean presentation payload.
- `pv_scene_log.py`: automatic exact PNG references, fault metadata, CSV/ZIP, atomic encoding and validated backup/scene restore.
- `pv_scene_ui.py`: shared collection/review/export controls and capture confirmation from the current display selection.
- `pv_real_farm_catalog.py`: controlled random batches that preserve manual faults.
- `pv_far_surface.py`: advisory visibility measurements rather than threshold rejection.
- `pv_render_rgb.py`, `pv_render_thermal.py`: severity-independent morphology/reference RNG; existing severity response equations retained.
- `pv_ui.py`, `pv_real_farm.py`, `pv_real_farm_camera.py`, `pv_data_collection.py`: entry points and concise controls; historical cases and individual controls retained.

This is a review branch. Main remains `edb3737`; nothing has been merged or deployed.
