# DATA COLLECTION integration

Branch: `codex/pv-data-collection-20261029`

Base: approved `origin/main`, commit `3db093bfeb48e3636c29375d46a64cd48bd5fe9f` (approved Real Farm geometry merge), fetched 2026-10-29. The existing local main checkout and its unrelated changes were left untouched. Work is in a separate Git worktree.

## Use

Choose **DATA COLLECTION** in either Generated PV Farm or Real PV Farm Image. Filter the original plan by priority, layout/photo slot, scenario, status or search; select a case and prepare it. Generated cases use the currently selected terrain. Real Farm cases require an actual prepared photograph from the existing 14-image catalog; RF01–RF14 are experimental slots and are not assumed to match catalog order.

Generated preparation resolves the selected terrain, exact rows/columns, row, array, module and cell. It creates the requested fault with the existing `fault_record` function, renders it through the existing `module_field_image` path, requires a nonzero image delta, and puts that exact record into the normal fault state. Healthy preparation opens a fault-free working scene. The existing engines, renderers and assets are unchanged. Current Thermal is selected before validation.

The workbook's G/S identifiers specify layouts, not terrain IDs, so the operator explicitly chooses the intended terrain. The existing fixed-target Generated renderer does not accept an independent morphology seed. DATA COLLECTION records the requested seed and this limitation, and marks fault cases **Guided**, even after automatic target/type/severity application. It does not claim a seeded morphology sweep has been implemented.

For Real Farm, the operator explicitly maps the workbook slot to a photograph and chooses currently credible physical module IDs. The adapter retains the current source-matched geometry (including live edits), uses `real_farm_add_module_observation` with the actual severity and seed, and validates a nonzero native-coordinate scalar effect with `real_farm_apply_production_fault`. Multiple hotspots require at least two explicitly selected valid modules. Healthy cases stage no observations. Empty/ambiguous targets open a **Guided** workflow with no applied fault claim.

Real Farm full-scene generation remains an explicit **Thermal → Generate Thermal** operation: a full-scene preparation probe was stopped because it exceeded the bounded test target. Preparation does not start that operation, analysis/recovery, training or background work. Scalar validation is not evidence of a captured or verified Far/Close image. The normal camera view and Diagnostics → unverified analysis downloads are the acquisition exports. Verified-label dataset ZIP export remains subject to the unchanged geometry-confirmation gate.

For a Close diagnosis case, use **Preparation phase → Far observation**, inspect/record FAR, then return and choose **Close diagnosis** and prepare again. The existing engine replaces the hotspot with the diagnosis at that target. Both phase receipts are retained in progress JSON/CSV, including actual target, fault, severity, requested seed, seed behavior and validation evidence.

Return to DATA COLLECTION to log acquisition filenames, observation IDs, clean references, metadata ZIP, device, capture conditions and notes. Complete requires the required FAR/CLOSE views, saved exports and the operator's independent GT-verification checkbox. Blocked/unsupported setups cannot be marked Complete. Preparation always clears GT confirmation; re-preparing a completed case changes it to Needs review. This is an operator record, not automated evidence validation. Case completion does not bypass the existing geometry-verification gate for verified-label exports.

The 66 irregular-topology and 33 oversized stress cases remain in the catalog and CSV. Their preparation button is disabled because the approved renderer cannot prepare their exact arrangement; no substitute rectangle or clamped size is used. Position and severity sweeps within supported rectangular layouts can be prepared, subject to the independent-seed limitation. When Warehouse Rooftop is selected, 40 otherwise supported G06x06 SoilingDust cases are conservatively blocked because the known clean-main failure occurs within that layout/fault combination (two additional 6×6 irregular cases are already among the unsupported layouts). The same rectangular case may work on a different explicitly selected terrain; no automatic substitution occurs. Other engine errors or zero-delta results roll back preparation and mark the record Blocked.

Progress is browser-session state. Download **progress JSON** before leaving; restore it to replace that session's collection records. Backups include both modes. Restore validates statuses, flags and completion requirements before replacing anything, preserves text without truncation, and leaves simulator state unchanged. Each mode's CSV contains its entire plan and current tracking, regardless of active filters.

Selecting/filtering a case does not apply or erase faults. Clicking Prepare intentionally opens an isolated case and retains the full previous simulator context, including source, query/navigation, generated faults and Real Farm geometry. **Restore simulator from before the last preparation** restores it without replacing acquisition logs. **Restore simulator from before DATA COLLECTION** separately retains the original simulator across multiple preparations. These two browser-session snapshots are not disk backups or unlimited simulator history. A failed preparation restores the previous context automatically.

## Validation, 2026-10-29

The scenario CSV is byte-for-byte identical to the ZIP member. SHA-256: `03d82a4f958dcbfd9c91864d776fcb72677797caf87fc465689a87c10bcee0a4`.

| Catalog | Generated | Real Farm | Total |
| --- | ---: | ---: | ---: |
| All scenarios | 789 | 140 | 929 |
| P0 Essential | 22 | 20 | 42 |
| P1 Recommended | 86 | 50 | 136 |
| P2 Extended | 681 | 70 | 751 |

All case IDs TC0001–TC0929 are unique. The 14 prepared RGB sources, geometry, PV masks, healthy thermal images and ignore masks passed their saved SHA-256 checks. Every source also loaded through the existing catalog loader with its expected module count and no injected observations.

The bounded suites are `py -m pytest tests/test_data_collection.py -q --disable-warnings` and `py -m pytest tests/test_collection_acceptance.py -q --disable-warnings` (150-second outer limit per suite; individual Streamlit AppTest runs have a 45-second limit). They cover both routes, P0/P1/P2 and combined filters, empty results, P0 4×6 preparation and return, actual photo/module selection, unsupported-layout guards, recording forms, completion checks, JSON round-trip and atomic rejection, CSV counts/quoting/long notes, existing workflow navigation, and all 14 sources. AppTest operates in process; no web server or background job is started. Test-only capture/GT checkbox values are discarded with their test sessions; they are not experiment results.

Representative actual preparation checks include TC0001 healthy; TC0002/TC0387 hotspots; TC0003/TC0388 multiple hotspots; TC0092 shading; TC0005 obstruction; TC0094 soiling; TC0007 cracking; TC0173 diode; TC0394 bypass; TC0010 open circuit; and TC0011 reversed polarity. These span 1×1, 2×3, 3×4 and 6×6, different rows/arrays/modules/cells, all five terrains, and severities 3/6/9. Tests inspect actual state/metadata and nonzero rendered changes, not just catalog labels. Real Farm checks use image_01 and image_02, healthy, single/multiple hotspots, Close diode diagnosis, deterministic seeded placement, valid/invalid/manual module choices, unchanged geometry, metadata exports, and rollback.

Final verification: **36 distinct tests passed across the bounded runs**: 20 preparation/engine/baseline checks in 69.40 s; 14 UI/progress/export/regression checks in 106.88 s; the focused support-column check in 3.55 s; and an added original-simulator preservation check (run together with the existing preparation/navigation check: 2 passed in 19.47 s). The final focused checks verify all 40 affected rectangular rooftop cases are visibly blocked and the original simulator survives multiple preparations. Syntax compilation and CRLF-aware whitespace checks also pass. These are representative tests, not 929 completed acquisitions or full scenario coverage. Far/Close Real Farm full-scene capture remains a guided operator check; no final full-scene image quality claim is made.

## Known baseline failures

Two existing regression scripts fail against both this integration and the unmodified approved base:

1. `scripts/regression_real_farm.py`: its fixture calls `real_farm_dataset_zip` without explicitly confirmed geometry. The existing verified-label export gate rejects it. The same `ValueError` was reproduced using the approved base sources.
2. `scripts/regression_smoke.py`: the rooftop 6×6 SoilingDust matrix fails at row RT4, array R4-A01, module 3, cell RT4-C116. `apply_real_thermal_fault_patch` can return a NumPy array for clipped bounds, while its caller tests the result as a boolean. The identical error was reproduced with approved base sources and that exact target.

The acceptance tests load every approved-main section directly from the clean Git commit above, with unchanged repository assets. They reproduce the Real Farm regression's exact export rejection and the rooftop matrix's exact failing target. No working-copy application code is substituted in those baseline namespaces. These inherited failures remain unchanged. The export gate is correctly retained; it blocks verified-label export until independent user review. The rooftop combination is held as described above. Per the acceptance request, these inherited issues are documented rather than repaired or treated as new integration regressions. Merge/deployment still requires the user's decision after the acceptance report.

## Diff review

The complete diff is limited to the isolated collection UI and preparation adapters, original CSV, additive routes/banners in `pv_ui.py` and `pv_real_farm.py`, tests and this report. A CSV-specific `.gitattributes` rule prevents line-ending conversion of the original matrix; its Git blob matches the ZIP member exactly. AST comparison checks that the only changed pre-existing definitions are the two workflow menus and their routing functions (`main` and `real_farm_main_impl`). The supplied installer was reviewed and used once; its backups remain outside the branch. No renderer, fault engine, geometry, asset, prepared-image catalog, entrypoint, dependency or existing export implementation is changed.
