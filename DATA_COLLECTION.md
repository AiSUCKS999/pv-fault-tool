# DATA COLLECTION integration

Branch: `codex/pv-data-collection-20261029`

Base: approved `origin/main`, commit `3db093bfeb48e3636c29375d46a64cd48bd5fe9f` (approved Real Farm geometry merge), fetched 2026-10-29. The existing local main checkout and its unrelated changes were left untouched. Work is in a separate Git worktree.

## Use

Choose **DATA COLLECTION** in either Generated PV Farm or Real PV Farm Image. Filter the original plan by priority, layout/photo slot, scenario, status or search; select a case and prepare it. Generated cases use the currently selected terrain. Real Farm cases require an actual prepared photograph from the existing 14-image catalog; RF01–RF14 are experimental slots and are not assumed to match catalog order.

Preparation opens the existing Faults workflow with Current Thermal selected, including when the previous view used the experimental healthy-only renderer. Apply and inspect the planned fault, severity, seed and target using the existing controls. Preparation does not apply faults, clear existing Generated faults, record videos or verify ground truth. Real Farm loading uses the existing source-identity and saved-geometry checks. The actual source identity is copied into the case log.

Return to DATA COLLECTION to log acquisition filenames, observation IDs, clean references, metadata ZIP, device, capture conditions and notes. Complete requires the required FAR/CLOSE views, saved exports and the operator's independent GT-verification checkbox. This is an operator record, not automated evidence validation. Case completion does not bypass the existing geometry-verification gate for verified-label exports.

The 66 irregular-topology and 33 oversized stress cases remain in the catalog and CSV. Their preparation button is disabled because the approved renderer cannot prepare their exact arrangement; no substitute rectangle or clamped size is used. Position and severity sweeps within supported rectangular layouts can be prepared.

Progress is browser-session state. Download **progress JSON** before leaving; restore it to replace that session's collection records. Backups include both modes. Restore validates statuses, flags and completion requirements before replacing anything, preserves text without truncation, and leaves simulator state unchanged. Changing a prepared source invalidates the prior GT confirmation and changes its record to Needs review. Each mode's CSV contains its entire plan and current tracking, regardless of active filters.

## Validation, 2026-10-29

The scenario CSV is byte-for-byte identical to the ZIP member. SHA-256: `03d82a4f958dcbfd9c91864d776fcb72677797caf87fc465689a87c10bcee0a4`.

| Catalog | Generated | Real Farm | Total |
| --- | ---: | ---: | ---: |
| All scenarios | 789 | 140 | 929 |
| P0 Essential | 22 | 20 | 42 |
| P1 Recommended | 86 | 50 | 136 |
| P2 Extended | 681 | 70 | 751 |

All case IDs TC0001–TC0929 are unique. The 14 prepared RGB sources, geometry, PV masks, healthy thermal images and ignore masks passed their saved SHA-256 checks. Every source also loaded through the existing catalog loader with its expected module count and no injected observations.

The bounded integration suite is `py -m pytest tests/test_data_collection.py -q --disable-warnings` (180-second outer process limit; individual Streamlit AppTest runs have a 45-second limit). It covers both routes, P0/P1/P2 and combined filters, empty results, P0 4×6 preparation and return, actual photo selection, unsupported-layout guards, recording forms, completion checks, JSON round-trip and atomic rejection, CSV counts/quoting/long notes, and image accessibility. AppTest operates in process; no web server or background job is started. Test-only capture/GT checkbox values are discarded with their test sessions; they are not experiment results.

Final integration result: **12 passed in 71.29 seconds**. Every existing workflow section also opened without a Streamlit exception in both modes. Preparation correctly switches from the experimental healthy-only renderer to Current Thermal. Syntax compilation and diff whitespace checks passed (accepting the repository's Windows CRLF endings).

## Inherited regression failures: merge/deployment held

Two existing regression scripts fail against both this integration and the unmodified approved base:

1. `scripts/regression_real_farm.py`: its fixture calls `real_farm_dataset_zip` without explicitly confirmed geometry. The existing verified-label export gate rejects it. The same `ValueError` was reproduced using the approved base sources.
2. `scripts/regression_smoke.py`: the rooftop 6×6 SoilingDust matrix fails at row RT4, array R4-A01, module 3, cell RT4-C116. `apply_real_thermal_fault_patch` can return a NumPy array for clipped bounds, while its caller tests the result as a boolean. The identical error was reproduced with approved base sources and that exact target.

Both scripts were run with a 180-second limit; neither launched training or background work. These inherited failures remain unchanged to preserve the approved rendering and verification behavior. The integration has not been merged, pushed or deployed. Resolve these release blockers before considering merge/deployment.

## Diff review

The complete seven-file diff was reviewed. The production changes are the isolated `pv_data_collection.py` module, the original CSV and additive routes/banners in `pv_ui.py` and `pv_real_farm.py`. Tests and this report are included for reproducibility. A CSV-specific `.gitattributes` rule prevents line-ending conversion of the original matrix; the staged Git blob also matches the ZIP member exactly. AST comparison confirmed that the only changed pre-existing definitions are the two workflow menus and their routing functions (`main` and `real_farm_main_impl`). The supplied installer was reviewed and used once; its backups remain outside the branch. No renderer, fault engine, geometry, asset, prepared-image catalog, entrypoint, dependency or existing export implementation is changed.
