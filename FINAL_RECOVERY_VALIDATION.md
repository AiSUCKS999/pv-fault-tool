# PR #5 final validation — 2026-10-31

Tested code commit: `9d5f82a`. Placement correction: `92540d9`.

## Corrections

- Real Farm placement now derives independent module seeds from the experiment seed and module ID. Same-seed reruns retain positions; different modules receive different relative positions. Existing morphology, class counts and manual observations are preserved.
- Close export crops reuse the exact full-resolution scalar/sensor planes within one acquisition batch. The cache is local to that batch. Cached and uncached Close PNG pixels are identical.

## Clean regression

One complete, frozen-code run: `pytest tests -q --tb=short`.
**134 passed, 1 skipped in 531.07 seconds.** The explicit high-resolution stress case was skipped to avoid duplicating its separately successful run on this code. No failure-only reruns were needed for this integration run.

Coverage includes the actual Streamlit application browser smoke; fullscreen/FIT/FILL; Generated 6×6 mixed faults; RGB/thermal and Far/Close identity; clickable fault boxes and clean views; surface-condition acceptance; severity; random removal; current Review/Export; backup/restore; and historical scenarios. Image14 stress and the actual Streamlit parent smoke are distinct checks: the image14 viewer was exercised in a standalone browser presentation, while the Streamlit smoke verifies the live component-to-recording bridge. No claim is made about physical display acquisition or hosted-server capacity.

## Native image_14 stress

Original photograph: **4742×3159**, unchanged. **12 mixed faults** on distinct valid modules with consistent severity. Far and Close RGB/thermal acquisitions, all fault IDs, clean/annotated switching and fault-box selection passed. 60 browser view switches retained identities. Complete session ZIP: **97,305,526 bytes**; all reference hashes matched. Backup restoration matched the journal.

Measured initial native acquisition: **172.22 seconds**, compared with **198.02 seconds** before caching. Twenty cached acquisitions: **0.637 seconds**. Peak process working set: **1317.25 MiB**. Total stress test: **209.18 seconds**. Profile: two scalar fault-field evaluations and five sensor evaluations across the entire multi-crop acquisition, rather than recomputing the full field for every Close target. Most initial preparation time remains outside those evaluations. One intermediate 180-second test bound expired; the completed instrumented run used a 300-second bound.

Evidence: `artifacts/experiment_recovery/image14_stress.json`, `image14_stress_before.json`, `image14_profile_progress.json`; local browser screenshot `image14_stress_annotated.png` and native presentation `image14_stress.html`. Test logs: `../tmp/final_clean_regression.log`, `../tmp/final_highres_optimized.log`. Large screenshots/HTML are local evidence, excluded from Git.

## Preservation and limits

All **14 prepared sources / 70 asset hashes** match. No original assets, geometry files, masks, dependencies or historical scenario catalog changes. Existing records are retained. No training, merge or deployment occurred.

Ready for **manual acceptance testing**, with limitations: cold preparation for the largest source remains about three minutes and requires substantial memory; hosting capacity must be checked before deployment. Boxes are module-level annotations. Synthetic labels remain unverified physical ground truth. Physical LED-wall/drone validation remains explicitly unverified.
