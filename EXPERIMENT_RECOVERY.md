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
