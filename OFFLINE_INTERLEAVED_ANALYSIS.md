# Existing interleaved recording

`tools/analyze_interleaved_capture.py` reads an existing file sequentially, with a default limit of 1,800 frames. It creates frame classifications, contiguous RGB/thermal intervals, uncertain transitions, candidate neighboring pairs, motion estimates, and edge-registration diagnostics. It keeps only the current frames in memory and writes at most 30 reduced candidate pairs. It never generates a flashing display or modifies the original video.

Supply the matching clean simulator RGB and thermal references. Run locally with `--rgb-reference`, `--thermal-reference`, `--out`, and optionally `--max-frames`. No upload, training, or new dependency is required.

The classifier uses reference color similarity. A single reference histogram cannot prove that a rolling-shutter frame is clean: manually inspect representative frames and every extracted transition. Registration uses an affine fit on edges, reports failed/weak fits, and leaves `alignment_verified` false even when its internal checks pass. Confirm correspondence on independent module corners distributed across the frame. Review motion, blur, occlusion, repeated panel patterns, exposure changes and within-frame display transitions before using any pair for evaluation.

Direct aligned simulator exports or sequential acquisition are preferable for new captures. The existing drone recording was not supplied with this task, so no claims about its usable pair count or alignment can yet be made.
