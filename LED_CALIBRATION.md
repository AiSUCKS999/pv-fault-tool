# LED filming calibration

Use the existing presentation controls on the laptop. Choose **LED Calibration Mode**, then use the existing fullscreen button or separate display window. Those output surfaces contain only the pattern; controls and status remain outside them.

- **Gray:** constant RGB code 128 across the entire display viewport; no flashing or automatic alternation. This is a digital code, not a calibrated physical luminance measurement.
- **PV detail:** original RGB from the selected existing Close frame, or original Far RGB when no fault target exists. No invented detail.
- **Fixed actual fault:** pins the selected fault's existing Close image and modality. Original RGB switches to faulted RGB for this pattern. Exit and reselect the pattern to pin another target or modality. A healthy scene cannot supply a fixed-fault pattern.
- **FIT/FILL and Zoom:** retain the image aspect ratio. Zoom is presentation magnification/cropping only; it does not increase source resolution or change exported research pixels.
- **CLEAN FILMING:** hides boxes and labels. Use fullscreen or the separate window to remove operator controls from the filmed output. Calibration always suppresses annotations, independently of the box toggle.

Calibration settings are optional, session-scoped experiment metadata when an active matching experiment exists. Settings/events are included in the session ZIP and restored with it. Recording confirmations include the provided presentation settings and latest calibration settings. Exit calibration before confirming an experiment recording. No reference images, geometry, labels, fault physics or original photographs are modified.

Physical LED pixels, black gaps and moiré cannot be eliminated by software alone. Physical wall/camera validation remains unverified. No blur, synthetic detail, fault exaggeration or automatic flashing has been introduced.

Validation: `tests/test_led_calibration.py`, `tests/test_presentation.py`, `tests/test_scene_log_ui.py`: **6 passed in 88.67 seconds**. Browser checks cover full-viewport gray, image/label exclusion, FIT/FILL zoom aspect ratio, pinned fault pixels, separate display cleanliness and unchanged source payload. Metadata ZIP/restore and affected existing presentation/logging checks passed. No broad regression or high-resolution stress rerun was performed.
