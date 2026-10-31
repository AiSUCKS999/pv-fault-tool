# Technical context

Routine explanatory captions are collected here rather than displayed under ordinary controls. Actionable errors and verification restrictions remain in the interface.

Use this tab to choose the row, panel, module, and target cell. Use Cell Faults to assign faults across the selected panel.

Switch Camera view to Thermal to run the classifier check without generating an extra thermal scene in RGB mode.

Optional thermal-only observations. Hotspots do not imply visible RGB scorching. Use Visible Far conditions above for paired surface appearance and thermal response.

Generation settings are intentionally lightweight in the UI. Blender generation, topology, and safe placement are controlled by the renderer scripts.

Detector confidence is not physical verification. Inspect numbered outlines and native RGB crops, including missing modules, separators and image edges.

Rejected and unresolved regions remain recorded and block verified export; their empty canonical support is not a healthy-background label.

The blue polygon is only a coarse search ROI. Auto-Refine finds visible PV material; review the green result before accepting it as thermal support.

Clean exports contain no debug masks, selection outlines, labels, or support overlays. The annotated image is provided separately.

The real RGB image supplies the scene. Only accepted refined PV support or an explicit manual final polygon supplies the thermal mask; a coarse search ROI is never thermalized.

Click a recovered module marker in the scene. The selected stable module ID drives hotspot placement, Close View, Review, and Export.

One active observation per physical module. Adding again replaces that module's observation. Batch generation uses separate accepted modules.

Inspect physical boundaries before verified training export. Geometry correction is optional for thermal preview.

Geometry is stored in the canonical Real Farm state. Use the Export section for images, masks, module records, and metadata.

RGB review evidence is separate from the user's export approval. Partial visible faces are not automatically defective. Edits invalidate the affected review record.

Green: clear RGB support; cyan: legitimate clipped face; amber: unresolved; red: confirmed error. These colors do not grant training-label approval. Expand images to inspect native pixels.

Layout changes are proposals. Frame refinement uses this region's RGB; unsupported boundaries remain unresolved. Accept a complete table only after inspecting its RGB perimeters.

Select a module below, then drag its green corner handles onto the physical frame. Save updates thermal and export immediately. Draw a missed module with four perimeter clicks.

Confirm a small neighboring group to provide healthy thermal context for diagnosis. Unreviewed candidates remain outside the PV mask.

Select a diagnosis and press Apply Diagnosis to replace this observation's hotspot. Far, Close and export share the applied result. Enlarging the display adds no acquired detail.

Synthetic relative diagnostic: dedicated fault-specific morphology; not a production reference or measured temperature.

This creates two editable proposals. Fit/confirm each child's real frame afterward; a mathematical split does not verify it.

Click an outlined proposal or another PV surface. Adjust its four corners if needed, save the outline, then confirm to create a hotspot. Simulation only; not training verification.

Uncertain outlines may contain cell fragments. Thermal preview uses source-supported PV surfaces; these are not verified individual-module labels.
