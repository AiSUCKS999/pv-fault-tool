# Fault and hotspot release

- Integrates the 26 approved thermal hotspot residuals, 14 robust bypass diode references, and 273 bypassed substring references.
- Restores close-view fault visibility with a shared robust scalar palette and the close-view sensor response.
- Keeps hotspots and partial shading in far-view observations and the seven root-cause labels in close diagnostics.
- Fixes crop assertions for visible module geometry at image boundaries.
- Includes the runtime RGB scenes and geometry metadata for all 180 supported layouts.

These are relative thermal anomaly patterns. Cell cracking, dust/soiling, partial shading, and reversed polarity still lack qualified production reference banks; surface obstruction and offline-module evidence remain limited. Their existing diagnostic options remain available.

Validation: regression smoke across five scenes and 1x1, 3x3, and 6x6 layouts, plus production-bank and 180-layout asset checks. Research collections and intermediate previews are retained in the original local repository.
