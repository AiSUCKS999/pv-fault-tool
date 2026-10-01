# Thin Streamlit entrypoint. The app is split into editable section files.
# The sections are executed in one shared namespace to preserve the old script behavior
# while making each part small enough to edit in VS Code.
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent
_SECTION_FILES = [
    "pv_faults.py",
    "pv_patch_bank.py",
    "pv_layout.py",
    "pv_annotations.py",
    "pv_render_rgb.py",
    "pv_render_thermal.py",
    "pv_ui.py",
]

for _section in _SECTION_FILES:
    _path = _TOOL_DIR / _section
    exec(compile(_path.read_text(encoding="utf-8-sig"), str(_path), "exec"), globals())

if __name__ == "__main__":
    main()
