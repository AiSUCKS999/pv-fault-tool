"""Importable component registration for the app's shared-exec section loader."""
from pathlib import Path
import streamlit.components.v1 as components

quad_editor = components.declare_component(
    "pv_module_quad_editor", path=str(Path(__file__).resolve().parent / "pv_geometry_editor")
)
