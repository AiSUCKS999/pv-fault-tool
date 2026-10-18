"""Small Streamlit widget-tree smoke check for the active workflows."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def assert_no_exception(app, stage):
    if app.exception:
        raise AssertionError(f"{stage}: {app.exception[0].message}")


def main():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert_no_exception(app, "Generated Farm")
    if app.slider:
        raise AssertionError("Active Generated Farm still exposes continuous slider controls")
    source = next(item for item in app.radio if item.label == "Source mode")
    source.set_value("Real PV Farm Image")
    app.run()
    assert_no_exception(app, "Real Farm")
    if app.slider:
        raise AssertionError("Active Real Farm still exposes continuous slider controls")
    navigation = next(item for item in app.radio if item.label == "Real Farm workflow")
    expected = ["Farm Setup", "Thermal", "Faults", "Review", "Export"]
    if list(navigation.options) != expected:
        raise AssertionError(f"Real Farm workflow sections differ: {navigation.options}")
    source_text = (ROOT / "pv_real_farm.py").read_text(encoding="utf-8")
    if 'with st.expander("Advanced / Manual Geometry Correction", expanded=False)' not in source_text:
        raise AssertionError("Manual geometry correction is not collapsed by default")
    print("targeted_ui_smoke: PASS")


if __name__ == "__main__":
    main()
