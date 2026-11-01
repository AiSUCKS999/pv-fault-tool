"""Regression fixture for retained internal/robotics-era controls, not a public page.
The public website's entry point is tested in test_public_workflow.py and browser acceptance.
"""
from pathlib import Path
import runpy
application=runpy.run_path(str(Path(__file__).resolve().parents[1]/'app.py'),run_name='legacy_control_regression')
application['main']()
