# PV Fault Tool - Project Map

## Main Application
- app.py
- pv_ui.py
- pv_real_farm.py
- pv_render_thermal.py

## Assets
- assets/

## Production Fault Reference Banks
- assets/fault_reference_bank/

## Current Production-Ready Engines
- Thermal Hotspot
- Bypass Diode Fault
- Bypassed Substring / Section

## Real Farm Workflow
- Real RGB PV farm image
- Manual / refined PV geometry
- Synthetic thermal scene generation
- Production-bank hotspot/fault injection

## Generated Outputs
- artifacts/

## Large External Datasets
Large datasets are NOT expected to live inside Git.
Their local locations are configured in:
- config/paths.yaml

The shared template is:
- config/paths.example.yaml

Each team member keeps their own paths.yaml.

## Important Rule
Application code should not contain personal absolute paths such as:
C:\Users\ammar.AMMAR\...
or machine-specific D:\ dataset paths.

Project-internal files should resolve relative to PROJECT_ROOT.
