# Online deployment for the PV Fault Tool

This folder is the website version of the exact Streamlit tool.

## Fastest public link: Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload everything inside this `pv_fault_website` folder to the repository root.
3. Go to https://share.streamlit.io/ or https://streamlit.io/cloud.
4. Create a new app from that GitHub repo.
5. Set the main file path to:

```text
app.py
```

6. Deploy. Streamlit will install `requirements.txt` and `packages.txt` automatically.

## Files that must be included

- `app.py`
- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`
- `assets/`
- `annotations/`

Do not move the `assets` or `annotations` folders. The app reads them relative to `app.py`.
