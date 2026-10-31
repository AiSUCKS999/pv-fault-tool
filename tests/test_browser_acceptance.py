"""Bounded browser acceptance: clean/annotated views and real Streamlit controls."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/experiment_recovery"


def test_scene_browser_boxes_and_same_fault_navigation():
    # Fresh checkouts have no local evidence files. Reuse the existing bounded
    # scenario tests to construct the two exact scenes, rather than fake assets.
    if not all((OUT/(name+".html")).exists() for name in ("generated_6x6","real_05")):
        import runpy
        helpers=runpy.run_path(str(ROOT/"tests/test_scene_experiment.py"))
        ns=helpers["ns"].__wrapped__()
        helpers["test_generated_full_scene_and_recovery"](ns)
        helpers["test_real_multiple_faults_same_geometry_and_photograph"](ns)
    playwright=pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080})
            for name in ("generated_6x6","real_05"):
                page.goto((OUT/(name+".html")).as_uri())
                page.locator("#base").evaluate("e=>e.decode()")
                original_ids=page.evaluate("payload.faults.map(f=>f.id)")
                page.locator("#fullscreen").click();page.wait_for_function("document.fullscreenElement!==null")
                page.keyboard.press("r");page.screenshot(path=str(OUT/(name+"_browser_clean_rgb.png")))
                assert page.locator("[data-fault]").count()==0
                page.keyboard.press("b");page.screenshot(path=str(OUT/(name+"_browser_boxes_rgb.png")))
                assert page.locator("[data-fault]").count()==len(original_ids)
                page.keyboard.press("t");page.screenshot(path=str(OUT/(name+"_browser_boxes_thermal.png")))
                assert page.evaluate("payload.faults.map(f=>f.id)")==original_ids
                page.locator("[data-fault] rect").first.click()
                assert page.evaluate("state.distance")=="CLOSE"
                selected=page.evaluate("fault().id")
                assert selected in original_ids
                page.screenshot(path=str(OUT/(name+"_browser_close.png")))
                page.keyboard.press("ArrowRight")
                assert page.evaluate("fault().id")!=selected
                page.keyboard.press("f");page.keyboard.press("b")
                assert page.locator("[data-fault]").count()==0
                assert page.evaluate("payload.faults.map(f=>f.id)")==original_ids
                page.evaluate("document.exitFullscreen()")
        finally:browser.close()


def test_live_local_streamlit_no_typing_and_capture_bridge():
    playwright=pytest.importorskip("playwright.sync_api")
    OUT.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
    # Child server exists only inside this foreground, bounded test. Always reap.
    with (OUT/"browser_server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,"-m","streamlit","run","app.py","--server.headless=true",
            f"--server.port={port}","--browser.gatherUsageStats=false"],cwd=ROOT,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+25
            while time.monotonic()<deadline:
                try:
                    urllib.request.urlopen(f"http://localhost:{port}/_stcore/health",timeout=1).close();break
                except OSError:time.sleep(.2)
            else:raise AssertionError("Bounded test server did not start")
            with playwright.sync_playwright() as p:
                browser=p.chromium.launch(channel="msedge",headless=True)
                try:
                    page=browser.new_page(viewport={"width":1920,"height":1080})
                    page.set_default_timeout(55000)
                    page.goto(f"http://localhost:{port}/?location=grass_open&custom_layout=1&row_count=6&panels_per_row=6&section=DATA%20COLLECTION")
                    page.get_by_role("button",name="GENERATE EXPERIMENT",exact=True).click()
                    page.get_by_role("button",name="SAVE EXPERIMENT",exact=True).wait_for()
                    assert page.locator('[data-testid="stException"]').count()==0
                    page.screenshot(path=str(OUT/"generated_collection_ui.png"),full_page=True)
                    frame=next(f for f in page.frames if f.locator("#screen").count())
                    frame.locator('[data-distance="CLOSE"]').click()
                    button=page.get_by_role("button",name="CONFIRM CLOSE RECORDED",exact=True)
                    from playwright.sync_api import expect
                    expect(button).to_be_enabled(timeout=15000)
                    frame.locator("#next").click()
                    expect(button).to_be_enabled()
                    button.click()
                    page.get_by_text("Close recordings confirmed: 1/6",exact=True).wait_for()
                    page.get_by_role("button",name="CONFIRM FAR RECORDED",exact=True).click()
                    page.get_by_role("button",name="SAVE EXPERIMENT",exact=True).click()
                    page.get_by_role("button",name="NEXT EXPERIMENT",exact=True).wait_for()
                    page.screenshot(path=str(OUT/"generated_collection_saved_ui.png"),full_page=True)
                    with page.expect_download() as download:
                        page.get_by_role("button",name="DOWNLOAD SESSION ZIP / BACKUP",exact=True).click()
                    download.value.save_as(OUT/"browser_scene_session.zip")
                    import zipfile
                    with zipfile.ZipFile(OUT/"browser_scene_session.zip") as z:
                        data=json.loads(z.read("scene_log.json"))
                        exp=next(iter(data["experiments"].values()))
                        assert len(exp["revisions"][0]["faults"])==6
                        event=next(e for e in exp["events"] if e.get("stage")=="CLOSE")
                        assert event["fault_id"]==exp["revisions"][0]["faults"][1]["id"]
                    assert page.locator('[data-testid="stException"]').count()==0
                finally:browser.close()
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
