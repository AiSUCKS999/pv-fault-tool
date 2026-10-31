"""Real browser acceptance of the standalone image surface (no app server)."""
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pv_presentation import presentation_html
from pv_scene_frames import acquire_scene, presentation_payload


def test_native_generated_presentation_browser(tmp_path):
    playwright=pytest.importorskip("playwright.sync_api")
    ns=runpy.run_path(str(ROOT/"tests/test_collection_acceptance.py"))["namespace"]()
    ns["qp_set"](location="grass_open",custom_layout=1,row_count=6,panels_per_row=6)
    frames=acquire_scene(ns,"Generated")
    payload=presentation_payload(ns,frames)
    out=ROOT/"artifacts/experiment_recovery"
    out.mkdir(parents=True,exist_ok=True)
    html=out/"healthy_generated_presentation.html"
    html.write_text(presentation_html(payload),encoding="utf-8")
    evidence=[]
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(channel="msedge",headless=True)
        try:
            page=browser.new_page(viewport={"width":1920,"height":1080})
            page.goto(html.as_uri())
            page.locator("#base").evaluate("e=>e.decode()")
            page.get_by_text("PRESENT FULLSCREEN",exact=True).click()
            page.wait_for_function("document.fullscreenElement !== null")
            for width,height in [(1920,1080),(3840,2160),(3440,1440)]:
                page.set_viewport_size({"width":width,"height":height})
                for fit in ("fit","fill"):
                    page.evaluate("value=>{state.fit=value;changed()}",fit)
                    measured=page.evaluate("""()=>({screen:[screen.clientWidth,screen.clientHeight],
                        plane:[plane.clientWidth,plane.clientHeight],source:[frame().width,frame().height],
                        fullscreen:document.fullscreenElement.id})""")
                    assert measured["fullscreen"]=="screen"
                    assert measured["screen"]==[width,height]
                    w,h=measured["plane"];sw,sh=measured["source"]
                    assert abs(w/h-sw/sh)<.005
                    if fit=="fit":assert w<=width+1 and h<=height+1
                    else:assert w>=width-1 and h>=height-1
                    page.screenshot(path=str(out/f"presentation_{width}x{height}_{fit}.png"))
                    evidence.append({"viewport":[width,height],"fit":fit,**measured})
            page.keyboard.press("t")
            assert page.evaluate("state.mode")=="thermal"
            page.keyboard.press("r")
            assert page.evaluate("state.mode")=="rgb"
            page.evaluate("document.exitFullscreen()")
            with page.expect_popup() as opened:
                page.get_by_text("OPEN DISPLAY",exact=True).click()
            display=opened.value
            display.wait_for_function("document.body.classList.contains('presentation')")
            page.get_by_text("THERMAL",exact=True).click()
            display.wait_for_function("state.mode==='thermal'")
            assert display.locator("#controls").is_hidden()
            display.close()
        finally:browser.close()
    (out/"presentation_browser_metrics.json").write_text(json.dumps(evidence,indent=2))
