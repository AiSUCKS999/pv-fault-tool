"""Regressions use actual rendered PV acquisitions, not colored placeholder frames."""
from pathlib import Path
import sys,runpy,json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from pv_scene_frames import acquire_scene,presentation_payload
from pv_presentation import presentation_html
from presentation_browser_checks import exercise_normal_controls

def test_calibration_exit_all_normal_buttons_actual_pixels(tmp_path):
    ns=runpy.run_path(str(ROOT/'tests/test_collection_acceptance.py'))['namespace']()
    ns['st'].query_params.update(location='grass_open',custom_layout='1',row_count='2',panels_per_row='4')
    records=[]
    for i,(r,c,code) in enumerate([(1,1,'CellCracking'),(2,3,'SoilingDust'),(2,4,'SurfaceObstruction')]):
        row,cell=ns['default_cell_for_array']('grass_open',r,c,1,20)
        record=ns['fault_record'](ns['location_by_id']('grass_open'),'thermal',ns['row_by_id']('grass_open',row),cell,code,7)
        record['experiment_fault_id']='calibration-exit-'+str(i);records.append(record)
    ns['st'].session_state['random_fault_records']=records
    payload=presentation_payload(ns,acquire_scene(ns,'Generated'))
    file=tmp_path/'presentation.html';file.write_text(presentation_html(payload),encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport={'width':1600,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(file.as_uri());report=exercise_normal_controls(page)
            assert len(report['checks'])==24 and not errors
            page.locator('#calibration').select_option('gray');page.locator('#screen').click();page.keyboard.press('f')
            assert page.evaluate('state.calibration')=='off' and page.evaluate('frame().width')==payload['far']['width']
            page.locator('#calibration').select_option('detail')
            with page.expect_popup() as opened:page.locator('#open').click()
            display=opened.value;display.wait_for_function("state.calibration==='detail'")
            page.locator('[data-distance=FAR]').click();display.wait_for_function("state.calibration==='off' && state.distance==='FAR'")
            assert display.locator('#controls').is_hidden() and display.locator('#calibration-status').is_hidden()
            assert display.locator('#base').evaluate('e=>e.naturalWidth')==payload['far']['width']
            display.close()
            out=ROOT.parent/'outputs/calibration_hotfix';out.mkdir(parents=True,exist_ok=True)
            (out/'native_browser_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            page.screenshot(path=str(out/'native_far_after_calibration.png'),timeout=15000)
        finally:browser.close()
