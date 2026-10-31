"""Actual acquired images, real metadata, and browser-rendered SVG acceptance."""
from pathlib import Path
import sys,runpy,hashlib,json
import numpy as np
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_scene_frames import acquire_scene,presentation_payload
from pv_presentation import presentation_html


def test_multifault_clean_images_and_labeled_boxes(tmp_path):
    ns=runpy.run_path(str(ROOT/'tests/test_collection_acceptance.py'))['namespace']()
    ns['st'].query_params.update(location='grass_open',custom_layout='1',row_count='2',panels_per_row='4')
    records=[]
    for i,(r,c,code) in enumerate([(1,1,'CellCracking'),(2,3,'SoilingDust'),(2,4,'SurfaceObstruction')]):
        row,cell=ns['default_cell_for_array']('grass_open',r,c,1,20)
        record=ns['fault_record'](ns['location_by_id']('grass_open'),'thermal',ns['row_by_id']('grass_open',row),cell,code,7)
        record['experiment_fault_id']='bounding-acceptance-'+str(i);records.append(record)
    ns['st'].session_state['random_fault_records']=records
    scene=acquire_scene(ns,'Generated')
    hashes={key:hashlib.sha256(scene['far'][key].tobytes()).hexdigest() for key in ['original','rgb','thermal']}
    payload=presentation_payload(ns,scene)
    assert {b['name'] for b in payload['far']['boxes']}=={'Cell Cracking','Dust / Soiling','Surface Obstruction'}
    # The box is the actual module extent, never a location guessed from pixels.
    assert {b['id']:b['bbox'] for b in payload['far']['boxes']}=={f['id']:f['module_bbox'] for f in scene['faults']}
    out=ROOT.parent/'outputs/presentation_ui';out.mkdir(exist_ok=True)
    html=out/'multifault_presentation.html';html.write_text(presentation_html(payload),encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport={'width':1920,'height':1080});page.goto(html.as_uri());page.locator('#base').evaluate('e=>e.decode()')
            assert page.get_by_role('button',name='BOUNDING BOX',exact=True).is_visible()
            assert not page.locator('#advanced').evaluate('e=>e.open') and page.locator('#calibration').is_hidden()
            assert page.locator('#status').is_hidden() and page.locator('[data-fault]').count()==0
            page.screenshot(path=str(out/'far_rgb_clean.png'))
            for distance in ['FAR','CLOSE']:
                page.locator('[data-distance='+distance+']').click()
                for mode in ['original','rgb','thermal']:
                    page.locator('[data-mode='+mode+']').click()
                    source=page.locator('#base').get_attribute('src')
                    page.locator('#boxes').click()
                    assert page.locator('#boxes').get_attribute('aria-pressed')=='true'
                    expected=page.evaluate('frame().boxes.length')
                    assert page.locator('[data-fault]').count()==expected
                    assert page.locator('#labels text').all_text_contents()==page.evaluate('frame().boxes.slice().sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0]).map(b=>b.name)')
                    boxes=page.locator('[data-label]').evaluate_all('els=>els.map(e=>{const r=e.getBBox();return [r.x,r.y,r.x+r.width,r.y+r.height]})')
                    for i,a in enumerate(boxes):
                        for b in boxes[i+1:]:assert min(a[2],b[2])<=max(a[0],b[0]) or min(a[3],b[3])<=max(a[1],b[1])
                    assert page.locator('#base').get_attribute('src')==source
                    if mode=='rgb':page.screenshot(path=str(out/(distance.lower()+'_rgb_boxes.png')))
                    page.locator('#boxes').click();assert page.locator('[data-fault]').count()==0
                    assert page.locator('#base').get_attribute('src')==source
            page.locator('[data-distance=FAR]').click();page.locator('[data-mode=rgb]').click();page.locator('#boxes').click()
            page.locator('#fullscreen').click();page.wait_for_function('document.fullscreenElement!==null')
            assert page.locator('[data-fault]').count()==3
            plane=page.locator('#plane').bounding_box();assert abs(plane['width']/plane['height']-3072/2048)<.005
            page.keyboard.press('b');assert page.locator('[data-fault]').count()==0
            page.keyboard.press('b');assert page.locator('[data-fault]').count()==3
            page.evaluate('document.exitFullscreen()');page.locator('#advanced > summary').click()
            page.locator('#clean').click();assert page.locator('[data-fault]').count()==0
            # One click must enable boxes even after Clean Filming was enabled.
            page.locator('#boxes').click();assert page.locator('[data-fault]').count()==3
            page.locator('#calibration').select_option('gray');assert page.locator('[data-fault]').count()==0
            assert page.locator('#boxes').is_disabled()
        finally:browser.close()
    assert hashes=={key:hashlib.sha256(scene['far'][key].tobytes()).hexdigest() for key in hashes}
    (out/'acceptance.json').write_text(json.dumps({'faults':[f['id'] for f in scene['faults']],'image_hashes_unchanged':hashes,'far_close_modalities':6,'label_overlap':'none in tested frames','fullscreen':'passed','clean_toggle':'passed'},indent=2))
