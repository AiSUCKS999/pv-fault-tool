"""Display-only calibration: no render-engine calls or large source regeneration."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import uuid

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_presentation import presentation_html,image_data
from pv_scene_log import SceneJournal


def test_calibration_browser_and_clean_filming(tmp_path):
    def png(color):
        out=io.BytesIO();Image.new('RGB',(160,100),color).save(out,format='PNG');return image_data(out.getvalue())
    frame={'width':160,'height':100,'original':png('gray'),'rgb':png('blue'),'thermal':png('red'),
           'boxes':[{'id':'F1','bbox':[20,20,60,60],'kind':'module'}]}
    payload={'identity':'test','revision':'r1','faults':[{'id':'F1'}],'far':frame,'close':{'F1':frame}}
    before=deepcopy(payload)
    file=tmp_path/'led.html';file.write_text(presentation_html(payload),encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport={'width':1280,'height':800});page.goto(file.as_uri())
            page.locator('#boxes').click();assert page.locator('[data-fault]').count()==1
            page.locator('#advanced > summary').click()
            page.locator('#clean').click();assert page.locator('[data-fault]').count()==0
            page.locator('#calibration').select_option('gray')
            page.locator('#fullscreen').click();page.wait_for_function('document.fullscreenElement!==null')
            assert page.locator('#gray').evaluate('e=>getComputedStyle(e).backgroundColor')=='rgb(128, 128, 128)'
            assert page.locator('#gray').bounding_box()=={'x':0,'y':0,'width':1280,'height':800}
            assert page.locator('[data-fault]').count()==0
            page.evaluate('document.exitFullscreen()')
            page.locator('#calibration').select_option('detail')
            assert page.locator('#base').get_attribute('src')==frame['original']
            page.locator('#calibration').select_option('fault')
            pinned=page.locator('#base').get_attribute('src')
            page.keyboard.press('t');page.keyboard.press('f')
            assert page.locator('#base').get_attribute('src')==pinned==frame['rgb']
            for fit in ['fit','fill']:
                page.locator('#fit').select_option(fit)
                page.locator('#zoom').fill('1.5');page.locator('#zoom').dispatch_event('input')
                assert abs(page.locator('#plane').evaluate('e=>e.clientWidth/e.clientHeight')-1.6)<.01
            assert page.evaluate('payload')==before
            page.locator('#calibration').select_option('off')
            page.locator('#clean').click();assert page.locator('[data-fault]').count()==1
            # Separate display remains image-only during calibration too.
            page.locator('#calibration').select_option('gray')
            with page.expect_popup() as popup:page.locator('#open').click()
            display=popup.value;display.wait_for_function("document.body.classList.contains('presentation')")
            display.wait_for_function("state.calibration==='gray'")
            assert display.locator('#controls').is_hidden() and display.locator('#status').is_hidden()
            assert display.locator('[data-fault]').count()==0
            display.close()
        finally:browser.close()


def test_optional_settings_export_restore_without_reference_changes():
    journal=SceneJournal({});eid='scene-'+uuid.uuid4().hex
    exp={'experiment_id':eid,'mode':'Generated','revisions':[{'signature':'r1','revision':1,'faults':[],
        'references':[]}],'events':[]}
    journal.data['experiments'][eid]=exp;journal.data['active']['Generated']=eid
    prior=deepcopy(exp['revisions'])
    settings={'pattern':'gray','gray_code':128,'zoom':1.5,'fit':'fill','viewport':[1920,1080]}
    value={'signature':'r1','calibration_settings':settings}
    journal.record_presentation('Generated',value);journal.record_presentation('Generated',value)
    assert len(exp['events'])==1 and exp['led_calibration']==settings
    assert exp['revisions']==prior and not journal.blobs
    restored=SceneJournal({});restored.restore(journal.bundle())
    assert restored.active('Generated')['led_calibration']==settings
    journal.record_presentation('Generated',{'signature':'other','calibration_settings':settings})
    assert len(exp['events'])==1
