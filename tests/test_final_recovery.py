from copy import deepcopy
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zipfile

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def namespace():
    return runpy.run_path(str(ROOT/'tests/test_collection_acceptance.py'))['namespace']()


def test_random_module_positions_independent_repeatable_and_exact_counts():
    ns=namespace();ns['st'].session_state['real_farm_prepared_picker']='image_05';ns['real_farm_catalog_selected']()
    state=ns['st'].session_state['real_farm_state'];image=ns['st'].session_state['real_farm_image']
    targets=ns['real_farm_camera_candidates'](state,image)[1]
    manual=ns['real_farm_camera_new_observation'](state,image,targets[0]);saved=deepcopy(manual)
    def generate(seed):
        ns['real_farm_camera_generate_random'](state,image,targets,10,seed,['SoilingDust','SingleDiode'],
            policy='Counts per class',counts={'SoilingDust':6,'SingleDiode':4},severity_distribution='Fixed',severity=5)
        return deepcopy([o for o in state['camera_observations'] if o.get('placement_origin')=='random'])
    first=generate(137);second=generate(137)
    def positions(items):return [(o['module_id'],o['center_rgb_px'],o['local_uv'],o['severity'],o['diagnostic_fault_type']) for o in items]
    assert positions(first)==positions(second)
    assert len({tuple(np.round(o['local_uv'],4)) for o in first})==10
    assert positions(generate(271))!=positions(first)
    assert sum(o['diagnostic_fault_type']=='SoilingDust' for o in first)==6
    assert len({o['observation_id'] for o in state['camera_observations']})==11
    assert next(o for o in state['camera_observations'] if o['observation_id']==saved['observation_id'])==saved
    lookup={m['module_id']:m for m in targets}
    for o in first:
        x,y=np.rint(o['center_rgb_px']).astype(int);q=np.float32(lookup[o['module_id']]['active_polygon_rgb'])
        assert cv2.pointPolygonTest(q,tuple(map(float,o['center_rgb_px'])),False)>=0
        assert ns['real_farm_module_mask'](image.size,[lookup[o['module_id']]],True,(x,y,x+1,y+1)).any()


@pytest.mark.skipif(os.environ.get('PV_HIGHRES_STRESS')!='1',reason='Explicit bounded stress run; avoid repeating in routine regression')
def test_native_image14_stress():
    from pv_scene_generate import generate_scene
    from pv_scene_frames import acquire_scene,presentation_payload
    from pv_scene_log import SceneJournal
    from pv_presentation import presentation_html
    ns=namespace();start=time.perf_counter()
    codes=list(ns['CLOSE_DIAGNOSTIC_FAULT_TYPES'])+['SingleHotSpot']
    request=dict(source='image_14',count=12,classes=codes,policy='Counts per class',
        counts={c:2 if i<4 else 1 for i,c in enumerate(codes)},seed=137,severity=7,severity_distribution='Fixed',placement='Spread')
    scene=generate_scene(ns,'Real Farm',request);acquire_seconds=time.perf_counter()-start
    assert scene['source']['size']==[4742,3159] and len(scene['faults'])==12
    ids=[f['id'] for f in scene['faults']]
    assert len(set(ids))==12 and len({f['module_id'] for f in scene['faults']})==12
    for f in scene['faults']:
        assert f['severity']==7
        close=scene['close'][f['id']]
        assert close['rgb'].size==close['thermal'].size
    fast=time.perf_counter()
    for _ in range(20):assert acquire_scene(ns,'Real Farm') is scene
    switch_seconds=time.perf_counter()-fast
    j=SceneJournal({});exp=j.capture(ns,scene,request,new=True);raw=j.bundle()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        meta=json.loads(z.read('scene_log.json'))['experiments'][exp['experiment_id']]['revisions'][0]
        assert [f['id'] for f in meta['faults']]==ids
        assert all(hashlib.sha256(z.read(r['filename'])).hexdigest()==r['sha256'] for r in meta['references'])
    restored=SceneJournal({});restored.restore(raw);assert restored.data==j.data
    out=ROOT/'artifacts/experiment_recovery';out.mkdir(exist_ok=True,parents=True)
    (out/'image14_stress.html').write_text(presentation_html(presentation_payload(ns,scene)),encoding='utf-8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport={'width':1920,'height':1080})
            page.goto((out/'image14_stress.html').as_uri());page.locator('#base').evaluate('e=>e.decode()')
            page.locator('#boxes').click();assert page.locator('[data-fault]').count()==12
            page.locator('[data-fault] rect').first.click();assert page.evaluate('state.distance')=='CLOSE'
            for _ in range(12):
                for key in ['t','r','f','c','ArrowRight']:page.keyboard.press(key)
            assert page.evaluate('payload.faults.map(f=>f.id)')==ids
            page.keyboard.press('f');page.keyboard.press('t')
            page.screenshot(path=str(out/'image14_stress_annotated.png'))
            page.keyboard.press('b');assert page.locator('[data-fault]').count()==0
        finally:browser.close()
    class Memory(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    mem=Memory();mem.cb=ctypes.sizeof(mem)
    handle=ctypes.windll.kernel32.GetCurrentProcess
    handle.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle()),ctypes.byref(mem),mem.cb)
    report={'source_size':scene['source']['size'],'faults':12,'native_acquisition_seconds':acquire_seconds,
        'cached_acquisition_20_calls_seconds':switch_seconds,'zip_bytes':len(raw),
        'peak_working_set_MiB':mem.PeakWorkingSetSize/1048576,'total_seconds':time.perf_counter()-start,
        'browser_switches':60,'physical_display_verified':False}
    (out/'image14_stress.json').write_text(json.dumps(report,indent=2));print(report)
