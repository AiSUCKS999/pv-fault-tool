"""Public navigation and native simulation/export integrity after removing debug UI."""
from pathlib import Path
import sys,io,json,zipfile,hashlib
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_scene_log import SceneJournal
REMOVED=['Advanced Options','ROS mission export','CONFIRM FAR','CONFIRM CLOSE','SAVE EXPERIMENT','RESTORE','Restore session','Technical context','standalone presentation']

def check(app):
    assert not app.exception,[e.message for e in app.exception]
    assert not app.error,[e.value for e in app.error]
    labels=[b.label for b in app.button]+[b.label for b in app.get('download_button')]+[e.label for e in app.expander]
    assert not any(term in label for term in REMOVED for label in labels),labels

def test_public_generated_and_real_native_exports():
    a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run();check(a)
    assert a.radio(key='public_section_Generated').options==['Scene Setup','Layout','Faults','Presentation','Review','Export','DATA COLLECTION']
    for mode in ['Generated','Real Farm']:
        if mode=='Real Farm':a.radio(key='public_mode').set_value('Real PV Farm').run();check(a)
        next(b for b in a.button if b.label=='LOAD FARM').click().run();check(a)
        if mode=='Generated':
            a.radio(key='public_section_'+mode).set_value('Layout').run();check(a)
            next(x for x in a.number_input if x.label=='Rows').set_value(2)
            next(x for x in a.number_input if x.label=='Arrays per row').set_value(4)
            next(b for b in a.button if b.label=='APPLY LAYOUT').click().run();check(a)
            assert a.session_state['pv_scene_frames_Generated'][1]['source']['rows']==2
            assert a.session_state['pv_scene_frames_Generated'][1]['source']['columns']==4
        a.radio(key='public_section_'+mode).set_value('Faults').run();check(a)
        next(b for b in a.button if b.label=='GENERATE FAULTS').click().run();check(a)
        exp=SceneJournal(a.session_state.filtered_state).active(mode);ids=[f['id'] for f in exp['revisions'][-1]['faults']]
        assert len(ids)==3 and len(set(ids))==3
        for section in ['Presentation','Review','Export','DATA COLLECTION']:
            a.radio(key='public_section_'+mode).set_value(section).run();check(a)
            assert [f['id'] for f in SceneJournal(a.session_state.filtered_state).active(mode)['revisions'][-1]['faults']]==ids
        a.radio(key='public_section_'+mode).set_value('Export').run();check(a)
        downloads={b.label for b in a.get('download_button')}
        assert downloads=={'DOWNLOAD IMAGES & METADATA','DOWNLOAD ANNOTATIONS','DOWNLOAD FAULT CSV'}
        journal=SceneJournal(a.session_state.filtered_state)
        with zipfile.ZipFile(io.BytesIO(journal.bundle())) as z:
            assert z.testzip() is None
            assert {'scene_log.json','faults.csv'}.issubset(z.namelist())
            revision=exp['revisions'][-1]
            for ref in revision['references']:assert hashlib.sha256(z.read(ref['filename'])).hexdigest()==ref['sha256']
        a.radio(key='public_section_'+mode).set_value('DATA COLLECTION').run();check(a)
        # First case per mode is an existing healthy case; no invented targets.
        next(b for b in a.button if b.label=='PREPARE SCENARIO').click().run();check(a)
        assert a.session_state['public_collection_results'][-1]['Mode']==mode
        assert any(b.label=='DOWNLOAD COLLECTION LOG' for b in a.get('download_button'))
        a.radio(key='public_section_'+mode).set_value('Scene Setup').run();check(a)
    assert len(a.selectbox(key='public_farm_Real').options)==14
