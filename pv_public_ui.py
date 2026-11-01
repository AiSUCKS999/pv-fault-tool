"""Focused public workflow over the existing simulator, acquisitions and exports.
Robotics, recovery tools and manual recording controls are not public pages.
"""
from copy import deepcopy
import csv,io,json
from pathlib import Path

from pv_scene_frames import acquire_scene,current_signature,presentation_payload
from pv_scene_ui import _generate,_capture,_error,_remove
from pv_scene_log import SceneJournal,json_bytes
from pv_presentation import show_presentation

SECTIONS=['Scene Setup','Layout','Faults','Presentation','Review','Export','DATA COLLECTION']


def csv_bytes(rows,fields):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def configuration(ns,mode):
    st=ns['st']
    if mode=='Generated':
        scene=ns['selected_location']()['id']
        return {'scene':scene,'rows':ns['custom_row_count'](scene),'columns':ns['custom_panels_per_row'](scene)}
    return {'source':st.session_state.get('real_farm_state',{}).get('prepared_catalog_id')}


def request(ns,mode,count=0,classes=(),severity=7):
    return {**configuration(ns,mode),'count':count,'classes':list(classes),'severity':severity,
        'seed':42,'policy':'Balanced','placement':'Spread','clear_existing':True}


def generate(ns,mode,req):
    # Existing generation/capture remains transactional and uses native geometry.
    try:_generate(ns,mode,req)
    except MemoryError:ns['st'].session_state['pv_scene_error']='This image exceeds available memory. Choose a smaller photograph.'
    if message:=ns['st'].session_state.pop('pv_scene_error',None):ns['st'].error(message)


def source_setup(ns,mode):
    st=ns['st'];config=configuration(ns,mode)
    with st.form('public_source_'+mode):
        if mode=='Generated':
            ids=[l['id'] for l in ns['LOCATIONS'] if l['id'] in ns['LAYOUT_CAPS']]
            choice=st.selectbox('PV farm',ids,index=ids.index(config['scene']) if config['scene'] in ids else 0,
                format_func=lambda i:ns['location_by_id'](i)['name'],key='public_farm_Generated')
            req=request(ns,mode);req['scene']=choice
            # The existing engine rejects unavailable layouts; never substitute one.
        else:
            entries=ns['real_farm_catalog_entries']();ids=[e['id'] for e in entries]
            default=config.get('source') or ('image_05' if 'image_05' in ids else ids[0])
            choice=st.selectbox('PV photograph',ids,index=ids.index(default),format_func=lambda i:next(e['label'] for e in entries if e['id']==i),key='public_farm_Real')
            req=request(ns,mode);req['source']=choice
        submitted=st.form_submit_button('LOAD FARM',type='primary')
    if submitted:generate(ns,mode,req)
    if mode=='Real Farm' and st.session_state.get('real_farm_image') is not None:st.image(st.session_state['real_farm_image'],use_container_width=True)
    elif mode=='Generated' and SceneJournal(st.session_state).active(mode):st.image(acquire_scene(ns,mode)['far']['original'],use_container_width=True)


def layout(ns,mode):
    st=ns['st'];config=configuration(ns,mode)
    if mode=='Real Farm':
        state=st.session_state.get('real_farm_state')
        if not state:st.info('Load a photograph in Scene Setup first.');return
        st.write(f"{len(state.get('modules',[]))} prepared modules. Photograph geometry is fixed.")
        return
    with st.form('public_layout'):
        scene=config['scene'];cols=st.columns(2)
        rows=cols[0].number_input('Rows',1,ns['max_custom_rows'](scene),config['rows'],key='public_rows_'+scene+f"_{config['rows']}x{config['columns']}")
        columns=cols[1].number_input('Arrays per row',1,6,config['columns'],key='public_columns_'+scene+f"_{config['rows']}x{config['columns']}")
        st.caption('Four modules per array.')
        req=request(ns,mode);req.update(rows=rows,columns=columns)
        submitted=st.form_submit_button('APPLY LAYOUT',type='primary')
    if submitted:generate(ns,mode,req)


def faults(ns,mode):
    st=ns['st']
    if mode=='Real Farm' and st.session_state.get('real_farm_image') is None:st.info('Load a photograph in Scene Setup first.');return
    with st.form('public_faults_'+mode):
        codes=[*ns['CLOSE_DIAGNOSTIC_FAULT_TYPES'],'SingleHotSpot','PartialShading']
        count=st.number_input('Faulted modules',0,12,3,key='public_fault_count_'+mode)
        classes=st.multiselect('Fault types',codes,default=['SoilingDust','SurfaceObstruction','CellCracking'],format_func=ns['fault_display_name'],key='public_fault_classes_'+mode)
        severity=st.slider('Severity',1,10,7,key='public_fault_severity_'+mode)
        submitted=st.form_submit_button('GENERATE FAULTS',type='primary')
    if submitted:generate(ns,mode,request(ns,mode,count,classes,severity))
    exp=SceneJournal(st.session_state).active(mode)
    if exp:
        st.button('CLEAR FAULTS',on_click=_remove,args=(ns,mode,True))
        frames=acquire_scene(ns,mode);st.image(frames['far']['rgb'],use_container_width=True)


def frames_if_ready(ns,mode):
    st=ns['st']
    if mode=='Real Farm' and st.session_state.get('real_farm_image') is None:st.info('Load a photograph in Scene Setup first.');return None
    if not SceneJournal(st.session_state).active(mode):_capture(ns,mode)
    frames=acquire_scene(ns,mode)
    return frames


def review(ns,frames):
    st=ns['st']
    rows=[{'Fault ID':f['id'],'Fault':ns['fault_display_name'](f['fault_type']),'Module':f['module_id'],'Severity':f['severity']} for f in frames['faults']]
    st.dataframe(rows,hide_index=True,use_container_width=True)


def export(ns,mode,frames):
    st=ns['st'];journal=SceneJournal(st.session_state)
    st.download_button('DOWNLOAD IMAGES & METADATA',journal.bundle(),file_name='pv_experiment.zip',mime='application/zip',on_click='ignore')
    annotations={'source':frames['source'],'image_size':list(frames['far']['rgb'].size),'coordinate_system':'native source pixels',
        'synthetic_only':True,'physical_ground_truth_verified':False,'faults':frames['faults']}
    st.download_button('DOWNLOAD ANNOTATIONS',json_bytes(annotations),file_name='pv_annotations.json',mime='application/json',on_click='ignore')
    st.download_button('DOWNLOAD FAULT CSV',csv_bytes(frames['faults'],['id','fault_type','module_id','row','array','module','severity','cell_ids','module_bbox']),file_name='pv_faults.csv',mime='text/csv',on_click='ignore')
    st.caption('Session data is temporary. Download exports to keep it.')


def prepare_case(ns,mode,case,options):
    def action():
        if mode=='Real Farm' and case['Scenario stage']!='Healthy':
            required=2 if case['FAR observable code']=='Multiple SingleHotSpot observations' else 1
            if len(options.get('modules',[]))<required:raise ValueError(f'Select at least {required} target module(s) for this case.')
        from pv_data_collection_prepare import prepare_generated,prepare_real
        result=(prepare_generated if mode=='Generated' else prepare_real)(ns,case,options)
        exp=_capture(ns,mode,new=True);exp['case_id']=case['Case ID'];exp['planned_case']=deepcopy(case);exp['preparation']=result
        rows=ns['st'].session_state.setdefault('public_collection_results',[])
        rows.append({'Case ID':case['Case ID'],'Mode':mode,'Result':result['status'],'Source':exp['revisions'][-1]['source']['identity'],
            'Fault IDs':','.join(f['id'] for f in exp['revisions'][-1]['faults']),'Experiment ID':exp['experiment_id']})
    _error(ns,action)


def collection(ns,mode):
    st=ns['st'];from pv_data_collection import _load_cases
    cases=[c for c in _load_cases() if c['Source mode']==mode]
    priorities=list(dict.fromkeys(c['Priority'] for c in cases))
    chosen=st.multiselect('Priority',priorities,default=priorities,key='public_priorities_'+mode)
    filtered=[c for c in cases if c['Priority'] in chosen]
    if not filtered:st.info('Select a priority.');return
    byid={c['Case ID']:c for c in filtered}
    cid=st.selectbox('Test case',list(byid),format_func=lambda i:f"{i} · {byid[i]['Planned scenario']}",key='public_case_'+mode)
    case=byid[cid];options=configuration(ns,mode)
    if mode=='Real Farm':
        if not options.get('source'):st.info('Load the intended photograph in Scene Setup first.');return
        if case['Scenario stage']!='Healthy':
            state=st.session_state['real_farm_state'];image=st.session_state['real_farm_image']
            modules=ns['real_farm_camera_candidates'](state,image)[1];ids=[m['module_id'] for m in modules]
            required=2 if case['FAR observable code']=='Multiple SingleHotSpot observations' else 1
            options['modules']=st.multiselect('Target modules',ids,default=ids[:required],key='public_case_modules_'+cid+'_'+str(options.get('source')))
            if not options['modules']:st.info('Select a target module.');return
    st.button('PREPARE SCENARIO',on_click=prepare_case,args=(ns,mode,case,options),type='primary')
    results=[r for r in st.session_state.get('public_collection_results',[]) if r['Mode']==mode]
    if results:
        st.dataframe(results,hide_index=True,use_container_width=True)
        st.download_button('DOWNLOAD COLLECTION LOG',csv_bytes(results,list(results[0])),file_name='pv_collection.csv',mime='text/csv',on_click='ignore')
    st.download_button('DOWNLOAD CASE CATALOG',Path(__file__).with_name('pv_data_collection_cases.csv').read_bytes(),file_name='pv_cases.csv',mime='text/csv',on_click='ignore')


def render_public_app(ns):
    st=ns['st']
    with st.sidebar:
        mode=st.radio('PV farm',['Generated PV Farm','Real PV Farm'],index=1 if st.query_params.get('base_source')=='real_farm' else 0,key='public_mode')
        mode='Generated' if mode=='Generated PV Farm' else 'Real Farm'
        section_key='public_section_'+mode
        default=st.session_state.get(section_key,'Scene Setup')
        section=st.radio('Workflow',SECTIONS,index=SECTIONS.index(default),key=section_key)
    # Retain the existing Real Farm memory safeguard; these are reproducible
    # caches, not photographs, fault records, scene frames or journal data.
    if mode=='Generated':st.session_state.pop('real_farm_generated_cache_released',None)
    elif not st.session_state.get('real_farm_generated_cache_released'):
        for name in ('cached_native_thermal_master','cached_solar_render_thermal_base','cached_module_field_thermal_scene'):
            cached=ns.get(name)
            if cached is not None and hasattr(cached,'clear'):cached.clear()
        st.session_state['real_farm_generated_cache_released']=True
    values={'base_source':'generated' if mode=='Generated' else 'real_farm','section':section}
    for key,value in values.items():
        if st.query_params.get(key)!=value:st.query_params[key]=value
    if message:=st.session_state.pop('pv_scene_error',None):st.error(message)
    st.subheader(section)
    try:
        if section=='Scene Setup':source_setup(ns,mode)
        elif section=='Layout':layout(ns,mode)
        elif section=='Faults':faults(ns,mode)
        elif section=='DATA COLLECTION':collection(ns,mode)
        else:
            frames=frames_if_ready(ns,mode)
            if frames is None:return
            if section=='Presentation':
                saved=st.session_state.get('pv_scene_presentation_'+mode)
                if not saved or saved[0]!=frames['signature']:
                    saved=(frames['signature'],presentation_payload(ns,frames));st.session_state['pv_scene_presentation_'+mode]=saved
                show_presentation(ns,saved[1],key=mode)
            elif section=='Review':review(ns,frames)
            elif section=='Export':export(ns,mode,frames)
    except (ValueError,RuntimeError,KeyError,OSError) as exc:st.error(str(exc))
    except MemoryError:st.error('This photograph exceeds available memory. Choose a smaller source.')
    except Exception as exc:
        if ns.get('is_allocation_failure',lambda error:False)(exc):st.error('This photograph exceeds available memory. Choose a smaller source.')
        else:raise

PUBLIC_UI_REVISION = "simple-public-20261101-v1"
