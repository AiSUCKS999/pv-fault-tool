"""Explicit logical array-to-simulation map; never pixel-to-metre conversion."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json

SCHEMA='pv_ros_mission_v1'

def export_mission(scene,selected_ids):
    if scene.get('mode')!='Generated' or (scene['source'].get('rows'),scene['source'].get('columns'))!=(2,4):
        raise ValueError('Only Generated 2x4 arrays are mapped. Real Farm and other layouts require an explicit reviewed map.')
    lookup={f['id']:f for f in scene['faults']};selected=list(dict.fromkeys(selected_ids))
    if not selected or any(i not in lookup for i in selected):raise ValueError('Select current, mapped fault IDs')
    panels={}
    for oid in selected:
        f=lookup[oid];r,c=f.get('row'),f.get('array')
        if r not in (1,2) or c not in (1,2,3,4) or f.get('module') not in (1,2,3,4):raise ValueError(f'Unmapped source fault: {oid}')
        pid=f'R{r}C{c}';x=2+2.5*(c-1);y=2. if r==1 else 5.
        target=panels.setdefault(pid,{'panel_id':pid,'source_array':{'row':r,'column':c},
            'center_world_m':[x,y,.8],'inspection_waypoint_m':[x,.75 if r==1 else 3.75],
            'fault_ids':[],'source_faults':[]})
        target['fault_ids'].append(oid);target['source_faults'].append(deepcopy(f))
    mission={'schema':SCHEMA,'map_id':'pv_farm_2x4_v1','frame_id':'pv_farm','units':'metres',
        'mapping':'logical Generated array -> one surrogate Gazebo panel; source module/cell IDs retained; image pixels are not world coordinates',
        'simulated':True,'physical_ground_truth_verified':False,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source':deepcopy(scene['source']),'source_revision':scene['signature'],'targets':list(panels.values())}
    mission['mission_id']='mission-'+hashlib.sha256(json.dumps(mission,sort_keys=True).encode()).hexdigest()[:20]
    return mission

def render_export(ns,scene):
    st=ns['st']
    with st.expander('ROS 2 mission export (simulation)',expanded=False):
        st.caption('Generated 2x4 arrays map to eight Gazebo panels. Module IDs remain provenance; no image-to-world calibration is claimed.')
        ids=[f['id'] for f in scene['faults']]
        chosen=st.multiselect('Inspection faults',ids,default=ids,key='ros_mission_fault_ids',
            format_func=lambda oid:next(f"{oid} | row {f.get('row')} array {f.get('array')} module {f.get('module')} | {f['fault_type']}" for f in scene['faults'] if f['id']==oid))
        try:
            mission=export_mission(scene,chosen)
            st.download_button('EXPORT ROS MISSION JSON',json.dumps(mission,indent=2),file_name='pv_mission.json',mime='application/json')
            if st.button('SEND TO LOCAL ROS INBOX (NEXT LAUNCH)'):
                try:st.success('Saved local mission: '+str(write_local_inbox(mission)))
                except (OSError,ValueError) as exc:st.error(str(exc))
        except ValueError as exc:st.info(str(exc))


def render_prepare(ns,mode):
    st=ns['st']
    if mode!='Generated':return
    loc=ns['selected_location']()['id']
    if (ns['custom_row_count'](loc),ns['custom_panels_per_row'](loc))!=(2,4):return
    with st.expander('Prepare selected ROS inspection faults (development)',expanded=False):
        # Read existing native geometry, never invent missing panel polygons.
        native=ns['cached_native_thermal_master'](loc,ns['solar_render_cache_signature'](loc))
        if native is None:st.error('Native geometry unavailable');return
        supported=[f'R{r}C{c}' for r in (1,2) for c in range(1,5) if (r,c) in native[1]]
        missing=[f'R{r}C{c}' for r in (1,2) for c in range(1,5) if (r,c) not in native[1]]
        if missing:st.warning('Existing renderer cannot resolve these source targets: '+', '.join(missing))
        panels=st.multiselect('Source arrays to fault',supported,default=[p for p in ['R1C1','R1C4','R2C2'] if p in supported],key='ros_prepare_panels')
        fault=st.selectbox('Simulated fault class',ns['CLOSE_DIAGNOSTIC_FAULT_TYPES'],index=3,key='ros_prepare_class')
        level=st.slider('ROS preparation severity',1,10,7,key='ros_prepare_severity')
        def prepare():
            from pv_scene_generate import simulator_snapshot,restore_simulator
            from pv_scene_ui import _capture
            saved=simulator_snapshot(ns)
            try:
                records=[]
                if not panels:raise ValueError('Select at least one supported source array')
                for pid in panels:
                    r,c=int(pid[1]),int(pid[3]);row,cell=ns['default_cell_for_array'](loc,r,c,1,20)
                    record=ns['fault_record'](ns['location_by_id'](loc),'thermal',ns['row_by_id'](loc,row),cell,fault,level)
                    record['experiment_fault_id']=f'ros-selected-{pid}-{fault}'
                    records.append(record)
                st.session_state['random_fault_records']=records;st.session_state['panel_cell_fault_records']=[]
                _capture(ns,'Generated',new=True)
            except Exception as exc:
                restore_simulator(ns,saved);st.session_state['pv_scene_error']=str(exc)
        st.button('PREPARE SELECTED ROS FAULTS (REPLACE CURRENT)',on_click=prepare,disabled=not panels)


def write_local_inbox(mission):
    from pathlib import Path
    # Deliberately local fixed destination, not a remotely exposed file server.
    root=Path(__file__).resolve().parent.parent
    if root.name!='PV_Gazebo_WS' or not (root/'src/pv_inspection_demo/package.xml').is_file():
        raise ValueError('Local inbox is only available inside PV_Gazebo_WS')
    path=root/'outputs/local_ros_inbox.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(mission,indent=2),encoding='utf-8')
    temp.replace(path)
    return path
