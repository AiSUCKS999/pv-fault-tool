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
        source=deepcopy(f)
        source['native_array']={'row':1,'column':c+2} if scene['source'].get('identity')=='grass_open' and r==2 and c in (3,4) else {'row':r,'column':c}
        target['fault_ids'].append(oid);target['source_faults'].append(source)
    mission={'schema':SCHEMA,'map_id':'pv_farm_2x4_v1','frame_id':'pv_farm','units':'metres',
        'mapping':'logical Generated array -> one surrogate Gazebo panel; source module/cell IDs retained; image pixels are not world coordinates',
        'simulated':True,'physical_ground_truth_verified':False,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source':deepcopy(scene['source']),'source_revision':scene['signature'],'targets':list(panels.values())}
    mission['mission_id']='mission-'+hashlib.sha256(json.dumps(mission,sort_keys=True).encode()).hexdigest()[:20]
    return mission


def local_ros_available(root=None):
    from pathlib import Path
    root=Path(root) if root is not None else Path(__file__).resolve().parent.parent
    return root.name=='PV_Gazebo_WS' and (root/'src/pv_inspection_demo/package.xml').is_file()

def render_export(ns,scene):
    st=ns['st']
    with st.expander('ROS mission export · local simulator only',expanded=False):
        local=local_ros_available()
        if not local:st.caption('Download a mission for your local simulator. This website cannot control your computer.')
        st.caption('Generated 2x4 arrays map to eight Gazebo panels. Module IDs remain provenance; no image-to-world calibration is claimed.')
        ids=[f['id'] for f in scene['faults']]
        if 'ros_mission_fault_ids' in st.session_state:
            st.session_state['ros_mission_fault_ids']=[i for i in st.session_state['ros_mission_fault_ids'] if i in ids]
        chosen=st.multiselect('Inspection faults',ids,default=ids,key='ros_mission_fault_ids',
            format_func=lambda oid:next(f"{oid} | row {f.get('row')} array {f.get('array')} module {f.get('module')} | {f['fault_type']}" for f in scene['faults'] if f['id']==oid))
        try:
            mission=export_mission(scene,chosen)
            st.download_button('EXPORT ROS MISSION JSON',json.dumps(mission,indent=2),file_name='pv_mission.json',mime='application/json')
            if st.button('Send Inspection Mission',disabled=not local):
                try:
                    attach_textures(scene,mission)
                    write_local_inbox(mission)
                    st.session_state['ros_sent_mission']=mission
                    st.success('Mission queued for the foreground ROS terminal')
                except (OSError,ValueError) as exc:st.error(str(exc))
        except ValueError as exc:st.info(str(exc))
        render_results(st)


def render_prepare(ns,mode):
    st=ns['st']
    if mode!='Generated' or not local_ros_available():return
    loc=ns['selected_location']()['id']
    if (ns['custom_row_count'](loc),ns['custom_panels_per_row'](loc))!=(2,4):return
    with st.expander('Prepare selected ROS inspection faults (development)',expanded=False):
        st.caption('Source asset is 6+2 arrays: Gazebo R2C3/R2C4 use native R1C5/R1C6. Physical source polygons are unchanged.')
        # Read explicitly mapped existing geometry; no invented polygons.
        native=ns['cached_native_thermal_master'](loc,ns['solar_render_cache_signature'](loc))
        if native is None:st.error('Native geometry unavailable');return
        supported=[f'R{r}C{c}' for r in (1,2) for c in range(1,5) if (r,c) in native[1]]
        missing=[f'R{r}C{c}' for r in (1,2) for c in range(1,5) if (r,c) not in native[1]]
        if missing:st.warning('Existing renderer cannot resolve these source targets: '+', '.join(missing))
        panels=st.multiselect('Gazebo panels to fault (explicit source map)',supported,default=[p for p in ['R1C1','R2C3','R2C4'] if p in supported],key='ros_prepare_panels')
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
    raw=json.dumps(mission,indent=2).encode('utf-8')
    temp.write_bytes(raw)
    saved=root/'outputs/sent_missions'/str(mission['mission_id']+'.json')
    saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(raw)
    temp.replace(path)
    return path


def attach_textures(scene,mission):
    """Rectify actual rendered module pixels; never make up fault imagery."""
    from pathlib import Path
    import cv2,numpy as np
    if not local_ros_available():raise ValueError('ROS imagery export is local-only')
    root=Path(__file__).resolve().parent.parent
    folder=root/'outputs/mission_media'/mission['mission_id'];folder.mkdir(parents=True,exist_ok=True)
    rgb=np.array(scene['far']['rgb'])
    for target in mission['targets']:
        for fault in target['source_faults']:
            name=f"{target['panel_id']}-M{fault['module']}.png"
            quad=np.float32(fault['polygon_rgb'])
            dest=np.float32([[0,0],[255,0],[255,399],[0,399]])
            # A texture is a rectified source projection, not new image detail.
            patch=cv2.warpPerspective(rgb,cv2.getPerspectiveTransform(quad,dest),(256,400))
            cv2.imwrite(str(folder/name),cv2.cvtColor(patch,cv2.COLOR_RGB2BGR))
            fault['gazebo_texture']='mission_media/'+mission['mission_id']+'/'+name


def read_results(mission):
    if not local_ros_available():return None
    from pathlib import Path
    root=Path(__file__).resolve().parent.parent
    path=root/'outputs/mission_status.json'
    if not path.exists():return None
    result=json.loads(path.read_text(encoding='utf-8'))
    if result.get('mission_id')!=mission['mission_id']:return None
    # Only show results associated with the exact sent mission contents.
    snapshot=root/'outputs/sent_missions'/(mission['mission_id']+'.json')
    if not snapshot.exists():snapshot=root/'outputs/executing_mission.json'
    if not snapshot.exists():return None
    raw=snapshot.read_bytes()
    if json.loads(raw)!=mission or result.get('mission_sha256')!=hashlib.sha256(raw).hexdigest():return None
    return result


def render_results(st):
    from pathlib import Path
    mission=st.session_state.get('ros_sent_mission')
    if not mission:return
    st.button('Refresh Inspection Results')
    try:result=read_results(mission)
    except (OSError,ValueError):result=None
    if result is None:st.info('Waiting for this mission in the foreground ROS terminal');return
    survey_path=Path(__file__).resolve().parent.parent/'outputs/survey_status.json'
    if survey_path.exists():
        survey=json.loads(survey_path.read_text(encoding='utf-8'))
        if survey.get('mission_id')==mission['mission_id'] and survey.get('mission_sha256')==result.get('mission_sha256'):
            st.write('Survey robot: '+survey['status'])
            st.write('Surveyed arrays: '+', '.join(survey.get('visited',[])))
            st.metric('Survey Gazebo travel',f"{survey.get('gazebo_pose_travel_m',0):.2f} m")
            with st.expander('Survey observations (simulated)',expanded=False):
                st.caption('ROS 2 local coordination · simulated manifest screening')
                output=(Path(__file__).resolve().parent.parent/'outputs').resolve()
                for item in survey.get('inspection_images',[]):
                    if item.get('path'):
                        picture=(output/item['path']).resolve()
                        if output in picture.parents and picture.is_file():st.image(str(picture),caption='Survey '+item['panel'])

    st.write(result['status'])
    st.write('Visited: '+', '.join(result.get('visited',[])))
    st.metric('Actual Gazebo travel',f"{result.get('gazebo_pose_travel_m',0):.2f} m")
    from pathlib import Path
    root=(Path(__file__).resolve().parent.parent/'outputs').resolve()
    for item in result.get('inspection_images',[]):
        if item.get('path'):
            path=(root/item['path']).resolve()
            if root in path.parents and path.is_file():st.image(str(path),caption=item['panel']+' · simulated Gazebo camera')
    st.download_button('Download Inspection Results',json.dumps(result,indent=2),file_name=mission['mission_id']+'-results.json')
