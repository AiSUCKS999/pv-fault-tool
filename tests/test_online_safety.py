from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pv_ros_mission as ros


def test_cloud_has_download_but_cannot_send_or_prepare_local_mission(monkeypatch):
    monkeypatch.setattr(ros,'local_ros_available',lambda root=None:False)
    scene={'mode':'Generated','source':{'rows':2,'columns':4,'identity':'grass_open'},'signature':'unit-test',
      'faults':[{'id':'unit-fixture','row':1,'array':1,'module':1,'fault_type':'CellCracking'}]}
    code='''import streamlit as st
from pv_ros_mission import render_export,render_prepare
render_prepare({'st':st},'Generated')
render_export({'st':st},SCENE)
'''
    app=AppTest.from_string(code.replace('SCENE',repr(scene)),default_timeout=20).run()
    assert not app.exception
    send=next(b for b in app.button if b.label=='Send Inspection Mission')
    assert send.disabled
    assert any(b.label=='EXPORT ROS MISSION JSON' for b in app.get('download_button'))
    assert ros.read_results({'mission_id':'unit'}) is None
    import pytest
    with pytest.raises(ValueError,match='local-only'):ros.attach_textures({}, {})


def test_local_workspace_requires_actual_ros_package(tmp_path):
    root=tmp_path/'PV_Gazebo_WS';root.mkdir()
    assert not ros.local_ros_available(root)
    path=root/'src/pv_inspection_demo/package.xml';path.parent.mkdir(parents=True);path.write_text('<package/>')
    assert ros.local_ros_available(root)
