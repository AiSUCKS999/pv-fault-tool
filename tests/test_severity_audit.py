"""Bounded pixel/scalar severity audit; saves evidence, not visibility claims."""
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from PIL import Image,ImageDraw
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pv_far_surface import generated_rgb_scene
from pv_real_farm_rgb_view import rgb_fault_preview,VISIBLE_RGB_EFFECTS


def metrics(a,b):
    delta=np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32)
    return {"mean_absolute_delta":float(np.abs(delta).mean()),"max_absolute_delta":float(np.abs(delta).max()),
        "changed_pixels":int(np.count_nonzero(np.any(delta!=0,axis=2))),
        "clipped_channel_fraction":float(np.mean((np.asarray(a)==0)|(np.asarray(a)==255)))}


@pytest.mark.parametrize("mode",["Generated","Real Farm"])
def test_severity_audit(mode):
    ns=runpy.run_path(str(ROOT/"tests/test_collection_acceptance.py"))["namespace"]()
    codes=[*ns["CLOSE_DIAGNOSTIC_FAULT_TYPES"],"SingleHotSpot","PartialShading"]
    out=ROOT/"artifacts/experiment_recovery";out.mkdir(exist_ok=True,parents=True)
    results={}
    if mode=="Generated":
        loc="grass_open";ns["qp_set"](location=loc,custom_layout=1,row_count=1,panels_per_row=1)
        original=ns["solar_render_image"](loc,None).convert("RGB")
        scalar,geo=ns["cached_native_thermal_master"](loc,ns["solar_render_cache_signature"](loc))
        row,cell=ns["default_cell_for_array"](loc,1,1,1,20)
        record=ns["fault_record"](ns["location_by_id"](loc),"thermal",ns["row_by_id"](loc,row),cell,"SingleHotSpot",7)
        target=ns["rendered_fault_cell_from_record"](loc,record,row,cell["id"],geo)
        box=tuple(target["module_bbox_px"][k] for k in ("x1","y1","x2","y2"))
        baseline=ns["finalize_solar_render_thermal_image"](loc,scalar,geo,row,cell["id"],[],"far").crop(box)
    else:
        ns["st"].session_state["real_farm_prepared_picker"]="image_05";ns["real_farm_catalog_selected"]()
        state=ns["st"].session_state["real_farm_state"];original=ns["st"].session_state["real_farm_image"]
        layers=ns["real_farm_camera_build_layers"](state,original);module=layers["accepted_modules"][0]
        observation=ns["real_farm_camera_new_observation"](state,original,module)
        observation.update(seed=421,local_uv=[.5,.5])
        box=ns["real_farm_close_context_bounds"](module,layers["camera_candidates"],(original.height,original.width),.25)
        baseline=ns["real_farm_camera_acquisition"](layers,[],"far").crop(box)
    for code in codes:
        values=[];pictures=[];refs=[]
        for level in (1,3,5,7,10):
            if mode=="Generated":
                record.update(fault_type=code,fault_scale=level)
                full=ns["finalize_solar_render_thermal_image"](loc,scalar,geo,row,cell["id"],[record],"far")
                far=full.crop(box)
                close=ns["finalize_solar_render_thermal_image"](loc,scalar,geo,row,cell["id"],[record],"close").crop(box)
                rgb=generated_rgb_scene(ns,original,loc,row,cell["id"],[record])[0].crop(box)
                scalar_metrics=None
            else:
                observation.update(diagnostic_fault_type=None if code=="SingleHotSpot" else code,severity=level)
                far=ns["real_farm_camera_acquisition"](layers,[observation],"far").crop(box)
                close=ns["real_farm_camera_acquisition"](layers,[observation],"close",box)
                rgb=rgb_fault_preview(ns,original,layers["accepted_modules"],[observation])[0].crop(box)
                field=ns["real_farm_camera_observation_field"](layers,[observation])
                scalar_metrics={"absolute_peak":float(np.max(abs(field-layers["master_scalar"]))),
                    "absolute_energy":float(np.sum(abs(field-layers["master_scalar"]),dtype=np.float64))}
                refs.append(deepcopy(observation.get("production_reference_ids",{})))
            png=ns["export_current_view_bytes"](far)
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(png))),np.asarray(far))
            values.append({"severity":level,"far":metrics(far,baseline),"close_vs_far":metrics(close,far),
                "rgb":metrics(rgb,original.crop(box)),"scalar":scalar_metrics})
            pictures.append((far.copy(),close.copy(),rgb.copy()))
        assert not np.array_equal(np.asarray(pictures[0][0]),np.asarray(pictures[-1][0])),(mode,code)
        energies=[v["far"]["mean_absolute_delta"] for v in values]
        assert all(a<b for a,b in zip(energies,energies[1:])),(mode,code,energies)
        if code in VISIBLE_RGB_EFFECTS:
            rgb_energy=[v["rgb"]["mean_absolute_delta"] for v in values]
            assert all(a<b for a,b in zip(rgb_energy,rgb_energy[1:])),(mode,code,rgb_energy)
        if code not in VISIBLE_RGB_EFFECTS:
            assert all(v["rgb"]["changed_pixels"]==0 for v in values),(mode,code)
        # Save all levels; rows: Far thermal, Close thermal, RGB. Scaling is
        # exclusively for the audit sheet, never passed back into the simulator.
        tile_w,tile_h=240,190
        sheet=Image.new("RGB",(5*tile_w,3*tile_h+32),(18,24,30));draw=ImageDraw.Draw(sheet)
        for i,level in enumerate((1,3,5,7,10)):
            draw.text((i*tile_w+8,8),f"Severity {level}",fill="white")
            for j,img in enumerate(pictures[i]):
                img.thumbnail((tile_w-8,tile_h-8));sheet.paste(img,(i*tile_w+4,32+j*tile_h))
        sheet.save(out/f"severity_{mode.replace(' ','_')}_{code}.png")
        results[code]={"levels":values,"reference_ids":refs,"endpoint_change":metrics(pictures[-1][0],pictures[0][0])}
    (out/f"severity_{mode.replace(' ','_')}.json").write_text(json.dumps(results,indent=2))
