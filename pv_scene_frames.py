"""Acquire one effective scene through the existing renderers for every consumer.

Coordinates are native source pixels. Close is an acquisition/crop of the same
fault instances, never a second diagnosis or a replacement of the Far scene.
"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from pv_experiment_capture import _real_layers, actual_state
from pv_far_surface import generated_rgb_scene, visibility
from pv_presentation import image_data
from pv_real_farm_rgb_view import VISIBLE_RGB_EFFECTS, rgb_fault_preview, view_bounds


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def effective_real_observations(state):
    """Camera state is authoritative when present, including an explicit clear.

    Older projects without a camera collection keep their legacy observations.
    A retained legacy collection is never silently combined or discarded.
    """
    return state.get("camera_observations", state.get("observations", []))


def current_signature(ns, mode):
    if mode == "Generated":
        loc=ns["selected_location"]()["id"]
        return digest([mode, loc, ns["solar_render_cache_signature"](loc),
            ns["thermal_records_for_distance"](ns["records_for_location"](loc,"thermal"),"far")])
    state=ns["st"].session_state.get("real_farm_state", {})
    # Render audits are outputs, not scene mutations; exclude them from keys.
    observations=[{k:o.get(k) for k in ("observation_id","module_id","severity","seed","local_uv",
        "diagnostic_fault_type","observation_type","source_rgb_pixel_sha256")} for o in effective_real_observations(state)]
    return digest([mode,state.get("source_rgb_pixel_sha256"),state.get("modules"),observations])


def _box(polygon):
    points=np.asarray(polygon)
    return [float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]


def _boxes(faults, bounds=None):
    result=[]
    x,y= bounds[:2] if bounds else (0,0)
    for fault in faults:
        a,b,c,d=fault["module_bbox"]
        if bounds and (c<=bounds[0] or a>=bounds[2] or d<=bounds[1] or b>=bounds[3]):continue
        result.append({"id":fault["id"],"bbox":[a-x,b-y,c-x,d-y],"kind":"module",
                       "class":fault["fault_type"],"severity":fault["severity"]})
    return result


def acquire_scene(ns, mode):
    st=ns["st"]
    signature=current_signature(ns,mode)
    cached=st.session_state.get("pv_scene_frames_"+mode)
    if cached and cached[0]==signature:return cached[1]
    faults=[];close={};geometry=[]
    if mode=="Generated":
        loc=ns["selected_location"]()["id"]
        row=ns["panels"](loc)[0]["id"];cell=ns["cells_for_row"](loc,row)[0]["id"]
        records=ns["thermal_records_for_distance"](ns["records_for_location"](loc,"thermal"),"far")
        original=ns["solar_render_image"](loc,None)
        if original is None:raise ValueError("Exact approved source layout is unavailable")
        original=original.convert("RGB")
        native=ns["cached_native_thermal_master"](loc,ns["solar_render_cache_signature"](loc))
        if native is None:raise ValueError("Native source geometry is unavailable")
        scalar,arrays=native
        rgb,report=generated_rgb_scene(ns,original,loc,row,cell,records)
        thermal=ns["finalize_solar_render_thermal_image"](loc,scalar,arrays,row,cell,records,"far")
        healthy=ns["finalize_solar_render_thermal_image"](loc,scalar,arrays,row,cell,[],"far")
        close_thermal=ns["finalize_solar_render_thermal_image"](loc,scalar,arrays,row,cell,records,"close") if records else thermal
        groups={}
        for record in records:
            target=ns["rendered_fault_cell_from_record"](loc,record,row,cell,arrays)
            if target is None:raise ValueError("An active fault has no native target geometry")
            oid=record.get("experiment_fault_id") or record.get("far_surface_group_id") or "G-"+digest([loc,record["row_id"],record["target_cell_id"]])[:12]
            if oid not in groups:
                logical=ns["cell_by_id"](loc,record["row_id"],record["target_cell_id"])
                polygon=deepcopy(target["active_module_polygon"])
                groups[oid]={"id":oid,"fault_type":record["fault_type"],"severity":record["fault_scale"],
                    "module_id":f"{record['row_id']}-A{logical['panel_number']}-M{logical['module_number_in_panel']}",
                    "row":ns["row_number_from_id"](loc,record["row_id"]),"array":logical["panel_number"],
                    "module":logical["module_number_in_panel"],"cell_ids":[],"polygon_rgb":polygon,
                    "module_bbox":_box(polygon),"placement_seed":record.get("placement_seed"),
                    "seed_behavior":"Placement seeded; existing Generated morphology uses its internal target seed",
                    "physical_ground_truth_verified":False}
            groups[oid]["cell_ids"].append(record["target_cell_id"])
        faults=list(groups.values())
        # Existing export geometry supplies all occupied arrays/modules and masks.
        regions=ns["array_regions"](loc,original)
        _,geometry,mask=ns["generated_export_geometry"](loc,original,regions,original.size,records)
        if not isinstance(mask,Image.Image):mask=Image.fromarray(np.uint8(mask))
        source={"identity":loc,"size":list(original.size),"rows":ns["custom_row_count"](loc),
                "columns":ns["custom_panels_per_row"](loc),"occupied_arrays":len(regions),
                "source_pixel_sha256":hashlib.sha256(original.tobytes()).hexdigest()}
        for fault in faults:
            a,b,c,d=fault["module_bbox"];pad=max(12,int(max(c-a,d-b)*.3))
            bounds=(max(0,int(a)-pad),max(0,int(b)-pad),min(original.width,int(c)+pad+1),min(original.height,int(d)+pad+1))
            close[fault["id"]]={"original":original.crop(bounds),"rgb":rgb.crop(bounds),"thermal":close_thermal.crop(bounds),"bounds":bounds}
    else:
        state=st.session_state["real_farm_state"];original=st.session_state["real_farm_image"]
        observations=effective_real_observations(state)
        layers=_real_layers(ns);geometry=layers["accepted_modules"]
        lookup={m["module_id"]:m for m in geometry}
        if any(o["module_id"] not in lookup for o in observations):raise ValueError("An active observation has no supported thermal target")
        rgb,report=rgb_fault_preview(ns,original,geometry,observations)
        healthy=ns["real_farm_camera_acquisition"](layers,[],"far")
        thermal=ns["real_farm_camera_acquisition"](layers,observations,"far")
        for observation in observations:
            module=lookup[observation["module_id"]]
            polygon=deepcopy(module["active_polygon_rgb"])
            oid=observation["observation_id"]
            faults.append({"id":oid,"module_id":module["module_id"],
                "fault_type":observation.get("diagnostic_fault_type") or observation.get("observation_type","SingleHotSpot"),
                "severity":observation["severity"],"polygon_rgb":polygon,"module_bbox":_box(polygon),
                "center_rgb_px":deepcopy(observation.get("center_rgb_px")),"local_uv":deepcopy(observation.get("local_uv")),
                "applied_seed":observation.get("seed"),"physical_ground_truth_verified":False,
                "scalar_footprint":deepcopy(observation.get("footprint_rgb",{})),
                "reference_ids":deepcopy(observation.get("production_reference_ids",{}))})
            bounds=view_bounds(ns,original,module,layers["camera_candidates"],"Close View")
            close[oid]={"original":original.crop(bounds),"rgb":rgb.crop(bounds),
                "thermal":ns["real_farm_camera_acquisition"](layers,observations,"close",bounds),"bounds":bounds}
        mask=Image.fromarray(layers["pv_module_union_mask"].astype(np.uint8)*255) if layers["pv_module_union_mask"].max()<=1 else Image.fromarray(np.uint8(layers["pv_module_union_mask"]))
        source={"identity":state.get("prepared_catalog_id","uploaded"),"size":list(original.size),
            "source_pixel_sha256":ns["real_farm_source_identity"](original),"filename":state.get("original_rgb_filename"),
            "geometry_signature":ns["real_farm_geometry_signature"](state["modules"])}
    if not (original.size==rgb.size==thermal.size==healthy.size):raise ValueError("RGB/thermal native acquisition sizes differ")
    for fault in faults:
        fault["rgb_classification"]="Surface effect" if fault["fault_type"] in VISIBLE_RGB_EFFECTS else "Thermal only"
        # Measured differences are evidence, never a visibility acceptance gate.
        box=tuple(map(int,fault["module_bbox"]))
        fault["acquisition_metrics"]={"RGB":visibility(original.crop(box),rgb.crop(box)),
            "thermal":visibility(healthy.crop(box),thermal.crop(box))}
    result={"mode":mode,"source":source,"faults":faults,"geometry":deepcopy(geometry),"mask":mask,
        "far":{"original":original,"rgb":rgb,"thermal":thermal,"healthy_thermal":healthy,"bounds":None},
        "close":close,"rgb_report":report,"signature":signature}
    st.session_state["pv_scene_frames_"+mode]=(current_signature(ns,mode),result)
    return result


def presentation_payload(ns, acquired):
    def encode(frame):
        return {**{k:image_data(ns["export_current_view_bytes"](frame[k])) for k in ("original","rgb","thermal")},
            "width":frame["rgb"].width,"height":frame["rgb"].height,"bounds":frame["bounds"],
            "boxes":_boxes(acquired["faults"],frame["bounds"])}
    return {"identity":acquired["source"]["identity"],"revision":acquired["signature"],
            "faults":acquired["faults"],"far":encode(acquired["far"]),
            "close":{oid:encode(frame) for oid,frame in acquired["close"].items()},
            "limitation":"Native source resolution; scaling adds no detail"}


def presentation_panel(ns, mode):
    from pv_presentation import show_presentation
    st=ns["st"]
    if st.button("PREPARE PRESENTATION",key="present_prepare_"+mode):
        try:
            frames=acquire_scene(ns,mode)
            st.session_state["pv_scene_presentation_"+mode]=(frames["signature"],presentation_payload(ns,frames))
        except (ValueError,RuntimeError) as exc:st.error(str(exc))
    saved=st.session_state.get("pv_scene_presentation_"+mode)
    if saved:
        if saved[0]!=current_signature(ns,mode):st.warning("Scene changed. Prepare presentation to refresh the display.")
        else:show_presentation(ns,saved[1],key=mode)
