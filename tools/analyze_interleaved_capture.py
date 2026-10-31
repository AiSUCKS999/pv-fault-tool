"""Offline assessment of an existing RGB/thermal recording, never display flashing.

Frames stream through OpenCV. Reference images must be the matching simulator
exports; color classification and edge registration are candidate evidence only.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def reduced(image,width=640):
    scale=min(1.,width/image.shape[1])
    return cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)


def histogram(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    value=cv2.calcHist([hsv],[0,1],None,[30,16],[0,180,0,256]).flatten()
    return value/(np.linalg.norm(value)+1e-9)


def classify(image,rgb_hist,thermal_hist,margin=.12):
    h=histogram(image);rgb=float(h@rgb_hist);thermal=float(h@thermal_hist)
    if max(rgb,thermal)<.3 or abs(rgb-thermal)<margin:return "mixed_or_uncertain",rgb,thermal
    return ("rgb" if rgb>thermal else "thermal"),rgb,thermal


def register(rgb,thermal):
    """Estimate on edges; never treat a successful fit as ground-truth alignment."""
    if rgb.shape!=thermal.shape:return {"status":"incompatible_frame_sizes"}
    def edge(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(cv2.Canny(gray,40,120).astype(np.float32)/255,(5,5),0)
    a,b=edge(rgb),edge(thermal)
    matrix=np.eye(2,3,dtype=np.float32)
    try:
        score,matrix=cv2.findTransformECC(a,b,matrix,cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,1e-4),None,5)
        shift=float(np.linalg.norm(matrix[:,2]));det=float(np.linalg.det(matrix[:,:2]))
        reliable=score>=.65 and shift<min(a.shape)*.1 and .8<det<1.2
        return {"status":"candidate_requires_review" if reliable else "unreliable", "edge_correlation":float(score),
                "affine_thermal_to_rgb_inverse_map":matrix.tolist(),"translation_pixels":shift,
                "scale_area":det,"alignment_verified":False}
    except cv2.error:return {"status":"registration_failed","alignment_verified":False}


def analyze(video,rgb_ref,thermal_ref,out,max_frames=1800,margin=.12):
    if max_frames<1 or max_frames>18000:raise ValueError("max_frames must be 1–18000")
    refs=[cv2.imread(str(p)) for p in (rgb_ref,thermal_ref)]
    if any(image is None for image in refs):raise ValueError("Both matching reference images are required")
    rh,th=[histogram(reduced(image)) for image in refs]
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened():raise ValueError("Cannot open recording")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fps=float(capture.get(cv2.CAP_PROP_FPS));rows=[];pairs=[];intervals=[];previous_rgb=None;pending=None
    try:
        for index in range(max_frames):
            ok,frame=capture.read()
            if not ok:break
            image=reduced(frame);kind,r,t=classify(image,rh,th,margin)
            timestamp=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            row={"frame":index,"timestamp_seconds":timestamp,"classification":kind,"rgb_score":r,"thermal_score":t}
            if not intervals or intervals[-1]["classification"]!=kind:
                intervals.append({"classification":kind,"first_frame":index,"last_frame":index,"start_seconds":timestamp,"end_seconds":timestamp})
            else:intervals[-1].update(last_frame=index,end_seconds=timestamp)
            if kind=="rgb":
                if previous_rgb is not None:row["rgb_motion_assessment"]=register(previous_rgb,image)
                previous_rgb=image;pending=(index,timestamp,image.copy())
            elif kind=="thermal" and pending is not None:
                rid,rt,rgb=pending;gap=timestamp-rt
                if 0<=gap<=.25:
                    alignment=register(rgb,image)
                    pair={"rgb_frame":rid,"thermal_frame":index,"gap_seconds":gap,**alignment}
                    pairs.append(pair)
                    # Bound extracted evidence; untouched frames, no warped data
                    # labelled as aligned ground truth.
                    if len(pairs)<=30:
                        cv2.imwrite(str(out/f"pair_{len(pairs):03}_rgb.png"),rgb)
                        cv2.imwrite(str(out/f"pair_{len(pairs):03}_thermal.png"),image)
                pending=None
            else:pending=None
            rows.append(row)
    finally:capture.release()
    report={"video_filename":Path(video).name,"reported_fps":fps,"frames_read":len(rows),"frame_limit":max_frames,
        "classification":"reference color similarity; unverified", "intervals":intervals,"pairs":pairs,
        "mixed_transition_frames":[r["frame"] for r in rows if r["classification"]=="mixed_or_uncertain"],
        "alignment_verified":False,"limitations":["Display transitions and rolling shutter can mix modalities within one frame.",
        "Consecutive frames can move; a shared camera does not ensure registration.","Cross-spectral edge matches require independent landmark review.",
        "Extracted previews are reduced to at most 640 pixels wide; original video is unchanged."]}
    (out/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (out/"frames.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["frame","timestamp_seconds","classification","rgb_score","thermal_score","rgb_motion_assessment"])
        writer.writeheader();writer.writerows(rows)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",type=Path);parser.add_argument("--rgb-reference",type=Path,required=True)
    parser.add_argument("--thermal-reference",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--max-frames",type=int,default=1800);parser.add_argument("--classification-margin",type=float,default=.12)
    args=parser.parse_args()
    report=analyze(args.video,args.rgb_reference,args.thermal_reference,args.out,args.max_frames,args.classification_margin)
    print(json.dumps({k:report[k] for k in ("video_filename","reported_fps","frames_read","alignment_verified")}))
