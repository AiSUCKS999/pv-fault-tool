# Fault names, aliases, visibility rules, classifier sanity check, and global setup.
import base64
import html
import json
import math
import random
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ANNOTATION_DIR = ROOT / "annotations"
PATCH_BANK_DIR = ASSET_DIR / "real_patch_bank"
FAULT_PATCH_DIR = PATCH_BANK_DIR / "faults"
PANEL_TEXTURE_DIR = PATCH_BANK_DIR / "panel_textures"
ACTUAL_PANEL_MODEL_TEXTURE = ASSET_DIR / "__disabled_actual_panel_model.png"
BASE_SIZE = (763, 402)
MAX_LAYOUT_ROWS = 30
MAX_PANELS_PER_ROW = 16
CUSTOM_MODULE_CELL_COLS = 6
CUSTOM_MODULE_CELL_ROWS = 2
PV_MODULE_MODEL_NAME = "PV module"
PV_MODULE_WIDTH_MM = 676
PV_MODULE_HEIGHT_MM = 780
CUSTOM_MODULE_ASPECT_RATIO = 1.62
CUSTOM_MODULE_GAP_RATIO = 0.055
DEFAULT_FAULT_SCALE = 8

FAULT_TYPES = {
    "PartialShading": "Root-cause fault: surface obstruction or partial shading that produces a local thermal hotspot symptom",
    "SoilingDust": "Root-cause fault: dust or soiling deposited on the panel surface",
    "CellCracking": "Root-cause fault: cracked/damaged cell path that can create localized heating",
    "SingleHotSpot": "Internal symptom label: one localized thermal hotspot pattern",
    "MultiHotSpot": "Internal symptom label: multiple localized thermal hotspot patterns",
    "SingleDiode": "Root-cause fault: bypass diode issue with a diode-region thermal pattern",
    "MultiDiode": "Root-cause fault: multiple bypass diode issues with repeated diode-region thermal patterns",
    "SingleByPassed": "Root-cause fault: one bypassed substring or section thermal pattern",
    "MultiByPassed": "Root-cause fault: multiple bypassed substrings or section thermal patterns",
    "StringOpenCircuit": "Root-cause fault: inactive/open string thermal contrast pattern",
    "StringReversedPolarity": "Root-cause fault: reversed string polarity thermal contrast pattern",
    "DB": "Removed legacy label: diode/bypass anomaly was too vague for the current tool",
    "DL": "Removed legacy label: damaged/disconnected line was too vague for the current tool",
    "TB": "Removed legacy label: thermal block/terminal box was too vague for the current tool",
}

FAULT_DISPLAY_NAMES = {
    "PartialShading": "Surface Obstruction / Partial Shading",
    "SoilingDust": "Dust / Soiling",
    "CellCracking": "Cell Cracking",
    "SingleHotSpot": "Thermal Hotspot Symptom",
    "MultiHotSpot": "Multiple Thermal Hotspot Symptoms",
    "SingleDiode": "Bypass Diode Fault",
    "MultiDiode": "Multiple Diode Faults",
    "SingleByPassed": "Bypassed Substring / Section",
    "MultiByPassed": "Multiple Bypassed Sections",
    "StringOpenCircuit": "String Open Circuit",
    "StringReversedPolarity": "String Reversed Polarity",
    "DB": "Diode / Bypass Fault",
    "DL": "Disconnected / Damaged Line",
    "TB": "Thermal Block / Terminal Box Fault",
}

HOTSPOT_FAULTS = {"SingleHotSpot", "MultiHotSpot"}
HIDDEN_SYMPTOM_FAULTS = HOTSPOT_FAULTS
LOW_CONFIDENCE_LEGACY_FAULTS = {"DB", "DL", "TB"}
HIDDEN_MULTICELL_FAULTS = {"MultiDiode", "MultiByPassed"}
HIDDEN_TOOL_FAULTS = HIDDEN_SYMPTOM_FAULTS | LOW_CONFIDENCE_LEGACY_FAULTS | HIDDEN_MULTICELL_FAULTS
SURFACE_OBSTRUCTION_FAULTS = {"PartialShading"}
SURFACE_OBSTRUCTION_LOCATIONS = {"scenario_1", "agri_rows"}
SOILING_FAULTS = {"SoilingDust"}
CRACKING_FAULTS = {"CellCracking"}
PREFERRED_CELL_CRACKING_PATCH_SUFFIXES = {
    "001", "024", "025", "026", "027", "028", "029", "030", "031", "032", "033", "034", "035",
    "046", "047", "048", "049", "050", "051", "052", "053",
    "064", "065", "066", "067", "071", "072", "084", "085", "088", "089", "090",
}
DIODE_FAULTS = {"SingleDiode", "MultiDiode"}
BYPASSED_FAULTS = {"SingleByPassed", "MultiByPassed"}
STRING_FAULTS = {"StringOpenCircuit", "StringReversedPolarity"}
LEGACY_LINE_FAULTS = {"DL"}
THERMAL_BLOCK_FAULTS = {"TB"}
USER_SELECTABLE_FAULT_TYPES = [
    fault_type for fault_type in FAULT_TYPES
    if fault_type not in HIDDEN_TOOL_FAULTS
]


def fault_available_in_location(location_id, fault_type):
    fault_type = canonical_fault_type(fault_type)
    if fault_type in SURFACE_OBSTRUCTION_FAULTS:
        return location_id in SURFACE_OBSTRUCTION_LOCATIONS
    return fault_type in USER_SELECTABLE_FAULT_TYPES


def selectable_fault_types_for_location(location_id):
    return [
        fault_type for fault_type in USER_SELECTABLE_FAULT_TYPES
        if fault_available_in_location(location_id, fault_type)
    ]

PATCH_FALLBACKS = {
    "MultiHotSpot": ["SingleHotSpot"],
    "MultiDiode": ["SingleDiode"],
    "SingleByPassed": ["SingleDiode"],
    "MultiByPassed": ["SingleDiode"],
    "StringReversedPolarity": ["StringOpenCircuit"],
    "SoilingDust": ["PartialShading"],
}

FAULT_ALIASES = {
    "Hotspot": "SingleHotSpot",
    "Partial shading": "PartialShading",
    "Surface obstruction": "PartialShading",
    "Soiling / dust": "SoilingDust",
    "Dust": "SoilingDust",
    "Dust shading": "SoilingDust",
    "Cell cracking": "CellCracking",
    "Bypass diode": "SingleDiode",
    "Electrical / open circuit": "StringOpenCircuit",
}

CLASSIFIER_CHECKPOINT = Path(
    r"D:\capstone2_slow_robot\runs\slow_robot_classifier\slow_robot_v1_merged6_resnet34_hardfix\best.pt"
)
CLASSIFIER_DISPLAY_NAMES = {
    "cracking": "Cracking",
    "diode_fault": "Diode Fault",
    "healthy": "Healthy",
    "hotspot_fault": "Thermal Hotspot Symptom",
    "offline_module": "Offline Module",
    "surface_obstruction": "Surface Obstruction / Partial Shading",
}


def canonical_fault_type(fault_type):
    return FAULT_ALIASES.get(fault_type, fault_type)


def fault_display_name(fault_type):
    fault_type = canonical_fault_type(fault_type)
    return FAULT_DISPLAY_NAMES.get(fault_type, fault_type)


@st.cache_resource(show_spinner=False)
def load_scene_classifier():
    if not CLASSIFIER_CHECKPOINT.exists():
        return None, [], f"Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}"
    try:
        import torch
        from torchvision import models
    except Exception as exc:
        return None, [], f"Classifier dependencies are not available: {exc}"

    try:
        checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu")
        class_names = list(checkpoint.get("class_names", []))
        model_args = checkpoint.get("args", {})
        arch = model_args.get("arch", "resnet34")
        if arch == "resnet18":
            model = models.resnet18(weights=None)
        else:
            model = models.resnet34(weights=None)
        model.conv1 = torch.nn.Conv2d(
            1,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(model.fc.in_features, len(class_names)),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model, class_names, ""
    except Exception as exc:
        return None, [], f"Could not load classifier: {exc}"


def classify_generated_scene(image):
    model, class_names, error = load_scene_classifier()
    if error:
        return {"available": False, "error": error, "top": [], "fault_like": 0.0}
    if model is None or not class_names:
        return {"available": False, "error": "Classifier is unavailable.", "top": [], "fault_like": 0.0}

    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"Classifier dependencies are not available: {exc}", "top": [], "fault_like": 0.0}

    img = ImageOps.fit(image.convert("L"), (288, 288), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    order = np.argsort(-probs)
    top = [
        {
            "class": class_names[int(idx)],
            "label": CLASSIFIER_DISPLAY_NAMES.get(class_names[int(idx)], class_names[int(idx)]),
            "confidence": float(probs[int(idx)]),
        }
        for idx in order[: min(5, len(order))]
    ]
    healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
    healthy_conf = float(probs[healthy_idx]) if healthy_idx is not None else 0.0
    return {
        "available": True,
        "error": "",
        "top": top,
        "fault_like": max(0.0, min(1.0, 1.0 - healthy_conf)),
        "healthy_confidence": healthy_conf,
    }


def render_classifier_check(image):
    result = classify_generated_scene(image)
    if not result["available"]:
        st.warning(result["error"])
        return

    top = result["top"]
    if not top:
        st.warning("Classifier returned no predictions.")
        return

    winner = top[0]
    st.metric("Classifier top read", winner["label"], f"{winner['confidence']:.0%} confidence")
    st.progress(result["fault_like"], text=f"Fault-like score: {result['fault_like']:.0%}")
    st.caption(
        "This is the saved ResNet34 classifier judging the whole generated thermal scene. "
        "It is a scene-level sanity check for symptoms/root-cause cues; it does not draw boxes, "
        "locate the fault, or prove the physical cause by itself."
    )
    st.table(
        [
            {
                "rank": rank,
                "class": item["label"],
                "confidence": f"{item['confidence']:.1%}",
            }
            for rank, item in enumerate(top, start=1)
        ]
    )
