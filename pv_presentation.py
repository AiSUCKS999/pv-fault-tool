"""Clean, standalone presentation of already acquired simulator frames."""
import base64
import json
from pathlib import Path


def presentation_html(payload, *, controller=True, channel="pv-presentation"):
    # No dimensions are baked into the viewport. Each image retains its native
    # pixels; contain/cover choose letterboxing/cropping, never stretching.
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    script = Path(__file__).with_name("pv_presentation.js").read_text(encoding="utf-8")
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PV presentation</title><style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;background:#000;color:#eee;font:14px system-ui;overflow:hidden}
body{display:flex;flex-direction:column}#controls{flex:none;padding:8px;background:#0c1520}
#toolbar,#advanced-content{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
details{margin-top:6px}summary{cursor:pointer;padding:6px;color:#bdcbd9}#advanced-content{padding:8px 0}
#notice{padding:4px 8px;color:#ffd28a}
button,select{font:inherit;color:#eee;background:#233446;border:1px solid #536579;border-radius:5px;padding:9px;cursor:pointer}
button[aria-pressed=true]{background:#17677b}#status{padding:4px 8px;color:#bbc9d5;font-size:12px}
#screen{position:relative;width:100%;flex:1;min-height:0;overflow:hidden;background:#000;outline:none}
#screen:fullscreen{width:100vw;height:100vh;cursor:none}body.presentation #screen{position:fixed;inset:0;width:100vw;height:100vh}
body.presentation #controls,body.presentation #status{display:none}
#gray{display:none;position:absolute;inset:0;background:rgb(128,128,128)}
#plane{position:absolute;overflow:hidden}#plane img{position:absolute;width:100%;height:100%;inset:0}
#compare{clip-path:inset(0 50% 0 0)}#labels{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
#labels [data-fault]{cursor:pointer}#labels text{fill:#fff;font-family:system-ui;font-weight:600}
</style></head><body><div id="controls"><div id="toolbar">
<button id="fullscreen">PRESENT FULLSCREEN</button>
<button data-distance="FAR">FAR</button><button data-distance="CLOSE">CLOSE</button>
<button data-mode="original">ORIGINAL RGB</button><button data-mode="rgb">FAULTED RGB</button><button data-mode="thermal">THERMAL</button>
<button id="boxes" aria-pressed="false">BOUNDING BOX</button>
<button id="previous" aria-label="Previous fault">PREVIOUS</button><button id="next" aria-label="Next fault">NEXT</button>
</div><details id="advanced"><summary>Advanced Options</summary><div id="advanced-content">
<button id="open">OPEN DISPLAY</button><button data-mode="compare">COMPARE</button>
<select id="fit" aria-label="Display scaling"><option value="fit">FIT · letterbox</option><option value="fill">FILL · crop edges</option></select>
<button id="clean">CLEAN FILMING</button>
<select id="calibration" aria-label="LED Calibration Mode"><option value="off">LED calibration: off</option><option value="gray">Gray</option><option value="detail">PV detail (original RGB)</option><option value="fault">Fixed actual fault</option></select>
<label>Zoom <input id="zoom" aria-label="Presentation zoom" type="range" min="1" max="3" step="0.05" value="1"></label>
</div><div id="status" role="status"></div></details></div><div id="notice" role="alert" hidden></div>
<div id="screen" tabindex="0"><div id="gray"></div><div id="plane"><img id="base" draggable="false"><img id="compare" draggable="false"><svg id="labels"></svg></div></div>
<script>const INITIAL=""" + data + ";const CONTROLLER=" + json.dumps(controller) + ";const CHANNEL=" + json.dumps(channel) + ";\n" + script + "</script></body></html>"


def image_data(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def show_presentation(ns, payload, key="scene"):
    import streamlit.components.v1 as components
    viewer=components.declare_component("pv_scene_display",path=str(Path(__file__).with_name("pv_presentation_component")))
    value=viewer(html=presentation_html(payload, channel="pv-display-" + key),
        signature=payload["revision"],key="scene_display_"+key,default=None)
    if isinstance(value,dict) and value.get("signature")==payload["revision"]:
        ns["st"].session_state["pv_scene_view_"+key]=value
        from pv_scene_log import SceneJournal
        SceneJournal(ns["st"].session_state).record_presentation(key,value)
    with ns["st"].expander("Advanced Options · Export & Help", expanded=False):
        ns["st"].caption("Keys: F/C Far/Close · R/T RGB/Thermal · O Original · B Bounding Box · arrows previous/next · Enter fullscreen. FIT keeps the whole image; FILL crops without distortion.")
        ns["st"].download_button("Download standalone presentation", presentation_html(payload),
            file_name="pv_presentation.html", mime="text/html", key="presentation_download_"+key)
