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
#controls{display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:8px;background:#0c1520}
button,select{font:inherit;color:#eee;background:#233446;border:1px solid #536579;border-radius:5px;padding:9px;cursor:pointer}
button[aria-pressed=true]{background:#17677b}#status{padding:4px 8px;color:#bbc9d5;font-size:12px}
#screen{position:relative;width:100%;height:calc(100% - 120px);overflow:hidden;background:#000;outline:none}
#screen:fullscreen{width:100vw;height:100vh;cursor:none}body.presentation #screen{position:fixed;inset:0;width:100vw;height:100vh}
body.presentation #controls,body.presentation #status{display:none}
#plane{position:absolute;overflow:hidden}#plane img{position:absolute;width:100%;height:100%;inset:0}
#compare{clip-path:inset(0 50% 0 0)}#labels{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
#labels [data-fault]{cursor:pointer}#labels text{paint-order:stroke;stroke:#001018;stroke-width:3px;fill:#fff;font-family:system-ui}
</style></head><body><div id="controls">
<button id="fullscreen">PRESENT FULLSCREEN</button><button id="open">OPEN DISPLAY</button>
<button data-distance="FAR">FAR</button><button data-distance="CLOSE">CLOSE</button>
<button data-mode="original">ORIGINAL RGB</button><button data-mode="rgb">FAULTED RGB</button><button data-mode="thermal">THERMAL</button><button data-mode="compare">COMPARE</button>
<button id="previous">PREVIOUS FAULT</button><button id="next">NEXT FAULT</button>
<select id="fit" aria-label="Display scaling"><option value="fit">FIT · letterbox</option><option value="fill">FILL · crop edges</option></select>
<button id="boxes">SHOW FAULT BOXES</button></div><div id="status" role="status"></div>
<div id="screen" tabindex="0"><div id="plane"><img id="base" draggable="false"><img id="compare" draggable="false"><svg id="labels"></svg></div></div>
<script>const INITIAL=""" + data + ";const CONTROLLER=" + json.dumps(controller) + ";const CHANNEL=" + json.dumps(channel) + ";\n" + script + "</script></body></html>"


def image_data(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def show_presentation(ns, payload, key="scene"):
    import streamlit.components.v1 as components
    components.html(presentation_html(payload, channel="pv-display-" + key), height=540, scrolling=False)
    with ns["st"].expander("Presentation help", expanded=False):
        ns["st"].write("F: Far · C: Close · R: RGB · T: thermal · O: original · V: compare · arrows: previous/next fault · B: boxes · X: fit/fill · Enter: fullscreen · Esc: exit. Move the separate display window to the projector; control it here. Compare: move the pointer to slide the divider.")
        ns["st"].caption("FIT retains the whole image with letterboxing. FILL crops edges without distortion. Display scaling adds no source detail. Browser fullscreen may require Enter in the display window.")
        ns["st"].download_button("Download standalone presentation", presentation_html(payload),
            file_name="pv_presentation.html", mime="text/html", key="presentation_download_"+key)
