"""Presentation code updates refresh without changing scene IDs or ordinary state."""
from pathlib import Path
from playwright.sync_api import sync_playwright

def test_component_refreshes_changed_html_only():
    html=(Path(__file__).resolve().parents[1]/'pv_presentation_component/index.html').read_text(encoding='utf-8')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page();page.set_content(html)
            first='<button id="version">A</button><script>let clicks=0;document.getElementById("version").onclick=()=>clicks++;</script>'
            def update(content):page.evaluate("html=>window.postMessage({type:'streamlit:render',args:{signature:'unchanged-scene',html}},'*')",content)
            update(first);display=page.frame_locator('#display');display.locator('#version').click()
            update(first);assert display.locator('#version').evaluate('()=>clicks')==1
            update('<button id="version">B</button>');display.get_by_role('button',name='B',exact=True).wait_for()
        finally:browser.close()
