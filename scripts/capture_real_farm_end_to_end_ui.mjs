import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1")), "..");
const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const appPort = process.env.REAL_FARM_PORT || "8502";
const uploadPath = path.join(root, "artifacts", "real_farm", "uploads", "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg");
const outputDir = path.join(root, "artifacts", "real_farm", "full_farm_mask_live_ui_8501");
const profilePath = path.join(outputDir, `.chrome-profile-${process.pid}`);
fs.mkdirSync(outputDir, { recursive: true });
const chrome = spawn(chromePath, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--remote-debugging-port=9223", `--user-data-dir=${profilePath}`, "--window-size=1900,1200",
  `http://localhost:${appPort}/?base_source=real_farm`,
], { stdio: "ignore" });

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function endpoint() {
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      const pages = await (await fetch("http://127.0.0.1:9223/json")).json();
      const page = pages.find(item => item.type === "page" && item.url.includes(`localhost:${appPort}`));
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await delay(250);
  }
  throw new Error("Chrome DevTools endpoint did not become ready");
}

let socket;
let commandId = 0;
const pending = new Map();
function command(method, params = {}) {
  const id = ++commandId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
async function evaluate(expression) {
  const response = await command("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (response.result.exceptionDetails) throw new Error(response.result.exceptionDetails.text);
  return response.result.result.value;
}
async function waitForText(text, seconds = 60) {
  for (let attempt = 0; attempt < seconds * 4; attempt++) {
    if (await evaluate(`document.body.innerText.includes(${JSON.stringify(text)})`)) return;
    await delay(250);
  }
  const body = await evaluate(`document.body.innerText.slice(0,10000)`);
  throw new Error(`Timed out waiting for UI text: ${text}; body=${body}`);
}
async function clickButton(text) {
  const clicked = await evaluate(`(async () => { const b=[...document.querySelectorAll('button')].find(x => x.innerText.trim().includes(${JSON.stringify(text)})); if(!b)return false; if(b.disabled)return 'disabled'; b.scrollIntoView({block:'center'}); await new Promise(r=>setTimeout(r,300)); b.click(); return true; })()`);
  if (clicked !== true) {
    const buttons = await evaluate(`JSON.stringify([...document.querySelectorAll('button')].map(x => x.innerText.trim()))`);
    throw new Error(`Button unavailable (${clicked}): ${text}; available=${buttons}`);
  }
}
async function clickLabel(text) {
  const clicked = await evaluate(`(() => { const nodes=[...document.querySelectorAll('label,p,span')]; const e=nodes.find(x => x.innerText.trim()===${JSON.stringify(text)}); if(!e)return false; const target=e.closest('label') || e; target.scrollIntoView({block:'center'}); target.click(); return true; })()`);
  if (!clicked) {
    const labels = await evaluate(`JSON.stringify([...document.querySelectorAll('label,p,span')].map(x => x.innerText.trim()).filter(Boolean).slice(-120))`);
    throw new Error(`Label unavailable: ${text}; labels=${labels}`);
  }
  await delay(900);
}
async function selectOption(labelText, optionText) {
  const opened = await evaluate(`(() => { const labels=[...document.querySelectorAll('label')]; const l=labels.find(x => x.innerText.trim()===${JSON.stringify(labelText)}); if(!l)return false; let root=l.parentElement; for(let i=0;i<5 && root;i++,root=root.parentElement){const input=root.querySelector('input[role=combobox]'); if(input){input.scrollIntoView({block:'center'}); input.click(); return true;}} return false; })()`);
  if (!opened) throw new Error(`Selectbox unavailable: ${labelText}`);
  await delay(400);
  const selected = await evaluate(`(() => { const options=[...document.querySelectorAll('[role=option]')]; const e=options.find(x => x.innerText.trim().includes(${JSON.stringify(optionText)})); if(!e)return false; e.click(); return true; })()`);
  if (!selected) throw new Error(`Option unavailable: ${optionText}`);
  await delay(1200);
}
async function scrollToText(text, offset = -90) {
  await evaluate(`(() => { const nodes=[...document.querySelectorAll('h1,h2,h3,h4,p,div')]; const e=nodes.find(x => x.textContent.trim()===${JSON.stringify(text)}); if(e){e.scrollIntoView({block:'start'}); window.scrollBy(0,${offset}); return true;} return false; })()`);
  await delay(700);
}
async function screenshot(name) {
  const shot = await command("Page.captureScreenshot", { format: "png", captureBeyondViewport: false, fromSurface: true });
  fs.writeFileSync(path.join(outputDir, name), Buffer.from(shot.result.data, "base64"));
}
async function waitForImages(seconds = 45) {
  for (let attempt = 0; attempt < seconds * 4; attempt++) {
    const ready = await evaluate(`(() => { const images=[...document.querySelectorAll('img')].filter(x => { const r=x.getBoundingClientRect(); return r.bottom>0 && r.top<window.innerHeight; }); return images.length>0 && images.every(x => x.complete && x.naturalWidth>0); })()`);
    if (ready) return;
    await delay(250);
  }
  throw new Error("Visible UI images did not finish loading");
}
async function clickRecoveredModule(index = 5) {
  for (let attempt = 0; attempt < 40; attempt++) {
    const count = await evaluate(`document.querySelectorAll('[data-testid="stPlotlyChart"] .point, .js-plotly-plot .point').length`);
    if (count > 1) break;
    await delay(250);
  }
  const point = await evaluate(`(() => { const points=[...document.querySelectorAll('[data-testid="stPlotlyChart"] .scatterlayer .point, .js-plotly-plot .scatterlayer .point')].filter((x,i,a)=>a.indexOf(x)===i); const e=points[${index}] || points[0]; if(!e)return null; e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2,count:points.length}; })()`);
  if (!point || point.count < 2) {
    const debug = await evaluate(`JSON.stringify({plots:document.querySelectorAll('[data-testid="stPlotlyChart"]').length, plotly:document.querySelectorAll('.js-plotly-plot').length, points:document.querySelectorAll('.point').length, canvases:document.querySelectorAll('canvas').length, tests:[...document.querySelectorAll('[data-testid]')].map(x=>x.getAttribute('data-testid')).filter(x=>/plot|chart/i.test(x)).slice(0,30)})`);
    throw new Error(`Clickable recovered module markers were not rendered: ${debug}`);
  }
  await command("Input.dispatchMouseEvent", { type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1 });
  await command("Input.dispatchMouseEvent", { type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1 });
  await delay(1800);
}

try {
  const wsUrl = await endpoint();
  socket = new WebSocket(wsUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const waiter = pending.get(message.id); pending.delete(message.id);
    message.error ? waiter.reject(new Error(message.error.message)) : waiter.resolve(message);
  };
  await command("Page.enable"); await command("DOM.enable"); await command("Runtime.enable");
  await waitForText("Upload one real RGB aerial/drone image");
  const documentNode = await command("DOM.getDocument", { depth: -1, pierce: true });
  const inputs = await command("DOM.querySelectorAll", { nodeId: documentNode.result.root.nodeId, selector: "input[type=file]" });
  if (!inputs.result.nodeIds.length) throw new Error("Upload input not found");
  await command("DOM.setFileInputFiles", { nodeId: inputs.result.nodeIds[0], files: [uploadPath] });
  await waitForText("Detect & Refine Full PV Farm", 90);
  await delay(1500);
  await clickButton("Detect & Refine Full PV Farm");
  await waitForText("Final union of", 180);
  await waitForText("Detected modules", 60);
  await waitForText("full-farm union mask can be generated", 180);
  await scrollToText("Refined PV Area");
  await screenshot("A_rgb_accepted_module_outlines.png");

  await clickButton("Generate Real Thermal Mask");
  await waitForText("Select Target Module", 240);
  await waitForText("Healthy Thermal", 120);
  await clickRecoveredModule();
  await clickLabel("Healthy Thermal");
  await scrollToText("Full-scene view");
  await waitForImages();
  await delay(1200);
  await screenshot("B_healthy_real_v2_full_field.png");

  await clickButton("Add Thermal Hotspot To Selected Module");
  await waitForText("Select generated hotspot", 90);
  await clickLabel("Faulted Thermal");
  await scrollToText("Full-scene view");
  await waitForImages();
  await delay(1200);
  await screenshot("C_full_field_with_hotspot.png");

  await clickButton("Close View");
  await waitForText("Actual fault for this hotspot", 60);
  await selectOption("Actual fault for this hotspot", "Bypass Diode Fault");
  await clickButton("Apply Fault To This Hotspot");
  await waitForText("Bypass Diode Fault assigned", 60);
  await waitForText("1 diagnosed", 60);
  await scrollToText("Actual fault for this hotspot", -170);
  await waitForImages();
  await screenshot("D_selected_module_bypass_diode.png");

  await selectOption("Actual fault for this hotspot", "Bypassed Substring / Section");
  await clickButton("Apply Fault To This Hotspot");
  await waitForText("Bypassed Substring / Section assigned", 60);
  await delay(2500);
  await waitForText("1 diagnosed", 60);
  await scrollToText("Actual fault for this hotspot", -170);
  await waitForImages();
  await screenshot("E_selected_module_bypassed_substring.png");

  await clickLabel("Review");
  await waitForText("Healthy Real V2 Thermal", 60);
  await scrollToText("Clean Real RGB");
  await screenshot("F_review_screen.png");

  await clickLabel("Export");
  await waitForText("Clean faulted thermal export preview", 60);
  await scrollToText("Clean faulted thermal export preview");
  await screenshot("G_clean_exported_thermal_result.png");
  console.log(outputDir);
} finally {
  try { socket?.close(); } catch {}
  chrome.kill("SIGTERM");
}
