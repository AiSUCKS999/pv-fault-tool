import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1")), "..");
const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const appPort = process.env.REAL_FARM_PORT || "8502";
const uploadPath = path.join(root, "artifacts", "real_farm", "uploads", "daniel-miksha-PNX3H9jkT4c-unsplash_c6a4eafd73c3f79d.jpg");
const outputPath = path.join(root, "artifacts", "real_farm", "live_mask_diagnostic", "live_ui_after.png");
const hotspotOutputPath = path.join(root, "artifacts", "real_farm", "live_mask_diagnostic", "live_ui_hotspot_after.png");
const profilePath = path.join(root, "artifacts", "real_farm", "live_mask_diagnostic", `.chrome-profile-${process.pid}`);
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
const chrome = spawn(chromePath, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--remote-debugging-port=9222", `--user-data-dir=${profilePath}`, "--window-size=1900,1200",
  `http://localhost:${appPort}/?base_source=real_farm`,
], { stdio: "ignore" });

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function endpoint() {
  for (let attempt = 0; attempt < 80; attempt++) {
    try {
      const pages = await (await fetch("http://127.0.0.1:9222/json")).json();
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
  return response.result.result.value;
}
async function waitForText(text, seconds = 45) {
  for (let attempt = 0; attempt < seconds * 4; attempt++) {
    if (await evaluate(`document.body.innerText.includes(${JSON.stringify(text)})`)) return;
    await delay(250);
  }
  const body = await evaluate(`document.body.innerText.slice(0,8000)`);
  throw new Error(`Timed out waiting for UI text: ${text}; body=${body}`);
}
async function clickButton(text) {
  const clicked = await evaluate(`(async () => { const b=[...document.querySelectorAll('button')].find(x => x.innerText.trim().includes(${JSON.stringify(text)})); if(!b)return false; if(b.disabled)return 'disabled'; b.scrollIntoView({block:'center'}); await new Promise(r=>setTimeout(r,350)); b.focus(); b.click(); return true; })()`);
  if (clicked !== true) {
    const buttons = await evaluate(`JSON.stringify([...document.querySelectorAll('button')].map(x => x.innerText.trim()))`);
    throw new Error(`Button unavailable (${clicked}): ${text}; available=${buttons}`);
  }
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
  let documentNode = await command("DOM.getDocument", { depth: -1, pierce: true });
  let inputs = await command("DOM.querySelectorAll", { nodeId: documentNode.result.root.nodeId, selector: "input[type=file]" });
  if (!inputs.result.nodeIds.length) throw new Error("Upload input not found");
  await command("DOM.setFileInputFiles", { nodeId: inputs.result.nodeIds[0], files: [uploadPath] });
  await waitForText("Loaded daniel-miksha");
  await waitForText("Manual Geometry Workspace", 60);
  await delay(1500);
  await clickButton("Add Module Polygon");
  await waitForText("Thermal generation stays locked", 60);
  await clickButton("Auto-Refine");
  await waitForText("Proposed final binary PV mask");
  await clickButton("Accept Mask");
  await waitForText("AUTO_REFINED_RGB");
  await delay(1200);
  await clickButton("Generate Real Thermal Mask");
  await waitForText("Thermal workspace", 60);
  await evaluate(`(() => { const nodes=[...document.querySelectorAll('*')]; const e=nodes.find(x => x.textContent.trim()==='Thermal workspace'); if(e){e.scrollIntoView({block:'start'}); window.scrollBy(0,-100);} })()`);
  await delay(1200);
  const shot = await command("Page.captureScreenshot", { format: "png", captureBeyondViewport: false, fromSurface: true });
  fs.writeFileSync(outputPath, Buffer.from(shot.result.data, "base64"));
  await clickButton("Generate Hotspots");
  await waitForText("Select generated hotspot", 60);
  await evaluate(`(() => { const nodes=[...document.querySelectorAll('*')]; const e=nodes.find(x => x.textContent.trim()==='Thermal workspace'); if(e){e.scrollIntoView({block:'start'}); window.scrollBy(0,-100);} })()`);
  await delay(1200);
  const hotspotShot = await command("Page.captureScreenshot", { format: "png", captureBeyondViewport: false, fromSurface: true });
  fs.writeFileSync(hotspotOutputPath, Buffer.from(hotspotShot.result.data, "base64"));
  console.log(outputPath);
  console.log(hotspotOutputPath);
} finally {
  try { socket?.close(); } catch {}
  chrome.kill("SIGTERM");
}
