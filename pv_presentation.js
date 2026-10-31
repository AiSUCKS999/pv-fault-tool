/* Presentation only: no fault generation, interpolation models, or annotations baked into pixels. */
let payload = INITIAL;
let state = {distance:'FAR', mode:'rgb', index:0, fit:'fit', boxes:false, split:50};
let popup = null;
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel(CHANNEL) : null;
const $ = id => document.getElementById(id);
const screen = $('screen'), plane = $('plane');
if (!CONTROLLER) document.body.classList.add('presentation');
function fault() { return (payload.faults || [])[state.index]; }
function frame() { return state.distance === 'CLOSE' && fault() ? payload.close[fault().id] : payload.far; }
function geometry() {
  const f = frame(); if (!f) return;
  const w = screen.clientWidth, h = screen.clientHeight;
  const scale = Math[state.fit === 'fill' ? 'max' : 'min'](w/f.width, h/f.height);
  Object.assign(plane.style, {width:f.width*scale+'px',height:f.height*scale+'px',left:(w-f.width*scale)/2+'px',top:(h-f.height*scale)/2+'px'});
}
function render() {
  if (state.index >= (payload.faults || []).length) state.index = 0;
  const f=frame(); if (!f) return;
  const base = state.mode === 'compare' ? f.thermal : f[state.mode];
  if (!base) { $('status').textContent='Requested acquisition is unavailable'; return; }
  $('base').src=base;
  $('compare').style.display=state.mode==='compare'?'block':'none';
  $('compare').src=f.rgb;
  $('compare').style.clipPath=`inset(0 ${100-state.split}% 0 0)`;
  const svg=$('labels'); svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);
  if(state.boxes) (f.boxes || []).forEach(box=>{
    const group=document.createElementNS('http://www.w3.org/2000/svg','g');group.dataset.fault=box.id;
    const rect=document.createElementNS(svg.namespaceURI,'rect');
    const [x1,y1,x2,y2]=box.bbox;
    Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'transparent',stroke:'#39e6cb','stroke-width':Math.max(1,f.width/650)}).forEach(([k,v])=>rect.setAttribute(k,v));
    const label=document.createElementNS(svg.namespaceURI,'text');label.setAttribute('x',x1);label.setAttribute('y',Math.max(14,y1-3));label.setAttribute('font-size',Math.max(12,f.width/85));label.textContent=`${box.id} · ${box.kind}`;
    group.append(rect,label); group.onclick=()=>{state.index=payload.faults.findIndex(x=>x.id===box.id);state.distance='CLOSE';changed();};svg.append(group);
  });
  document.querySelectorAll('[data-mode]').forEach(e=>e.setAttribute('aria-pressed',e.dataset.mode===state.mode));
  document.querySelectorAll('[data-distance]').forEach(e=>e.setAttribute('aria-pressed',e.dataset.distance===state.distance));
  $('boxes').setAttribute('aria-pressed',state.boxes);$('fit').value=state.fit;
  $('status').textContent=`${payload.identity} · ${state.distance} · ${f.width} × ${f.height} source pixels · ${fault()?.id || 'Healthy'} · ${state.fit==='fit'?'FIT: whole image, letterboxing':'FILL: edges cropped'}${payload.limitation?' · '+payload.limitation:''}`;
  geometry();
}
function send() { channel?.postMessage({type:'view',state}); if(popup && !popup.closed) popup.postMessage({type:'pv-view',state},'*'); }
function changed() {render();send();}
function step(direction) {const n=(payload.faults||[]).length;if(n)state.index=(state.index+direction+n)%n;changed();}
async function fullscreen() {
  try {if(document.fullscreenElement) await document.exitFullscreen();else await screen.requestFullscreen();}
  catch(error){$('status').textContent='Fullscreen unavailable here. Open Display, then press Enter or use browser fullscreen.';}
}
document.querySelectorAll('[data-mode]').forEach(e=>e.onclick=()=>{state.mode=e.dataset.mode;changed();});
document.querySelectorAll('[data-distance]').forEach(e=>e.onclick=()=>{state.distance=e.dataset.distance;changed();});
$('fullscreen').onclick=fullscreen;$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('fit').onchange=e=>{state.fit=e.target.value;changed();};$('boxes').onclick=()=>{state.boxes=!state.boxes;changed();};
$('open').onclick=()=>{
  popup=window.open('',CHANNEL,'popup,width=1280,height=720');
  if(!popup){$('status').textContent='Allow popups for the separate presentation display.';return;}
  const copy=document.documentElement.outerHTML.replace('const CONTROLLER=true','const CONTROLLER=false');
  popup.document.open();popup.document.write('<!doctype html>'+copy);popup.document.close();
  popup.addEventListener('load',()=>popup.postMessage({type:'pv-view',state},'*'),{once:true});
};
window.addEventListener('message',event=>{if(event.data?.type==='pv-view' && event.source===window.opener){state=event.data.state;render();}});
if(channel)channel.onmessage=event=>{if(event.data.type==='view'){state=event.data.state;render();}if(event.data.type==='payload'){payload=event.data.payload;render();}};
if(CONTROLLER)channel?.postMessage({type:'payload',payload});
document.addEventListener('keydown',event=>{
  if(event.target.tagName==='SELECT')return;
  const key=event.key.toLowerCase();
  if(key==='enter'){fullscreen();return;}
  if(key==='escape'){if(document.fullscreenElement)document.exitFullscreen();return;}
  if(key==='arrowright'){event.preventDefault();step(1);return;}if(key==='arrowleft'){event.preventDefault();step(-1);return;}
  if(key==='f')state.distance='FAR';else if(key==='c')state.distance='CLOSE';
  else if(key==='r')state.mode='rgb';else if(key==='t')state.mode='thermal';else if(key==='o')state.mode='original';else if(key==='v')state.mode='compare';
  else if(key==='b')state.boxes=!state.boxes;else if(key==='x')state.fit=state.fit==='fit'?'fill':'fit';else return;changed();
});
screen.addEventListener('pointermove',event=>{if(state.mode==='compare'){state.split=Math.max(0,Math.min(100,(event.clientX-plane.getBoundingClientRect().left)/plane.clientWidth*100));changed();}});
screen.ondblclick=fullscreen;new ResizeObserver(geometry).observe(screen);document.addEventListener('fullscreenchange',geometry);
// Decode once; view changes reuse the already acquired images and identities.
for(const f of [payload.far,...Object.values(payload.close||{})]) for(const k of ['rgb','thermal','original']) if(f?.[k]){const img=new Image();img.src=f[k];}
render();
