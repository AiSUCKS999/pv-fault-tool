/* Presentation only: no fault generation, interpolation models, or annotations baked into pixels. */
let payload = INITIAL;
let state = {distance:'FAR', mode:'rgb', index:0, fit:'fit', boxes:false, split:50, zoom:1, clean:false, calibration:'off', fixed:null};
let popup = null;
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel(CHANNEL) : null;
const $ = id => document.getElementById(id);
const screen = $('screen'), plane = $('plane');
if (!CONTROLLER) document.body.classList.add('presentation');
function fault() { return (payload.faults || [])[state.index]; }
function liveFrame() { return state.distance === 'CLOSE' && fault() ? payload.close[fault().id] : payload.far; }
function frame() {
  if(state.calibration==='fault' && state.fixed) return state.fixed.frame;
  if(state.calibration==='detail') return fault() ? payload.close[fault().id] : payload.far;
  return liveFrame();
}
function geometry() {
  const f = frame(); if (!f) return;
  const w = screen.clientWidth, h = screen.clientHeight;
  const scale = Math[state.fit === 'fill' ? 'max' : 'min'](w/f.width, h/f.height) * (state.zoom || 1);
  Object.assign(plane.style, {width:f.width*scale+'px',height:f.height*scale+'px',left:(w-f.width*scale)/2+'px',top:(h-f.height*scale)/2+'px'});
}
function notice(message) {$('notice').textContent=message;$('notice').hidden=!message;}
function drawBoxes(svg,f) {
  // Geometry comes only from the acquisition's metadata, in native pixels.
  const scale=Math[state.fit==='fill'?'max':'min'](screen.clientWidth/f.width,screen.clientHeight/f.height)*(state.zoom||1);
  const font=14/scale, pad=5/scale, height=22/scale, placed=[];
  const measure=document.createElement('canvas').getContext('2d');measure.font=`600 ${font}px system-ui`;
  const clamp=(v,low,high)=>Math.max(low,Math.min(v,high));
  const overlap=(a,b)=>Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const element=(name,attributes)=>{const e=document.createElementNS(svg.namespaceURI,name);Object.entries(attributes).forEach(([k,v])=>e.setAttribute(k,v));return e;};
  for(const box of [...(f.boxes||[])].sort((a,b)=>a.bbox[1]-b.bbox[1]||a.bbox[0]-b.bbox[0])) {
    const [a,b,c,d]=box.bbox;
    const x1=clamp(a,0,f.width),y1=clamp(b,0,f.height),x2=clamp(c,0,f.width),y2=clamp(d,0,f.height);
    if(x2<=x1||y2<=y1)continue;
    const name=box.name||box.class||'Fault';
    const width=Math.min(f.width,measure.measureText(name).width+2*pad);
    const candidates=[[x1,y1-height-pad],[x1,y2+pad],[x2-width,y1-height-pad],[x1+pad,y1+pad]];
    for(let y=0;y+height<=f.height;y+=height+pad)candidates.push([x1,y],[x2-width,y]);
    let best=null,bestScore=Infinity;
    for(const [cx,cy] of candidates) {
      const x=clamp(cx,0,Math.max(0,f.width-width)),y=clamp(cy,0,Math.max(0,f.height-height));
      const r=[x,y,x+width,y+height];
      const score=placed.reduce((n,p)=>n+overlap(r,p),0)*1e6+Math.abs(y-y1)+Math.abs(x-x1);
      if(score<bestScore){best=r;bestScore=score;}
    }
    placed.push(best);
    const [lx,ly,rx,by]=best;
    const group=element('g',{'data-fault':box.id,'aria-label':name});
    const title=element('title',{});title.textContent=`${name} · ${box.id}`;
    const rect=element('rect',{'data-bbox':box.id,x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:'#39e6cb','stroke-width':2,'vector-effect':'non-scaling-stroke'});
    const line=element('line',{x1:clamp((lx+rx)/2,x1,x2),y1:clamp((ly+by)/2,y1,y2),x2:(lx+rx)/2,y2:(ly+by)/2,stroke:'#39e6cb','stroke-width':1,'vector-effect':'non-scaling-stroke'});
    const label=element('g',{'data-label':box.id});
    const background=element('rect',{x:lx,y:ly,width:rx-lx,height:by-ly,rx:3/scale,fill:'#07141f','fill-opacity':.94,stroke:'#39e6cb','stroke-width':1,'vector-effect':'non-scaling-stroke'});
    const text=element('text',{x:lx+pad,y:ly+height*.70,'font-size':font});text.textContent=name;
    label.append(background,text);group.append(title,rect,line,label);
    group.onclick=()=>{state.index=payload.faults.findIndex(x=>x.id===box.id);state.distance='CLOSE';changed();};svg.append(group);
  }
}
function render() {
  if (state.index >= (payload.faults || []).length) state.index = 0;
  const f=frame(); if (!f) return;
  const calibrating=state.calibration && state.calibration!=='off';
  $('gray').style.display=state.calibration==='gray'?'block':'none';
  plane.style.visibility=state.calibration==='gray'?'hidden':'visible';
  screen.style.cursor=(calibrating || state.clean)?'none':'';
  const mode=state.calibration==='detail'?'original':state.calibration==='fault'?state.fixed.mode:state.mode;
  const base = mode === 'compare' ? f.thermal : f[mode];
  if (!base) { notice('Requested acquisition is unavailable'); return; }
  $('base').src=base;
  $('compare').style.display=mode==='compare' && !calibrating?'block':'none';
  $('compare').src=f.rgb;
  $('compare').style.clipPath=`inset(0 ${100-state.split}% 0 0)`;
  const svg=$('labels'); svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);
  if(state.boxes && !state.clean && !calibrating) drawBoxes(svg,f);
  document.querySelectorAll('[data-mode]').forEach(e=>e.setAttribute('aria-pressed',e.dataset.mode===state.mode));
  document.querySelectorAll('[data-distance]').forEach(e=>e.setAttribute('aria-pressed',e.dataset.distance===state.distance));
  $('boxes').setAttribute('aria-pressed',state.boxes && !state.clean && !calibrating);$('boxes').disabled=!!calibrating;$('fit').value=state.fit;
  $('zoom').value=state.zoom || 1;$('calibration').value=state.calibration || 'off';$('clean').setAttribute('aria-pressed',!!state.clean);
  $('status').textContent=`${payload.identity} · ${state.distance} · ${f.width} × ${f.height} source pixels · ${fault()?.id || 'Healthy'} · ${state.fit==='fit'?'FIT: whole image, letterboxing':'FILL: edges cropped'}${payload.limitation?' · '+payload.limitation:''}`;
  geometry();
}
function send() { channel?.postMessage({type:'view',state}); if(popup && !popup.closed) popup.postMessage({type:'pv-view',state},'*'); }
let lastOperator='';
function operatorState() {
  const next=JSON.stringify([state.distance,state.mode,state.index,state.boxes,state.fit,state.zoom,state.clean,state.calibration,state.fixed?.id]);
  if(CONTROLLER && next!==lastOperator){lastOperator=next;parent.postMessage({type:'pv-operator-state',state:{...state,fixed:undefined,calibration_settings:state.calibration!=='off'?{pattern:state.calibration,gray_code:state.calibration==='gray'?128:null,fault_id:state.fixed?.id || null,modality:state.calibration==='detail'?'original':state.fixed?.mode || state.mode,zoom:state.zoom || 1,fit:state.fit,viewport:[screen.clientWidth,screen.clientHeight],source_size:[frame().width,frame().height]}:null},signature:payload.revision},'*');}
}
function changed() {render();send();operatorState();}
function step(direction) {const n=(payload.faults||[]).length;if(n)state.index=(state.index+direction+n)%n;changed();}
async function fullscreen() {
  try {if(document.fullscreenElement) await document.exitFullscreen();else await screen.requestFullscreen();}
  catch(error){notice('Fullscreen unavailable. Use Open Display in Advanced Options.');}
}
document.querySelectorAll('[data-mode]').forEach(e=>e.onclick=()=>{state.mode=e.dataset.mode;changed();});
document.querySelectorAll('[data-distance]').forEach(e=>e.onclick=()=>{state.distance=e.dataset.distance;changed();});
$('fullscreen').onclick=fullscreen;$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('fit').onchange=e=>{state.fit=e.target.value;changed();};$('boxes').onclick=()=>{state.boxes=!(state.boxes && !state.clean);if(state.boxes)state.clean=false;changed();};
$('zoom').oninput=e=>{state.zoom=Number(e.target.value);changed();};
$('clean').onclick=()=>{state.clean=!state.clean;changed();};
$('calibration').onchange=e=>{
  if(e.target.value==='fault') {
    if(!fault()){e.target.value=state.calibration || 'off';notice('Generate a fault before using this pattern');return;}
    state.fixed={id:fault().id,frame:payload.close[fault().id],mode:state.mode==='compare'?'thermal':state.mode==='original'?'rgb':state.mode};
  }
  state.calibration=e.target.value;changed();
};
$('open').onclick=()=>{
  popup=window.open('',CHANNEL,'popup,width=1280,height=720');
  if(!popup){notice('Allow popups for the separate display');return;}
  const copy=document.documentElement.outerHTML.replace('const CONTROLLER=true','const CONTROLLER=false');
  popup.document.open();popup.document.write('<!doctype html>'+copy);popup.document.close();
  popup.addEventListener('load',()=>popup.postMessage({type:'pv-view',state},'*'),{once:true});
};
window.addEventListener('message',event=>{if(event.data?.type==='pv-view' && event.source===window.opener){state=event.data.state;render();}});
if(channel)channel.onmessage=event=>{if(event.data.type==='view'){state=event.data.state;render();operatorState();}if(event.data.type==='payload'){payload=event.data.payload;render();}};
if(CONTROLLER)channel?.postMessage({type:'payload',payload});
document.addEventListener('keydown',event=>{
  if(['SELECT','INPUT'].includes(event.target.tagName))return;
  const key=event.key.toLowerCase();
  if(key==='enter'){fullscreen();return;}
  if(key==='escape'){if(document.fullscreenElement)document.exitFullscreen();return;}
  if(key==='arrowright'){event.preventDefault();step(1);return;}if(key==='arrowleft'){event.preventDefault();step(-1);return;}
  if(key==='f')state.distance='FAR';else if(key==='c')state.distance='CLOSE';
  else if(key==='r')state.mode='rgb';else if(key==='t')state.mode='thermal';else if(key==='o')state.mode='original';else if(key==='v')state.mode='compare';
  else if(key==='b'){if(state.calibration!=='off')return;state.boxes=!(state.boxes && !state.clean);if(state.boxes)state.clean=false;}else if(key==='x')state.fit=state.fit==='fit'?'fill':'fit';else return;changed();
});
screen.addEventListener('pointermove',event=>{if(state.mode==='compare' && (!state.calibration || state.calibration==='off')){state.split=Math.max(0,Math.min(100,(event.clientX-plane.getBoundingClientRect().left)/plane.clientWidth*100));changed();}});
screen.ondblclick=fullscreen;new ResizeObserver(render).observe(screen);document.addEventListener('fullscreenchange',render);
// Decode once; view changes reuse the already acquired images and identities.
for(const f of [payload.far,...Object.values(payload.close||{})]) for(const k of ['rgb','thermal','original']) if(f?.[k]){const img=new Image();img.src=f[k];}
render();
