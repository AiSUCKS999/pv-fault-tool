"""Shared real-browser acceptance for local and deployed presentation surfaces."""
import base64,hashlib,io
from PIL import Image,ImageChops

def exercise_normal_controls(view):
    base=view.locator('#base')
    payload=base.evaluate('()=>payload')
    def decoded(url):return Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).convert('RGBA')
    assert ImageChops.difference(decoded(payload['far']['original']).convert('RGB'),decoded(payload['far']['rgb']).convert('RGB')).getbbox(),'Faulted Far RGB has no changed pixels'
    expected={}
    for identity,frame in [('FAR',payload['far']),*payload['close'].items()]:
        for mode in ('original','rgb','thermal'):expected[identity,mode]=hashlib.sha256(decoded(frame[mode]).tobytes()).hexdigest()
    for selector in ['#advanced','#calibration-options']:
        if not view.locator(selector).evaluate('e=>e.open'):view.locator(selector+' > summary').click()
    evidence=[]
    buttons=['[data-distance=FAR]','[data-distance=CLOSE]','[data-mode=original]','[data-mode=rgb]','[data-mode=thermal]','#boxes','#previous','#next']
    for pattern in ('gray','detail','fault'):
        for button in buttons:
            # Establish view-only preconditions in one notification. The calibration
            # selector and every exit button below are real browser interactions.
            base.evaluate("()=>normalView(()=>{state.distance='FAR';state.mode='rgb';})")
            view.locator('#calibration').select_option(pattern)
            base.evaluate("()=>{state.fit='fill';state.zoom=2;changed();}")
            assert view.locator('[data-distance][aria-pressed=true]').count()==0
            assert view.locator('#calibration-status').is_visible()
            previous=base.evaluate('()=>({index:state.index,boxes:state.boxes})')
            view.locator(button).click()
            actual=base.evaluate('''async e=>{
                await e.decode();const c=document.createElement('canvas');c.width=e.naturalWidth;c.height=e.naturalHeight;
                const ctx=c.getContext('2d');ctx.drawImage(e,0,0);
                const hash=await crypto.subtle.digest('SHA-256',ctx.getImageData(0,0,c.width,c.height).data);
                return {calibration:state.calibration,fixed:state.fixed,zoom:state.zoom,fit:state.fit,clean:state.clean,
                    distance:state.distance,mode:state.mode,index:state.index,boxes:state.boxes,
                    size:[e.naturalWidth,e.naturalHeight],declared:[frame().width,frame().height],
                    sourceCorrect:e.getAttribute('src')===frame()[state.mode],gray:getComputedStyle(document.getElementById('gray')).display,
                    visibility:getComputedStyle(plane).visibility,hash:Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('')};}''')
            assert actual['calibration']=='off' and actual['fixed'] is None
            assert actual['zoom']==1 and actual['fit']=='fit' and not actual['clean']
            assert actual['gray']=='none' and actual['visibility']=='visible' and actual['sourceCorrect']
            assert actual['size']==actual['declared']
            assert actual['distance']==('CLOSE' if button=='[data-distance=CLOSE]' else 'FAR')
            key='FAR' if actual['distance']=='FAR' else payload['faults'][actual['index']]['id']
            assert actual['hash']==expected[key,actual['mode']],actual
            if actual['distance']=='FAR':assert actual['size']==[payload['far']['width'],payload['far']['height']]
            else:assert actual['size'][0]<payload['far']['width']
            if button=='#next':assert actual['index']==(previous['index']+1)%len(payload['faults'])
            if button=='#previous':assert actual['index']==(previous['index']-1)%len(payload['faults'])
            if button=='#boxes':assert actual['boxes'] is True
            boxes=payload['far']['boxes'] if key=='FAR' else payload['close'][key]['boxes']
            assert sorted(view.locator('#labels text').all_text_contents())==(sorted(b['name'] for b in boxes) if actual['boxes'] else [])
            assert view.locator('#calibration-status').is_hidden()
            evidence.append({'pattern':pattern,'button':button,**actual})
            print('BROWSER_CHECK',pattern,button,actual['size'],flush=True)
    view.locator('[data-distance=FAR]').click();view.locator('[data-mode=rgb]').click()
    # Separate display follows the same normal-state transition via existing synchronization.
    return {'checks':evidence,'far_source_size':[payload['far']['width'],payload['far']['height']],
        'close_source_sizes':{oid:[f['width'],f['height']] for oid,f in payload['close'].items()},
        'fault_ids':[f['id'] for f in payload['faults']],'fault_names':sorted({b['name'] for b in payload['far']['boxes']})}
