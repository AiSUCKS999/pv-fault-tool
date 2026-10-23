"""Local RGB table/frame recovery. No source names, expected counts or review assets.

Region masks locate material; fitted line intersections locate physical instances.
Every output remains a proposal until visual review in the Streamlit workflow.
"""
import cv2
import numpy as np


def ordered_quad(points):
    q = np.asarray(points, np.float32).reshape(4, 2)
    center = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)


def material_mask(rgb):
    """Bound allocations by rows; retain weak blue under glare, exclude vegetation."""
    mask = np.empty(rgb.shape[:2], np.uint8)
    for start in range(0, rgb.shape[0], 256):
        part = rgb[start:start+256]
        hsv = cv2.cvtColor(part, cv2.COLOR_RGB2HSV)
        r, g, b = (part[:, :, i].astype(np.int16) for i in range(3))
        blue = ((hsv[:, :, 0] >= 78) & (hsv[:, :, 0] <= 136)
                & (hsv[:, :, 1] >= 14) & (hsv[:, :, 2] >= 20)
                & (b >= r+3) & (b >= g-5))
        mask[start:start+256] = np.uint8(blue & ~(g > (r+b)*.5+12))*255
    return mask


def regions_from_support(mask):
    # Do not shrink physical gutters before grouping: a 3-pixel close on an
    # 1800px thumbnail bridges 6-8 native pixels and fuses diagonal modules.
    # One native mask operation is bounded; per-instance processing stays local.
    scale = 1.
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < 120:
            continue
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, cv2.arcLength(hull, True)*.018, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = ordered_quad(approx.reshape(4, 2)/scale)
        else:
            q = ordered_quad(cv2.boxPoints(cv2.minAreaRect(hull))/scale)
        if min(np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)) < 7/scale:
            continue
        # A small blue connection can join neighboring tables. Split only at
        # sustained low-material gutters in this region's own orientation.
        edges = np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        rw,rh = max(12,round(float((edges[0]+edges[2])*.5))),max(12,round(float((edges[1]+edges[3])*.5)))
        local_scale=min(1.,1800/max(rw,rh));rw,rh=max(12,round(rw*local_scale)),max(12,round(rh*local_scale))
        dst=np.float32([[0,0],[rw-1,0],[rw-1,rh-1],[0,rh-1]])
        H=cv2.getPerspectiveTransform(q,dst);inv=cv2.getPerspectiveTransform(dst,q)
        bounds_low=np.maximum(0,np.floor(q.min(0)-2)).astype(int);bounds_high=np.minimum([mask.shape[1],mask.shape[0]],np.ceil(q.max(0)+3)).astype(int)
        bx,by=bounds_low;ex,ey=bounds_high
        shift=np.float64([[1,0,bx],[0,1,by],[0,0,1]])
        local=cv2.warpPerspective(mask[by:ey,bx:ex],H@shift,(rw,rh))>0
        spans=[]
        for axis,length in [(0,rw),(1,rh)]:
            profile=local.mean(axis=axis);missing=np.where(profile<.12)[0]
            groups=np.split(missing,np.where(np.diff(missing)>1)[0]+1) if len(missing) else []
            cuts=[(int(a[0]),int(a[-1])+1) for a in groups if len(a)>=max(6,round(7*local_scale)) and a[0]>3 and a[-1]<length-4]
            runs=[];start=0
            for left,right in cuts:
                if left-start>=12:runs.append((start,left-1))
                start=right
            if length-start>=12:runs.append((start,length-1))
            spans.append(runs or [(0,length-1)])
        for left,right in spans[0]:
            for top,bottom in spans[1]:
                if local[top:bottom+1,left:right+1].mean()<.35:continue
                sub=np.float32([[[left,top],[right,top],[right,bottom],[left,bottom]]])
                regions.append(ordered_quad(cv2.perspectiveTransform(sub,inv)[0]))
    regions.sort(key=lambda q: (float(q[:, 1].min()), float(q[:, 0].min())))
    return regions


def frame_positions(profile, low, high):
    """Broad neutral ridges, rather than the periodic fine cell-grid frequency."""
    profile = np.asarray(profile, np.float32)
    smooth = cv2.GaussianBlur(profile[None], (0, 0), .85).ravel()
    lo, hi = max(0, int(low)), min(len(smooth), int(high)+1)
    values = smooth[lo:hi]
    if not len(values):
        return []
    baseline = float(np.percentile(values, 45))
    bright = float(np.percentile(values, 98))
    threshold = max(110., baseline+max(18., (bright-baseline)*.52))
    indices = np.where(smooth >= threshold)[0]
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0]+1) if len(indices) else []
    peaks = []
    for group in groups:
        if len(group) < 2:
            continue
        weights = np.maximum(smooth[group]-baseline, 1)
        position = float(np.average(group, weights=weights))
        if low-7 <= position <= high+7:
            peaks.append((position, float(smooth[group].max()), len(group)))
    # Do not make two module boundaries from opposite flanks of one gutter.
    joined = []
    for p in peaks:
        if joined and p[0]-joined[-1][0] < 6:
            previous = joined[-1]
            joined[-1] = ((p[0]+previous[0])*.5, max(p[1], previous[1]), p[2]+previous[2])
        else:
            joined.append(p)
    return joined


def _endpoints(peaks, start, end, partial_start=False, partial_end=False):
    values = [p[0] for p in peaks if start-7 <= p[0] <= end+7]
    # The observed material extent supplies the exterior search boundary only.
    # Final module sides are independently RGB-fitted below.
    for point, partial in [(start, partial_start), (end, partial_end)]:
        nearby = [v for v in values if abs(v-point) <= 9]
        if not nearby:
            values.append(float(point))
    return sorted(values)


def _fit_side(plane, position, vertical, start, end, radius=4):
    limit = plane.shape[1 if vertical else 0]
    along=np.linspace(start,end,24)
    indices=np.clip(np.rint(along).astype(int),0,plane.shape[0 if vertical else 1]-1)
    lo,hi=max(0,round(position-radius)),min(limit,round(position+radius)+1)
    if hi<=lo:
        return (0., float(position), 0.)
    transverse=np.arange(lo,hi)
    samples=plane[indices[:,None],transverse[None,:]] if vertical else plane[transverse[None,:],indices[:,None]]
    points=np.column_stack([along,lo+np.argmax(samples,axis=1)])
    fit = np.polyfit(points[:, 0], points[:, 1], 1)
    residual = abs(points[:, 1]-np.polyval(fit, points[:, 0]))
    keep = residual <= max(1.2, float(np.median(residual))*2.5)
    if keep.sum() >= 5:
        fit = np.polyfit(points[keep, 0], points[keep, 1], 1)
    return (float(fit[0]), float(fit[1]), float(keep.mean()))


def table_gap_cuts(local):
    """Find sustained, wide material gutters with terminated frames on both sides.

    Coordinates are in a region's own perspective plane. A dark cell line alone
    cannot pass: width, longitudinal persistence and two bright frame flanks
    must agree. Returned lines describe the *gap*, never a clipped module.
    """
    h, w = local.shape[:2]
    if min(h, w) < 18:
        return []
    neutral = local.min(2).astype(np.float32)
    # Estimate coarse pitch from independently visible neutral frame ridges.
    intervals = []
    for fraction in (.25, .75):
        y = round((h-1)*fraction)
        profile = np.percentile(neutral[max(0,y-h//12):min(h,y+h//12+1)],40,axis=0)
        positions = [p[0] for p in frame_positions(profile,0,w-1)]
        intervals.extend(d for d in np.diff(positions) if h*.10 < d < h*1.4)
    pitch = float(np.percentile(intervals,40)) if len(intervals)>=3 else min(h,w)*.45
    strip = max(3, round(pitch*.075))
    r,g,b = [local[:,:,i].astype(np.float32) for i in range(3)]
    brightness = local.max(2)
    dark = brightness < np.median(brightness,axis=1)[:,None]*.62
    material = (dark | (g > (r+b)*.5+8)).astype(np.float32)
    lo,hi = round(h*.10),round(h*.90)
    yy = np.arange(h,dtype=np.float32)
    xx = np.arange(w,dtype=np.float32)
    candidates = []
    # Local shear handles slanted table ends; no farm-axis alignment is used.
    for slope in np.linspace(-.30,.30,25):
        mapx = np.broadcast_to(xx,(h,w)) + (yy[:,None]-h*.5)*slope
        aligned = cv2.remap(material,mapx.astype(np.float32),np.broadcast_to(yy[:,None],(h,w)),cv2.INTER_NEAREST)
        broad = cv2.boxFilter(aligned,-1,(strip,1),normalize=True)
        score = (broad[lo:hi]>.80).mean(0)
        indices = np.where(score>.72)[0]
        groups = np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(indices) else []
        for group in groups:
            center = float(group[np.argmax(score[group])])
            if center < pitch*.35 or center > w-1-pitch*.35:
                continue
            candidates.append((float(score[round(center)]),center,float(slope)))
    cuts=[]
    for score,center,slope in sorted(candidates,reverse=True):
        if any(abs(center-(c['left'][1]+c['right'][1])*.5)<pitch*.45 for c in cuts):
            continue
        samples=[]
        for y in range(lo,hi):
            x=round(center+(y-h*.5)*slope)
            row=material[y];a=max(0,x-round(pitch*.18));bnd=min(w,x+round(pitch*.18)+1)
            dark_indices=np.where(row[a:bnd]>0)[0]+a
            runs=np.split(dark_indices,np.where(np.diff(dark_indices)>1)[0]+1) if len(dark_indices) else []
            runs=[run for run in runs if len(run)>=strip and abs(float(run.mean())-x)<pitch*.12]
            if not runs:continue
            run=min(runs,key=lambda run:abs(float(run.mean())-x));l,rgt=int(run[0]),int(run[-1])+1
            if l<1 or rgt>=w-1 or rgt-l>pitch*.65:continue
            inside=float(np.median(neutral[y,l:rgt]))
            flank=max(2,round(pitch*.06))
            left=float(neutral[y,max(0,l-flank):l].max())
            right=float(neutral[y,rgt:min(w,rgt+flank)].max())
            if min(left,right)-inside<25:continue
            samples.append((y,l,rgt))
        if len(samples)<(hi-lo)*.65:continue
        a=np.float32(samples)
        left=np.polyfit(a[:,0]-h*.5,a[:,1],1);right=np.polyfit(a[:,0]-h*.5,a[:,2],1)
        residual=np.maximum(abs(np.polyval(left,a[:,0]-h*.5)-a[:,1]),abs(np.polyval(right,a[:,0]-h*.5)-a[:,2]))
        if np.percentile(residual,80)>max(2,pitch*.045):continue
        cuts.append({'left':left.tolist(),'right':right.tolist(),'pitch':pitch,
                     'width':float(np.median(a[:,2]-a[:,1])), 'row_agreement':len(samples)/(hi-lo)})
    return sorted(cuts,key=lambda c:c['left'][1])


def split_table_gaps(rgb, quad):
    """Separate independently framed table faces before module reconciliation."""
    q=ordered_quad(quad);edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    w,h=float((edges[0]+edges[2])*.5),float((edges[1]+edges[3])*.5)
    if h>w:q=np.roll(q,-1,axis=0);w,h=h,w
    scale=min(1.,2400/max(w,h));w,h=max(12,round(w*scale)),max(12,round(h*scale))
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H=cv2.getPerspectiveTransform(q,dst);inv=cv2.getPerspectiveTransform(dst,q)
    low=np.maximum(0,np.floor(q.min(0)-2)).astype(int);high=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(q.max(0)+3)).astype(int)
    x,y=low;ex,ey=high
    if ex<=x or ey<=y:return [q],[]
    shift=np.float64([[1,0,x],[0,1,y],[0,0,1]])
    local=cv2.warpPerspective(rgb[y:ey,x:ex],H@shift,(w,h))
    cuts=table_gap_cuts(local)
    def endpoints(line):return [float(np.polyval(line,-h*.5)),float(np.polyval(line,h-1-h*.5))]
    boundaries=[([0.,0.],None)]+[(endpoints(c['right']),endpoints(c['left'])) for c in cuts]+[(None,[w-1.,w-1.])]
    regions=[]
    for before,after in zip(boundaries,boundaries[1:]):
        left,right=before[0],after[1]
        if min(right[0]-left[0],right[1]-left[1])<4:continue
        rect=np.float32([[[left[0],0],[right[0],0],[right[1],h-1],[left[1],h-1]]])
        regions.append(ordered_quad(cv2.perspectiveTransform(rect,inv)[0]))
    return regions or [q],cuts


def physical_boundary_issues(rgb, quad):
    """Recomputed on edited geometry too: inspection cannot waive a table gap."""
    _,cuts=split_table_gaps(rgb,quad)
    if cuts:
        return ['Crosses a confirmed material/table gap: split or correct physical frames']
    q=ordered_quad(quad);edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    w,h=float((edges[0]+edges[2])*.5),float((edges[1]+edges[3])*.5)
    scale=min(1.,1200/max(w,h));w,h=max(12,round(w*scale)),max(12,round(h*scale))
    H=cv2.getPerspectiveTransform(q,np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]))
    low=np.maximum(0,np.floor(q.min(0)-2)).astype(int);high=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(q.max(0)+3)).astype(int)
    x,y=low;ex,ey=high
    if ex<=x or ey<=y:return []
    local=cv2.warpPerspective(rgb[y:ey,x:ex],H@np.float64([[1,0,x],[0,1,y],[0,0,1]]),(w,h)).min(2).astype(np.float32)
    for plane in (local,local.T):
        height,width=plane.shape
        profile=np.percentile(plane[round(height*.15):round(height*.85)],40,axis=0)
        peaks=frame_positions(profile,0,width-1)
        body=profile[round(width*.1):round(width*.9)]
        level=max(float(np.median(body))+35,float(np.percentile(body,90))+12)
        physical=[p[0] for p in peaks if p[1]>level and height*.18<p[0]<width-1-height*.18]
        # One central ridge may be a half-cell busbar. Repeated coarse ridges,
        # each sustained across the face, require physical-module reconciliation.
        if len(physical)>=2 and min(np.diff(physical))>height*.18:
            return ['Encloses repeated coarse physical frames: split merged module faces or correct the perimeter']
    return []


def regularize_table_grid(rgb, quad, proposals):
    """Fit a shared projective rectangular lattice, never across table regions.

    RGB observations constrain one homography for all faces, rather than four
    independent slopes per module. Missing faces are not invented. Outliers and
    incompatible sizes retain their original geometry and require review.
    """
    if len(proposals)<4:return proposals, {'status':'insufficient physical neighbors'}
    q=ordered_quad(quad);edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    w,h=float((edges[0]+edges[2])*.5),float((edges[1]+edges[3])*.5)
    base=cv2.getPerspectiveTransform(q,np.float32([[0,0],[w,0],[w,h],[0,h]]))
    measured=[ordered_quad(cv2.perspectiveTransform(np.float32(p['polygon'])[None],base)[0]) for p in proposals]
    centers=np.asarray([z.mean(0) for z in measured])
    sizes=np.asarray([[np.mean([np.linalg.norm(z[1]-z[0]),np.linalg.norm(z[2]-z[3])]),
                       np.mean([np.linalg.norm(z[3]-z[0]),np.linalg.norm(z[2]-z[1])])] for z in measured])
    typical=np.median(sizes,axis=0)
    usable=np.all(abs(sizes/typical-1)<.25,axis=1)
    # Clipped image-edge faces cannot establish full physical module dimensions.
    for i,p in enumerate(proposals):
        z=np.float32(p['polygon'])
        if np.any(z<1) or np.any(z[:,0]>rgb.shape[1]-2) or np.any(z[:,1]>rgb.shape[0]-2):usable[i]=False
    if usable.sum()<4:return proposals, {'status':'dimensions unresolved'}
    typical=np.median(sizes[usable],axis=0)
    if np.any(np.median(abs(sizes[usable]/typical-1),axis=0)>.12):
        return proposals, {'status':'mixed or unresolved module dimensions'}
    def axis_fit(values,dimension):
        groups=[]
        for i in np.argsort(values):
            if groups and abs(values[i]-np.median(values[groups[-1]]))<dimension*.32:groups[-1].append(i)
            else:groups.append([i])
        positions=np.asarray([np.median(values[a]) for a in groups])
        if len(positions)<2:return None
        differences=np.diff(positions);pitch=float(np.median(differences))
        if pitch<dimension*.75:return None
        index=np.rint((positions-positions[0])/pitch)
        if len(np.unique(index))!=len(index):return None
        slope,offset=np.polyfit(index,positions,1)
        if max(abs(positions-(offset+slope*index)))>dimension*.20:return None
        assigned=np.rint((values-offset)/slope)
        return assigned,float(slope)
    indices=np.where(usable)[0]
    xf=axis_fit(centers[usable,0],typical[0]);yf=axis_fit(centers[usable,1],typical[1])
    if xf is None or yf is None:return proposals, {'status':'no coherent two-dimensional grid'}
    ix,pitch_x=xf;iy,pitch_y=yf
    if len(set(zip(ix,iy))) != len(indices):
        return proposals, {'status':'ambiguous duplicate grid positions; reconcile faces first'}
    # Balance observed rows/columns: a partially visible last column or a row
    # with more detected faces must not erase the actual gutter between rows.
    width=min(float(np.median([np.median(sizes[indices[ix==column],0]) for column in np.unique(ix)])),pitch_x)
    height=min(float(np.median([np.median(sizes[indices[iy==row],1]) for row in np.unique(iy)])),pitch_y)
    canonical=[];observed=[];lattices={}
    for k,i in enumerate(indices):
        cx,cy=ix[k]*pitch_x,iy[k]*pitch_y
        z=np.float32([[cx-width/2,cy-height/2],[cx+width/2,cy-height/2],
                      [cx+width/2,cy+height/2],[cx-width/2,cy+height/2]])
        lattices[int(i)]=z;canonical.extend(z);observed.extend(ordered_quad(proposals[i]['polygon']))
    threshold=max(1.25,float(min(typical))*.022)
    # Independent proposals are the coarse seed, not final frame measurements.
    # Rejecting them at the final 1-2px RGB tolerance prevented coherent glare
    # tables from reaching the joint RGB fit. Keep the tighter tolerance below
    # for actual frame samples and retain all per-face displacement/RGB guards.
    seed_threshold=max(threshold,min(6.,float(min(typical))*.06))
    H,inliers=cv2.findHomography(np.float32(canonical),np.float32(observed),cv2.RANSAC,seed_threshold)
    if H is None or inliers.mean()<.50:return proposals, {'status':'perspective fit unresolved'}
    # One local native RGB plane serves all modules and iterations.
    low=np.maximum(0,np.floor(q.min(0)-16)).astype(int);high=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(q.max(0)+17)).astype(int)
    x,y=low;ex,ey=high
    neutral=cv2.GaussianBlur(rgb[y:ey,x:ex].min(2).astype(np.float32),(0,0),.55)
    radius=max(2.,min(float(min(typical))*.045,6.))
    offsets=np.linspace(-radius,radius,17,dtype=np.float32)
    def edge_samples(z):
        result=[]
        for a,b in zip(z,np.roll(z,-1,axis=0)):
            result.extend(a[None]*(1-np.linspace(.12,.88,12)[:,None])+b[None]*np.linspace(.12,.88,12)[:,None])
        return np.float32(result)
    # Joint line observations: a point may move only normal to its predicted
    # physical frame. All observations update the same projective transform.
    for iteration in range(2):
        src=[];dst=[]
        for z in lattices.values():
            for a,b in zip(z,np.roll(z,-1,axis=0)):
                s=a[None]*(1-np.linspace(.12,.88,12)[:,None])+b[None]*np.linspace(.12,.88,12)[:,None]
                mapped=cv2.perspectiveTransform(np.float32(s)[None],H)[0]
                direction=mapped[-1]-mapped[0];normal=np.float32([-direction[1],direction[0]])/max(1e-6,np.linalg.norm(direction))
                search=mapped[:,None,:]+offsets[None,:,None]*normal
                values=cv2.remap(neutral,(search[:,:,0]-x).astype(np.float32),(search[:,:,1]-y).astype(np.float32),cv2.INTER_LINEAR)
                chosen=np.argmax(values-offsets[None,:]**2*.65,axis=1)
                for j in range(len(s)):
                    if values[j].max()-np.median(values[j])<6:continue
                    src.append(s[j]);dst.append(search[j,chosen[j]])
        if len(src)<16:break
        fitted,mask=cv2.findHomography(np.float32(src),np.float32(dst),cv2.RANSAC,threshold)
        if fitted is None or mask.mean()<.55:break
        H=fitted
    output=[];changed=0;unresolved=[]
    def frame_signal(z):
        points=edge_samples(z)
        return float(np.median(cv2.remap(neutral,(points[:,0]-x)[None].astype(np.float32),(points[:,1]-y)[None].astype(np.float32),cv2.INTER_LINEAR)))
    for i,item in enumerate(proposals):
        item=dict(item)
        if i not in lattices:
            item.update(uncertain=True,grid_issue='Different type, partial face or unresolved dimensions; retained original polygon')
            output.append(item);unresolved.append(i);continue
        old=ordered_quad(item['polygon']);new=ordered_quad(cv2.perspectiveTransform(lattices[i][None],H)[0])
        shift=float(np.linalg.norm(new-old,axis=1).max())
        # The table perimeter is a hard separation constraint (with a frame-
        # width allowance for support-derived table edges). No neighboring-table
        # geometry participates in this fit.
        outside=min(cv2.pointPolygonTest(q,tuple(p),True) for p in new)
        reason=None
        if shift>min(typical)*.22:reason='Grid correction exceeds locally supported displacement'
        elif outside < -max(2.,min(typical)*.025):reason='Grid correction would leave the separated table perimeter'
        elif frame_signal(new)<frame_signal(old)-12:reason='RGB frames do not support the shared grid'
        elif physical_boundary_issues(rgb,new):reason='Grid correction crosses an unresolved physical boundary'
        if reason:
            item.update(uncertain=True,grid_issue=reason);unresolved.append(i)
        else:
            item.update(polygon=new.tolist(),grid_regularized=True,grid_max_corner_shift=shift,
                        grid_rectified_bounds=lattices[i].tolist())
            changed+=1
        output.append(item)
    return output, {'status':'shared projective frame grid; visual review required','regularized':changed,
                    'unresolved_indices':unresolved,'module_size':[width,height],
                    'pitch':[pitch_x,pitch_y],'homography_grid_to_rgb':H.tolist()}


def refine_table(rgb, support, quad, layout=None, regularize=True):
    """Native local rectification, row-local frame detection and four-side fitting.

    layout is an explicit user's table correction, never a global slicing rule.
    """
    q = ordered_quad(quad)
    lengths = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    w = max(12, round(float((lengths[0]+lengths[2])*.5)))
    h = max(12, round(float((lengths[1]+lengths[3])*.5)))
    # Rotation of a region is local; no assumption about the farm's axes.
    if h > w*1.7:
        q = np.roll(q, -1, axis=0)
        w, h = h, w
    scale = min(1., 2400/max(w, h))
    w, h = max(12, round(w*scale)), max(12, round(h*scale))
    pad = 10
    destination = np.float32([[pad, pad], [pad+w, pad], [pad+w, pad+h], [pad, pad+h]])
    to_local = cv2.getPerspectiveTransform(q, destination)
    to_rgb = cv2.getPerspectiveTransform(destination, q)
    # Warp only the native ROI. Full-frame pixels are never copied per module.
    low = np.maximum(0, np.floor(q.min(0)-20)).astype(int)
    high = np.minimum([rgb.shape[1], rgb.shape[0]], np.ceil(q.max(0)+21)).astype(int)
    x1, y1 = low; x2, y2 = high
    if x2 <= x1 or y2 <= y1:
        return [], {"unresolved": "empty region"}
    shift = np.float64([[1, 0, x1], [0, 1, y1], [0, 0, 1]])
    local = cv2.warpPerspective(rgb[y1:y2, x1:x2], to_local @ shift, (w+2*pad+1, h+2*pad+1))
    occupied = cv2.warpPerspective(support[y1:y2, x1:x2], to_local @ shift, (w+2*pad+1, h+2*pad+1)) > 0
    neutral = cv2.GaussianBlur(local.min(axis=2).astype(np.float32), (0, 0), .65)
    row_profile = np.percentile(neutral[:, pad+max(1,w//10):pad+w-max(1,w//10)], 40, axis=1)
    rows = _endpoints(frame_positions(row_profile, pad, pad+h), pad, pad+h)
    if layout:
        count=int(layout["rows"])
        rows=(pad+h*.5+(np.arange(count+1)-count*.5)*(h/count)*layout.get("pitch_y",1.)+layout.get("offset_y",0.)).tolist()
    proposals = []
    for top, bottom in zip(rows, rows[1:]):
        height = bottom-top
        if height < max(9, h*.006):
            continue
        a = max(0, round(top+height*.18)); b = min(len(neutral), round(bottom-height*.18)+1)
        if b <= a:
            continue
        profile = np.percentile(neutral[a:b], 40, axis=0)
        cols = _endpoints(frame_positions(profile, pad, pad+w), pad, pad+w)
        if layout:
            count=int(layout["columns"])
            cols=(pad+w*.5+(np.arange(count+1)-count*.5)*(w/count)*layout.get("pitch_x",1.)+layout.get("offset_x",0.)).tolist()
        for left, right in zip(cols, cols[1:]):
            width = right-left
            if width < max(9, w*.004):
                continue
            if max(width,height)/max(1,min(width,height))>4.5:
                continue
            ix1, iy1 = max(0, round(left+2)), max(0, round(top+2))
            ix2, iy2 = min(occupied.shape[1], round(right-1)), min(occupied.shape[0], round(bottom-1))
            area = occupied[iy1:iy2, ix1:ix2]
            coverage = float(area.mean()) if area.size else 0.
            if coverage < .60:
                continue
            # Fine repeated grid fragments are never promoted as confident modules.
            border_values = [row_profile[int(np.clip(round(p),0,len(row_profile)-1))] for p in (top,bottom)]
            sides = [_fit_side(neutral, top, False, left+width*.12, right-width*.12),
                     _fit_side(neutral, right, True, top+height*.12, bottom-height*.12),
                     _fit_side(neutral, bottom, False, left+width*.12, right-width*.12),
                     _fit_side(neutral, left, True, top+height*.12, bottom-height*.12)]
            def corner(row, side):
                y = (row[0]*side[1]+row[1])/max(.05, 1-row[0]*side[0])
                return [side[0]*y+side[1], y]
            lt, rt, bt, lf = sides
            fitted = np.float32([corner(lt,lf),corner(lt,rt),corner(bt,rt),corner(bt,lf)])
            initial=np.float32([[left,top],[right,top],[right,bottom],[left,bottom]])
            if np.max(abs(fitted-initial))>max(7.,min(width,height)*.15):
                # Unstable line intersections are unresolved, not giant quads.
                continue
            native = cv2.perspectiveTransform(fitted[None], to_rgb)[0]
            native[:, 0] = np.clip(native[:, 0], 0, rgb.shape[1]-1)
            native[:, 1] = np.clip(native[:, 1], 0, rgb.shape[0]-1)
            native = ordered_quad(native)
            if not cv2.isContourConvex(native) or abs(cv2.contourArea(native)) < 24:
                continue
            uncertain = max(width,height)/max(1,min(width,height)) > 3.8 or min(s[2] for s in sides) < .6
            if max(width,height)/max(1,min(width,height)) > 4.5:
                # Cell slivers / long unresolved runs stay region-level evidence,
                # not individual-module instances.
                continue
            proposals.append({"polygon": native.tolist(), "coverage": coverage,
                              "uncertain": bool(uncertain), "frame_inliers": [s[2] for s in sides],
                              "local_size": [width/scale,height/scale]})
    grid={}
    if regularize:proposals,grid=regularize_table_grid(rgb,quad,proposals)
    return proposals, {"quad": ordered_quad(quad).tolist(), "row_boundaries": len(rows),
                       "estimated_rows": max(1,len(rows)-1),
                       "estimated_columns": max(1,round(len(proposals)/max(1,len(rows)-1))),
                       "proposals": len(proposals), "native_roi": [int(x1),int(y1),int(x2),int(y2)],"grid_refinement":grid}


def recover(rgb, progress=None):
    support = material_mask(rgb)
    grouped_regions = regions_from_support(support)
    regions=[]
    original_regions=[]
    for q in grouped_regions:
        separated,_=split_table_gaps(rgb,q)
        regions.extend(separated)
        original_regions.append((q,separated))
    tables, proposals = [], []
    for index, q in enumerate(regions, 1):
        if progress:
            progress(f"Physical frame recovery: region {index}/{len(regions)}")
        children, evidence = refine_table(rgb, support, q,regularize=False)
        table_id = f"T{index:03d}"
        evidence.update(table_id=table_id)
        tables.append(evidence)
        for child in children:
            child["table_id"] = table_id
            proposals.append(child)
    # Keep unaffected instances byte-for-byte. Re-running ridge thresholds on a
    # smaller, glare-heavy table can otherwise turn an existing module into two
    # cell fragments. Replace only gap-crossing / independently resolved merges.
    stable=[]
    for original,separated in original_regions:
        old,_=refine_table(rgb,support,original,regularize=False)
        local=[p for p in proposals if cv2.pointPolygonTest(original,tuple(np.float32(p['polygon']).mean(0)),False)>=0]
        if len(separated)==1:
            stable.extend(local);continue
        pitch=float(np.median([min(p['local_size']) for p in old])) if old else 0
        plausible=[p for p in local if min(p['local_size'])>=pitch*.65]
        unchanged=[];replace=[]
        for item in old:
            q=np.float32(item['polygon']);area=abs(cv2.contourArea(q))
            faces=[p for p in plausible if cv2.pointPolygonTest(q,tuple(np.float32(p['polygon']).mean(0)),False)>=0]
            across=sum(cv2.intersectConvexConvex(q,r)[0]>area*.025 for r in separated)>1
            if across or (len(faces)>=2 and min(item['local_size'])>=pitch*.65):
                replace.append(item)
            else:
                item['table_id']=next((p['table_id'] for p in local if cv2.pointPolygonTest(np.float32(p['polygon']),tuple(q.mean(0)),False)>=0),None)
                if item['table_id'] is None:
                    item['table_id']=min(local,key=lambda p:np.linalg.norm(np.float32(p['polygon']).mean(0)-q.mean(0)))['table_id'] if local else 'UNRESOLVED'
                unchanged.append(item)
        # Refit a neighboring old window too when it cuts through a replacement
        # face; otherwise a skewed old corner suppresses the real adjacent face.
        changed=True
        while changed:
            changed=False
            for face in plausible:
                q=np.float32(face['polygon']);area=abs(cv2.contourArea(q))
                if not any(cv2.intersectConvexConvex(q,np.float32(p['polygon']))[0]>area*.20 for p in replace):continue
                for item in unchanged[:]:
                    r=np.float32(item['polygon'])
                    if cv2.intersectConvexConvex(q,r)[0]>min(area,abs(cv2.contourArea(r)))*.025:
                        unchanged.remove(item);replace.append(item);changed=True
        stable.extend(unchanged)
        for item in plausible:
            q=np.float32(item['polygon']);area=abs(cv2.contourArea(q))
            if not any(cv2.intersectConvexConvex(q,np.float32(p['polygon']))[0]>area*.20 for p in replace):continue
            if any(cv2.intersectConvexConvex(q,np.float32(p['polygon']))[0]>min(area,abs(cv2.contourArea(np.float32(p['polygon']))))*.025 for p in unchanged):continue
            stable.append(item)
        # Preserve unresolved evidence for batch correction, but never as PV.
        for item in replace:
            q=np.float32(item['polygon']);area=abs(cv2.contourArea(q))
            coverage=sum(cv2.intersectConvexConvex(q,np.float32(p['polygon']))[0] for p in stable if p in plausible)/max(1,area)
            if coverage<.80:
                item.update(uncertain=True,boundary_issues=['Unresolved merged physical faces: use table batch correction'],table_id=next((p['table_id'] for p in local),'UNRESOLVED'))
                # Keep only as a table-level issue, not an overlapping instance.
                for table in tables:
                    if cv2.intersectConvexConvex(q,np.float32(table['quad']))[0]>area*.025:
                        table.setdefault('unresolved_regions',[]).append(item['polygon'])
    proposals=stable
    # Reconcile duplicate windows using actual convex intersections.
    kept = []; spatial={}; bin_size=160
    for item in sorted(proposals, key=lambda p: (p["uncertain"], -p["coverage"])):
        q = np.float32(item["polygon"]);area = abs(cv2.contourArea(q));duplicate=False
        low=np.floor(q.min(0)/bin_size).astype(int);high=np.floor(q.max(0)/bin_size).astype(int)
        cells=[(x,y) for y in range(low[1],high[1]+1) for x in range(low[0],high[0]+1)]
        nearby=set(i for cell in cells for i in spatial.get(cell,[]))
        for i in nearby:
            other=kept[i]
            r = np.float32(other["polygon"])
            if np.any(q.max(0)<r.min(0)) or np.any(r.max(0)<q.min(0)):
                continue
            intersection,_=cv2.intersectConvexConvex(q,r)
            if intersection/max(1.,min(area,abs(cv2.contourArea(r)))) > .75:
                duplicate=True;break
        if not duplicate:
            kept.append(item)
            for cell in cells:spatial.setdefault(cell,[]).append(len(kept)-1)
    kept.sort(key=lambda p:(p["table_id"],np.mean(p["polygon"],axis=0)[1],np.mean(p["polygon"],axis=0)[0]))
    for table in tables:
        indices=[i for i,p in enumerate(kept) if p['table_id']==table['table_id']]
        refined,info=regularize_table_grid(rgb,table['quad'],[kept[i] for i in indices])
        for i,p in zip(indices,refined):kept[i]=p
        table['grid_refinement']=info
    for item in kept:
        issues=physical_boundary_issues(rgb,item['polygon'])
        if issues:
            item.update(uncertain=True,boundary_issues=issues)
    return kept, tables, support
