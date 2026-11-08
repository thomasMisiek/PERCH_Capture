"""Shared 320x320 intensity-frame STMD for stationary and moving cameras.

One motion-facilitated bank, eight equal-magnitude compass velocities.
All direction vectors have norm 1 pixel per 20 Hz observation.
There is no separate local optical-flow response branch. The detector produces a local directional activity map.

No simulator truth, 3-D distance, target species or background template
is required by this detector.
"""
from __future__ import annotations
import numpy as np
import cv2
# Local primitives keep the eight-direction demonstration self-contained.
SIGMA=3.
r=np.arange(-8,9,dtype=np.float32)
xx,yy=np.meshgrid(r,r)
g=np.exp(-(xx*xx+yy*yy)/(2*SIGMA**2));g/=g.sum()
G5=g[6:11,6:11].copy()
G5[(xx[6:11,6:11]**2+yy[6:11,6:11]**2)>2.5**2]=0
WMASS=float(G5.sum())
G5=G5.astype(np.float32)


def shift(arr,dx,dy,border=cv2.BORDER_REFLECT):
    M=np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(arr,M,(SIDE,SIDE),flags=cv2.INTER_LINEAR,borderMode=border)


class MotionBankDetector:
    """Unmodified motion-facilitated 24-channel bank formulas from the STMD repo."""
    def __init__(self,V,decay=.85,std_scale=7.,response_smoothing=.35,
                 density_weight=0.,density_threshold=10.):
        self.V=np.asarray(V,dtype=np.float32)
        self.bg=None;self.previous=None
        self.A=np.zeros((len(self.V),SIDE,SIDE),np.float32)
        self.decay=decay;self.std_scale=std_scale
        self.response_smoothing=response_smoothing
        self.density_weight=density_weight;self.density_threshold=density_threshold
    def step(self,frame,return_map=False):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg=gray.copy();self.previous=gray.copy()
            return {'peak':0.,'activity':np.zeros((SIDE,SIDE),np.float32) if return_map else None}
        dx,dy=cv2.phaseCorrelate(self.previous,gray)[0]
        pred_bg=shift(self.bg,dx,dy)
        residual=np.maximum(pred_bg-gray,0)
        response=np.maximum(cv2.GaussianBlur(residual,(0,0),.55)-cv2.GaussianBlur(residual,(0,0),3.),0)
        mu=(cv2.GaussianBlur(gray,(0,0),SIGMA)-cv2.filter2D(gray,-1,G5))/(1-WMASS)
        variance=(cv2.GaussianBlur(gray*gray,(0,0),SIGMA)-cv2.filter2D(gray*gray,-1,G5))/(1-WMASS)-mu*mu
        sd=np.sqrt(np.maximum(variance,0))
        evidence=response*np.exp(-sd/self.std_scale)
        if self.density_weight:
            active=(residual>self.density_threshold).astype(np.float32)
            density=cv2.GaussianBlur(active,(0,0),3.)
            evidence*=np.exp(-self.density_weight*density)
        if self.response_smoothing:
            evidence=cv2.GaussianBlur(evidence,(0,0),self.response_smoothing)
        for j,(vx,vy) in enumerate(self.V):
            self.A[j]=self.decay*shift(self.A[j],float(vx),float(vy),cv2.BORDER_CONSTANT)+evidence
        peakmap=self.A.max(axis=0)
        peakmap[:12,:]=peakmap[-12:,:]=peakmap[:,:12]=peakmap[:,-12:]=0
        _,best,_,_=cv2.minMaxLoc(peakmap)
        self.bg=.976*pred_bg+.024*gray
        self.previous=gray
        return {'peak':float(best),'activity':peakmap.copy() if return_map else None}


def candidates_from_maps(maps,scales,num=40,accept_floor=.22,sep=5):
    """Source repo's unmodified fused response / local maximum candidate selection."""
    normmaps=np.stack([np.clip(m/max(float(s),1e-5),0,100) for m,s in zip(maps,scales)])
    combo=normmaps.max(axis=0)
    combo[:12,:]=combo[-12:,:]=combo[:,:12]=combo[:,-12:]=0
    dil=cv2.dilate(combo,np.ones((7,7),np.uint8))
    mask=(combo>=dil-1e-6)&(combo>accept_floor)
    ys,xs=np.where(mask)
    if len(xs)==0:return [],combo
    vals=combo[ys,xs]
    indices=np.argsort(vals)[::-1]
    cand=[]
    for i in indices:
        x,y=int(xs[i]),int(ys[i])
        if any((x-c[0])**2+(y-c[1])**2<sep**2 for c in cand):continue
        channel=int(np.argmax(normmaps[:,y,x]))
        cand.append((x,y,float(vals[i]),channel))
        if len(cand)>=num:break
    return cand,combo



SIDE=320
BANK_CONFIG=dict(decay=.85,std_scale=7.,density_weight=0.)
DIRECTIONS=np.array(
    [(dx,dy) for dy in (-1.,0.,1.) for dx in (-1.,0.,1.)
     if dx != 0 or dy != 0],dtype=np.float32)
DIRECTIONS/=np.linalg.norm(DIRECTIONS,axis=1,keepdims=True)

class SingleBankSTMD:
    """STMD-compatible API with eight single-speed directional channels."""
    def __init__(self, scales=None):
        self.bank=MotionBankDetector(DIRECTIONS,**BANK_CONFIG)
        self.banks=[self.bank]
        self.scales=(np.ones(1,np.float32) if scales is None
                     else np.asarray(scales,dtype=np.float32))
        if self.scales.shape!=(1,):
            raise ValueError("SingleBankSTMD expects exactly one calibration scale")
    def step(self,frame,return_map=False):
        response=self.bank.step(frame,return_map=True)["activity"]
        candidates,combined=candidates_from_maps([response],self.scales)
        best=candidates[0] if candidates else (np.nan,np.nan,0.,-1)
        return dict(x=float(best[0]),y=float(best[1]),
                    score=float(best[2]),channel=int(best[3]),
                    candidate_count=len(candidates),
                    activity=combined if return_map else None,
                    bank_peaks=[float(response.max())])
    def flush_motion_memory(self):
        self.bank.A.fill(0.)

def calibrate_single_bank(frames):
    """Same target-free 99th-percentile normalization as the STMD interface."""
    detector=SingleBankSTMD()
    samples=np.asarray([detector.step(frame)["bank_peaks"] for frame in frames],
                       dtype=np.float64)
    return np.maximum(np.quantile(
        samples[max(0,len(samples)//4):],.99,axis=0),1e-5).astype(np.float32)
