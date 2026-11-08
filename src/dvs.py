"""Canonical PERCH 20 Hz frame-sampled DVS with modest sensor noise.

Original simple stateful log-contrast method; no spatial blur, temporal
low-pass, or visual persistence. RGB frame noise is a separate sensor model.
See README and Hu et al. (2021) v2e. This is NOT a hardware-calibrated DVS.
"""
from __future__ import annotations
import numpy as np

FPS = 20
CONTRAST_THRESHOLD = 0.20
THRESHOLD_MISMATCH_SD = 0.03
BACKGROUND_ACTIVITY_HZ = 0.02
LOG_FLOOR = 0.03
SEED = 2026

def linear_luminance(rgb):
    """sRGB uint8 -> linear Rec.709 luminance, preserving physical contrast."""
    rgb=np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected HxWx3 uint8 RGB")
    c=rgb.astype(np.float32)/255.
    linear=np.where(c <= .04045,c/12.92,((c+.055)/1.055)**2.4)
    return (.2126*linear[...,0]+.7152*linear[...,1]+.0722*linear[...,2]).astype(np.float32)

def frame_sensor_noise(rgb, seed=SEED, full_well_e=10000., read_noise_e=2.):
    """Poisson photon shot noise + Gaussian electronic read noise.

    Map normalized linear intensity to full-well electrons, sample Poisson,
    add Gaussian read noise (electrons), clip and re-encode sRGB.
    Reference indoor-light proxy; not calibrated to a particular camera.
    """
    if full_well_e<=0 or read_noise_e<0:raise ValueError("Invalid sensor noise parameters")
    c=np.asarray(rgb,dtype=np.uint8).astype(np.float32)/255.
    lin=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
    rng=np.random.default_rng(seed)
    electrons=rng.poisson(lin*full_well_e).astype(np.float32)
    if read_noise_e:electrons+=rng.normal(0,read_noise_e,size=electrons.shape).astype(np.float32)
    value=np.clip(electrons/full_well_e,0.,1.)
    srgb=np.where(value<=.0031308,12.92*value,1.055*value**(1/2.4)-.055)
    return np.uint8(np.clip(np.rint(srgb*255),0,255))

class FrameToDVS:
    """ON/OFF log threshold crossings, independent fixed pixel thresholds.

    First image establishes log reference; subsequent 20-Hz images produce
    integer ON/OFF crossing-count maps. Subthreshold residual is retained.
    Noise: fixed Gaussian per-pixel threshold mismatch (not independently
    redrawn each frame), plus sparse background Poisson ON/OFF events.
    DVS input uses CLEAN rendered intensity (not simulated noisy RGB stream),
    preventing RGB read noise from becoming excessive false DVS flicker.
    """
    def __init__(self, threshold=CONTRAST_THRESHOLD, threshold_sd=THRESHOLD_MISMATCH_SD,
                 background_hz=BACKGROUND_ACTIVITY_HZ, fps=FPS, seed=SEED):
        if threshold<=0 or threshold_sd<0 or background_hz<0 or fps<=0:
            raise ValueError("Invalid DVS parameters")
        self.threshold=float(threshold)
        self.threshold_sd=float(threshold_sd)
        self.background_hz=float(background_hz)
        self.fps=float(fps)
        self.rng=np.random.default_rng(seed)
        self.reference=None
        self.c_on=None
        self.c_off=None

    def step(self, rgb):
        luma=linear_luminance(rgb)
        log=np.log(np.maximum(luma,LOG_FLOOR))
        if self.reference is None:
            self.reference=log.copy()
            self.c_on=np.clip(self.rng.normal(self.threshold,self.threshold_sd,size=log.shape),.05,None).astype(np.float32)
            self.c_off=np.clip(self.rng.normal(self.threshold,self.threshold_sd,size=log.shape),.05,None).astype(np.float32)
            zero=np.zeros(log.shape,dtype=np.uint16)
            return zero.copy(),zero
        if log.shape!=self.reference.shape:raise ValueError("DVS shape changed; instantiate new sensor")
        delta=log-self.reference
        on=np.floor(np.maximum(delta,0.)/self.c_on).astype(np.uint16)
        off=np.floor(np.maximum(-delta,0.)/self.c_off).astype(np.uint16)
        self.reference+=on.astype(np.float32)*self.c_on-off.astype(np.float32)*self.c_off
        # Background activity is independent of scene contrast, ON/OFF equally likely.
        if self.background_hz:
            count=self.rng.poisson(self.background_hz/self.fps,size=log.shape)
            polarity=self.rng.random(log.shape)<.5
            on+=np.where(polarity,count,0).astype(np.uint16)
            off+=np.where(~polarity,count,0).astype(np.uint16)
        return on,off

def event_visual(on,off):
    """Immediate events only: black none, ON red, OFF cyan. NO persistence."""
    if on.shape!=off.shape:raise ValueError("Different event map shapes")
    out=np.zeros((*on.shape,3),dtype=np.uint8)
    out[...,0]=np.where(on>0,np.minimum(255,90+75*on.astype(np.int32)),0)
    out[...,1]=np.where(off>0,np.minimum(255,35+65*off.astype(np.int32)),0)
    out[...,2]=np.where(off>0,np.minimum(255,100+65*off.astype(np.int32)),0)
    return out

class DVSDisplay:
    """Compatibility wrapper: display-only mapping, no memory/persistence."""
    def __init__(self,**kwargs):pass
    def step(self,on,off):return event_visual(on,off)
