"""LGMD-inspired frame-based and simulated-DVS looming detectors.

Both models feed distinct visual inputs through three neural pathways:
an ON+OFF expanding-edge population, an independent ON expansion channel,
and an independent OFF expansion channel. Either polarity can fire.
No target labels, depth, location, future collision timing, or segmentation
enter a detector. The contact timing window is for offline evaluation only.

Scientific limitation: this is not a biophysically faithful locust LGMD.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from collections import deque

class _RadialField:
    """Symmetric orientation subunits and delayed radial expansion pathway."""

    def __init__(self, off_only: bool, halfwidth=46,
                 radii=(3, 5, 7, 10, 14, 20, 28), horizon_frames=3):
        self.off_only = bool(off_only)
        self.h = int(halfwidth)
        self.radii = tuple(radii)
        self.history = deque(maxlen=horizon_frames)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (.707, .707), (.707, -.707),
                           (-.707, .707), (-.707, -.707))
        self.lo = 80 - self.h
        self.hi = 80 + self.h
        axis = np.arange(-self.h, self.h)
        xx, yy = np.meshgrid(axis, axis)
        self.fixation_weight = np.exp(-(xx**2 + yy**2) /
                                      (2 * 30.**2)).astype(np.float32)

    def reset(self):
        self.history.clear()

    def update(self, on: np.ndarray, off: np.ndarray, ffi_weight=.2):
        counts = off if self.off_only else on + off
        energy = np.log1p(counts.astype(np.float32))
        ffi = float(np.mean(energy))
        background = cv2.GaussianBlur(energy, (0, 0), 5.)
        energy = np.maximum(energy - .9 * background, 0.)
        energy = cv2.GaussianBlur(energy, (0, 0), 1.15)

        fields = []
        for radius in self.radii:
            arms = []
            for dx, dy in self.directions:
                delta_x = int(round(dx * radius))
                delta_y = int(round(dy * radius))
                arms.append(energy[self.lo + delta_y:self.hi + delta_y,
                                   self.lo + delta_x:self.hi + delta_x])
            oriented = np.sort(np.stack(arms, axis=0), axis=0)
            fields.append(oriented[3:6].mean(axis=0) *
                          self.fixation_weight)

        peak = 0.
        for earlier in self.history:
            for j in range(1, len(fields)):
                delayed = cv2.dilate(earlier[j - 1], self.kernel)
                peak = max(peak, float(np.minimum(fields[j], delayed).max()))
        self.history.append(fields)
        return peak / (1 + float(ffi_weight) * ffi), ffi





@dataclass(frozen=True)
class NeuralSignal:
    neutral: float
    on: float
    off: float
    ffi: float


class _DualPolarityRetina:
    def __init__(self):
        self.both = _RadialField(off_only=False)
        self.on = _RadialField(off_only=True)
        self.off = _RadialField(off_only=True)

    def step(self, on, off):
        neutral, ffi = self.both.update(on, off)
        on_resp, _ = self.on.update(off, on)
        off_resp, _ = self.off.update(on, off)
        return NeuralSignal(float(neutral), float(on_resp),
                            float(off_resp), float(ffi))


class FrameDualLGMD:
    """Consecutive grayscale frames -> brightness ON/OFF -> LGMD."""
    def __init__(self, frame_threshold=.07, contrast_gain=2.0):
        self.retina = _DualPolarityRetina()
        self.prev = None
        self.th = float(frame_threshold)
        self.gain = float(contrast_gain)

    def step(self, image):
        if image.shape != (160, 160):
            raise ValueError("Expected a 160x160 grayscale image")
        now = np.log(np.maximum(image.astype("float32") / 255, .02))
        if self.prev is None:
            self.prev = now
            z = np.zeros_like(now)
            return self.retina.step(z, z)
        delta = now - self.prev
        self.prev = now
        return self.retina.step(
            self.gain * np.maximum(delta - self.th, 0),
            self.gain * np.maximum(-delta - self.th, 0)
        )


class DVSDualLGMD:
    """20 Hz ON/OFF DVS event-count maps -> LGMD; no brightness frames."""
    def __init__(self):
        self.retina = _DualPolarityRetina()

    def step(self, on, off):
        if on.shape != (160, 160) or off.shape != (160, 160):
            raise ValueError("Both DVS event maps must be 160x160")
        return self.retina.step(on, off)


@dataclass(frozen=True)
class LGMDDecision:
    signal: NeuralSignal
    looming_on: bool
    looming_off: bool
    basket_alarm: bool
    channel: str | None


class DualPolarityTrigger:
    """ON/OFF competition with a causal, refractory basket trigger.

    No 3D distance or collision time is supplied to this trigger.
    """
    def __init__(self, on_thresh, off_thresh, on_gate, off_gate,
                 refractory_frames=20, max_opponent_fraction=.8):
        self.on_thresh = float(on_thresh)
        self.off_thresh = float(off_thresh)
        self.on_gate = float(on_gate)
        self.off_gate = float(off_gate)
        self.refractory_frames = int(refractory_frames)
        self.max_opponent_fraction = float(max_opponent_fraction)
        self.frame_index = 0
        self.last_fire = -self.refractory_frames - 1

    def step(self, signal):
        on, off, neutral = signal.on, signal.off, signal.neutral
        on_looming = on >= self.on_gate
        off_looming = off >= self.off_gate
        # Paired ON/OFF activation is associated with translating edges.
        opponent_ratio = min(on, off) / (max(on, off) + 1.e-9)
        separate = opponent_ratio <= self.max_opponent_fraction
        on_candidate = neutral >= self.on_thresh and on_looming and separate
        off_candidate = neutral >= self.off_thresh and off_looming and separate
        fire = bool((on_candidate or off_candidate)
                    and self.frame_index - self.last_fire >= self.refractory_frames)
        channel = None
        if fire:
            self.last_fire = self.frame_index
            channel = ("ON" if on_candidate and
                       (not off_candidate or on >= off) else "OFF")
        self.frame_index += 1
        return LGMDDecision(signal, bool(on_looming), bool(off_looming),
                            fire, channel)


class FrameLGMDDetector:
    def __init__(self, **kwargs):
        self.neural = FrameDualLGMD()
        self.decision = DualPolarityTrigger(on_thresh=.45, off_thresh=.55,
                                           on_gate=.60, off_gate=.45, **kwargs)

    def step(self, image):
        return self.decision.step(self.neural.step(image))


class DVSLGMDDetector:
    def __init__(self, **kwargs):
        self.neural = DVSDualLGMD()
        self.decision = DualPolarityTrigger(on_thresh=1.65, off_thresh=1.60,
                                           on_gate=.60, off_gate=.60, **kwargs)

    def step(self, on_events, off_events):
        return self.decision.step(self.neural.step(on_events, off_events))
