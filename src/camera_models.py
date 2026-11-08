"""A square 84° x 84° RGB intensity-camera reference for a GENX320-class FoV.

GENX320-style array size: 320x320 pixels. The output is a rendered RGB/grayscale
FRAME, not asynchronous DVS events. Event generation requires temporal
log-intensity differences, contrast thresholds and camera dynamics.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

PROPH_FOV_DEG = (84.0, 84.0)
PROPH_RESOLUTION_PX = (320, 320)

def prophesee_reference_frame(scene, position, heading_deg=0., pitch_deg=0.,
                              resolution=PROPH_RESOLUTION_PX, as_gray=False):
    """Generate a square 84° H/V perspective view of the enclosed room."""
    if getattr(scene, "mode", None) != "simulation":
        raise ValueError("Only the fully enclosed simulation scene produces sensory frames")
    width, height = map(int, resolution)
    if width != height:
        raise ValueError("Square pixels and 84° H/V FoV require equal image dimensions")
    image = scene.snapshot(position=position, heading_deg=heading_deg,
                           pitch_deg=pitch_deg, size=(width, height),
                           horizontal_fov_deg=PROPH_FOV_DEG[0])
    if as_gray:
        return np.asarray(Image.fromarray(image).convert("L"))
    return image

def downsample_84_pixels(image):
    """Optional 84x84-pixel proxy. 84° FoV is an angular, not pixel, dimension."""
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or RGB image")
    return np.asarray(Image.fromarray(image).resize((84,84), Image.Resampling.LANCZOS))
