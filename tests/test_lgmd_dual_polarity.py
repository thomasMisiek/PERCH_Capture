"""Synthetic ON/OFF looming controls for both frame and DVS LGMDs."""
import unittest
import numpy as np
import cv2
from src.dvs import FrameToDVS
from src.lgmd_dual_polarity import FrameLGMDDetector, DVSLGMDDetector


class TestONOFFLGMD(unittest.TestCase):
    def sequence(self, background, foreground, expand=True):
        frame_model=FrameLGMDDetector()
        dvs_model=DVSLGMDDetector()
        sensor=FrameToDVS(seed=3101)
        out={"frame":[], "dvs":[]}
        for i in range(30):
            image=np.full((160,160),background,dtype=np.uint8)
            radius=min(45,2+i*3) if expand else 14
            cv2.circle(image,(80,80),radius,foreground,-1)
            full=cv2.resize(np.repeat(image[:,:,None],3,axis=2),
                            (320,320),interpolation=cv2.INTER_NEAREST)
            on,off=sensor.step(full)
            on=on.reshape(160,2,160,2).sum((1,3))
            off=off.reshape(160,2,160,2).sum((1,3))
            out["frame"].append(frame_model.step(image))
            out["dvs"].append(dvs_model.step(on,off))
        return out

    def test_dark_expansion_uses_off(self):
        for name,results in self.sequence(128,12).items():
            self.assertTrue(any(x.basket_alarm and x.channel=="OFF"
                                for x in results), name)

    def test_bright_expansion_uses_on(self):
        for name,results in self.sequence(60,255).items():
            self.assertTrue(any(x.basket_alarm and x.channel=="ON"
                                for x in results), name)

    def test_static_target_no_basket_fire(self):
        for name,results in self.sequence(128,12,expand=False).items():
            self.assertFalse(any(x.basket_alarm for x in results), name)

if __name__=="__main__":
    unittest.main()
