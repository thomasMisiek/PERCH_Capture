"""Fast structural regression tests for the common bedroom geometry."""
import json
import unittest
import numpy as np
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from src.bedroom import Bedroom, SHELL, SIZE, texture

ROOT=Path(__file__).resolve().parents[1]

class BedroomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim=Bedroom(mode="simulation")
        cls.viz=Bedroom(mode="visualization")

    def test_enclosure_and_geometry_invariants(self):
        shell=set(SHELL)-{"floor_collision"}
        self.assertTrue(shell.issubset(self.sim.objects))
        self.assertFalse(shell.intersection(self.viz.objects))
        self.assertIn("ceiling",self.sim.objects)
        self.assertIn("front_wall",self.sim.objects)
        self.assertIn("right_wall",self.sim.objects)
        self.assertIn("door_panel",self.sim.objects)
        self.assertNotIn("door_panel",self.viz.objects)
        self.assertIn("bed_frame",self.viz.objects)
        self.assertEqual(self.sim.colliders,self.viz.colliders)

    def test_dark_furniture(self):
        def avg(tex):
            return vtk_to_numpy(tex.GetInput().GetPointData().GetScalars()).mean()
        self.assertLess(avg(texture("wood")),avg(texture("floor")))

    def test_snapshot_restriction_and_rgb(self):
        a=self.sim.snapshot((3.43,3.06,.8),heading_deg=95,size=(160,120))
        self.assertEqual(a.shape,(120,160,3))
        self.assertGreater(float(a.std()),8.)
        with self.assertRaises(ValueError):
            self.viz.snapshot((3.43,3.06,.8))

    def test_continuous_360_seam(self):
        from src.panorama import directions
        rays=directions((360,64),heading_deg=30.)
        self.assertTrue(np.allclose(np.linalg.norm(rays,axis=2),1.,atol=1e-6))
        seam=np.linalg.norm(rays[:,0]-rays[:,-1],axis=-1)
        step=np.linalg.norm(rays[:,1]-rays[:,0],axis=-1)
        self.assertTrue(np.allclose(seam,step,atol=1e-5))
        im=self.sim.panorama((3.43,3.06,.8),heading_deg=95,
                                    size=(360,120),face_size=128)
        seam_rgb=np.abs(im[:,0].astype(float)-im[:,-1].astype(float)).mean()
        neighbors=np.abs(im[:,1:].astype(float)-im[:,:-1].astype(float)).mean()
        self.assertLess(seam_rgb, 4*neighbors+7)

    def test_panorama_and_sites(self):
        a=self.sim.panorama((3.43,3.06,.8),size=(360,120),face_size=128)
        self.assertEqual(a.shape,(120,360,3))
        cfg=json.loads((ROOT/"config"/"sites.json").read_text())
        self.assertEqual(cfg["room_dimensions_m"],list(SIZE))
        self.assertEqual([s["id"] for s in cfg["sites"] if s["enabled"]],["base"])
        for s in cfg["sites"]:
            for c, bound in zip(s["position_m"],SIZE):
                self.assertGreaterEqual(c,0)
                self.assertLessEqual(c,bound)

if __name__=="__main__":
    unittest.main()
