"""Reusable deterministic 3D bedroom scene for navigation experiments.
Coordinates in metres; room 7 x 6.2 x 2.8, z-up.
Run: python src/bedroom.py --output media
"""
import argparse
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PIL import Image

try:
    from src.panorama import render_equirectangular
except ModuleNotFoundError:  # when launched directly as src/bedroom.py
    from panorama import render_equirectangular

SIZE = (7.0, 6.2, 2.8)
SEED = 2026
SCENE_MODES = ("simulation", "visualization")
# The full 3D shell is always physically present as collision metadata;
# only its visibility differs across the two independent VTK scene objects.
SHELL = {
    "left_wall":(-.07,0,0,6.2,0,2.8),
    "right_wall":(7,7.07,0,6.2,0,2.8),
    "front_wall":(0,7,-.07,0,0,2.8),
    "rear_wall_left":(0,2.55,6.20,6.27,0,2.8),
    "rear_wall_right":(4.55,7,6.20,6.27,0,2.8),
    "rear_wall_below_window":(2.55,4.55,6.20,6.27,0,.94),
    "rear_wall_above_window":(2.55,4.55,6.20,6.27,2.35,2.8),
    "ceiling":(0,7,0,6.2,2.8,2.88),
    "floor_collision":(0,7,0,6.2,-.12,0),
}

def texture(kind, size=512):
    rng = np.random.default_rng(SEED + sum(map(ord, kind)))
    y, x = np.mgrid[:size, :size]
    noise = rng.normal(0, 1, (size, size))
    if kind == "wood":
        stripe = np.sin(x*.13 + np.sin(y*.018)*2) + .35*np.sin(x*.49)
        a = np.stack([103+12*stripe,72+9*stripe,48+8*stripe],-1)+noise[...,None]*4
    elif kind == "floor":
        plank = (y//68)%3
        grain = np.sin(x*.17 + 3*np.sin(y*.012))
        a = np.stack([175+11*grain+plank*5,132+9*grain+plank*4,91+7*grain+plank*3],-1)+noise[...,None]*4
        a[y%68<3] = [87,61,43]
    elif kind in ("fabric","blue"):
        weave = ((x%7<2).astype(float)+(y%7<2).astype(float))*7
        base = [208,209,205] if kind=="fabric" else [66,91,122]
        a = np.stack([v+weave for v in base],-1)+noise[...,None]*5
    elif kind=="rug":
        motif = ((x//45+y//45)%2)*23
        a = np.stack([174+motif,156+motif,142+motif],-1)+noise[...,None]*5
    elif kind=="wall":
        a = np.stack([222+noise*2,213+noise*2,200+noise*2],-1)
    elif kind=="leaves":
        v = (np.sin(x*.04)*np.cos(y*.07)+1)*24
        a = np.stack([44+v,91+v,46+v*.4],-1)+noise[...,None]*6
    else:
        raise ValueError(kind)
    pixels = np.uint8(np.clip(a,0,255))
    image = vtk.vtkImageData()
    image.SetDimensions(size,size,1)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,3)
    image.GetPointData().SetScalars(numpy_to_vtk(pixels[::-1].reshape(-1,3),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR))
    tex = vtk.vtkTexture()
    tex.SetInputData(image)
    tex.InterpolateOn()
    tex.RepeatOn()
    tex.MipmapOn()  # Filter fine texture detail as the camera moves.
    return tex

class Bedroom:
    """Reusable geometry: enclosed for sensory simulation, wall-free for visualization.

    The two objects have identical non-structural geometry and collision AABBs.
    """
    def __init__(self, mode="simulation"):
        if mode not in SCENE_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected {SCENE_MODES}")
        self.mode = mode
        self.room_size = SIZE
        self.rendered_shell = []
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(.71,.80,.88)
        self.renderer.SetAmbient(.32,.32,.32)
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)
        self.window.SetMultiSamples(4)  # Reduce unstable subpixel geometry flicker.
        self.textures = {k:texture(k) for k in ("wood","floor","fabric","blue","rug","wall","leaves")}
        self.objects = {}
        self.colliders = []
        for pos,focus,power in [((3,2.5,2.6),(3,3,0),1.25),((6.8,1,2.3),(2,3,.7),.65),((.6,5.8,2.4),(4,3,.8),.55)]:
            light=vtk.vtkLight()
            light.SetPosition(*pos)
            light.SetFocalPoint(*focus)
            light.SetIntensity(power)
            self.renderer.AddLight(light)
        self.build()

    def cube(self,name,bounds,color=(1,1,1),material=None,collision=False):
        source=vtk.vtkCubeSource()
        source.SetBounds(*bounds)
        mapper=vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor=vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetAmbient(.38)
        actor.GetProperty().SetDiffuse(.62)
        actor.GetProperty().SetSpecular(.09)
        if material:
            actor.SetTexture(self.textures[material])
        self.renderer.AddActor(actor)
        self.objects[name]=actor
        if collision:
            self.colliders.append((name,bounds))
        return actor

    def ball(self,name,center,radius,color,scale=(1,1,1)):
        source=vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetThetaResolution(20)
        source.SetPhiResolution(16)
        mapper=vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor=vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(*center)
        actor.SetScale(*scale)
        actor.GetProperty().SetColor(*color)
        self.renderer.AddActor(actor)
        self.objects[name]=actor

    def build(self):
        c=self.cube
        # Both scenes expose the SAME physical enclosure/collision metadata.
        self.colliders.extend(SHELL.items())
        c("floor",(0,7,0,6.2,-.09,-.035),material="floor")
        if self.mode == "simulation":
            for name,bounds in SHELL.items():
                if name != "floor_collision":
                    c(name,bounds,material="wall" if name!="ceiling" else "fabric")
                    self.rendered_shell.append(name)
            # The shut door is an interior landmark, not a navigable opening.
            c("door_panel",(1.30,2.31,.003,.035,.03,2.17),(.30,.21,.16))
            c("door_trim_left",(1.26,1.31,.004,.06,0,2.23),(.63,.55,.45))
            c("door_trim_right",(2.31,2.36,.004,.06,0,2.23),(.63,.55,.45))
            c("door_trim_top",(1.26,2.36,.004,.06,2.17,2.23),(.63,.55,.45))
            c("door_handle",(2.13,2.22,.038,.066,1.03,1.07),(.80,.76,.61))
        for yy in (0,6.13):
            c(f"skirting_{yy}",(0,7,yy,yy+.065,0,.12),(.65,.60,.54))
        c("window_frame",(2.55,4.55,6.13,6.2,.94,2.35),(.96,.94,.89))
        c("window_glass",(2.65,4.45,6.10,6.13,1.04,2.24),(.41,.66,.77))
        c("window_trees",(2.69,4.41,6.075,6.10,1.06,1.75),material="leaves")
        for xx in (2.63,3.52,4.43):
            c(f"mullion_{xx}",(xx,xx+.055,6.06,6.14,1,2.3),(.93,.91,.86))
        c("window_sill",(2.47,4.62,5.97,6.2,.89,.96),(.88,.84,.78))
        for i in range(12):
            c(f"radiator_{i}",(2.67+i*.145,2.74+i*.145,6.03,6.09,.22,.83),(.88,.87,.82))
        c("bed_frame",(.42,2.65,3.75,5.85,.13,.43),material="wood",collision=True)
        c("mattress",(.50,2.57,3.83,5.76,.43,.65),material="fabric")
        c("duvet",(.52,2.55,3.91,5.15,.66,.76),material="blue")
        c("duvet_fold",(.52,2.55,5.06,5.20,.77,.81),material="fabric")
        c("headboard",(.39,2.69,5.74,5.89,.15,1.28),material="wood")
        for i,x in enumerate((1.05,2.03)):
            self.ball(f"pillow_{i}",(x,5.43,.77),.33,(.91,.91,.86),(1.25,.7,.24))
        c("nightstand",(.05,.37,4.18,5.10,0,.67),material="wood",collision=True)
        c("nightstand_top",(.04,.42,4.14,5.13,.66,.72),material="wood")
        c("lamp_stem",(.19,.24,4.54,4.60,.73,1.22),(.78,.70,.54))
        c("lamp_shade",(.09,.34,4.42,4.72,1.20,1.44),(.98,.85,.59))
        c("carpet",(2.24,5.42,1.83,4.28,-.027,-.012),material="rug")
        for i in range(8):
            x=2.40+i*.39
            c(f"rug_band_{i}",(x,x+.055,1.93,4.17,-.011,-.006),(.75,.66,.60))
        c("desk_top",(4.55,6.73,.36,1.28,.70,.78),material="wood",collision=True)
        for x in (4.66,6.58):
            for y in (.45,1.16):
                c(f"desk_leg_{x}_{y}",(x,x+.065,y,y+.065,0,.71),material="wood")
        c("monitor_base",(5.48,5.84,.65,.89,.79,.82),(.13,.15,.18))
        c("monitor",(5.19,6.13,.92,.99,.89,1.53),(.12,.16,.21))
        c("monitor_screen",(5.25,6.07,.915,.922,.95,1.47),(.14,.34,.49))
        c("keyboard",(5.24,6.08,.45,.66,.78,.805),(.24,.25,.27))
        c("chair_seat",(4.27,4.93,1.40,2.08,.46,.55),material="blue",collision=True)
        c("chair_back",(4.27,4.93,2.03,2.15,.52,1.24),material="blue")
        for x in (4.36,4.82):
            for y in (1.5,1.97):
                c(f"chair_leg_{x}_{y}",(x,x+.04,y,y+.04,0,.47),(.22,.23,.25))
        c("wardrobe",(5.86,6.79,3.77,5.90,0,2.42),material="wood",collision=True)
        for i,y in enumerate((3.79,4.86)):
            c(f"wardrobe_door_{i}",(5.83,5.86,y,y+1.02,.06,2.35),(.38,.29,.23))
            c(f"wardrobe_handle_{i}",(5.80,5.83,y+.79,y+.82,1.05,1.50),(.76,.75,.69))
        c("bookcase",(.10,.78,.49,2.37,0,2.18),material="wood",collision=True)
        for z in (.45,.92,1.39,1.88):
            c(f"bookshelf_{z}",(.08,.80,.46,2.40,z,z+.055),material="wood")
        palette=[(.25,.41,.63),(.69,.35,.26),(.27,.51,.37),(.77,.63,.31)]
        for i in range(30):
            y=.59+(i%10)*.17
            z=.52+(i//10)*.47
            c(f"book_{i}",(.07,.14,y,y+.12,z,z+.30),palette[i%4])
        c("dresser",(2.84,3.75,4.50,5.18,0,.91),material="wood",collision=True)
        for z in (.21,.48,.74):
            c(f"drawer_{z}",(2.82,2.845,4.56,5.12,z,z+.19),(.41,.32,.25))
            c(f"handle_{z}",(2.80,2.82,4.77,4.93,z+.08,z+.10),(.83,.81,.72))
        for i,y in enumerate((.90,1.80)):
            c(f"art_frame_{i}",(-.02,.02,y,y+.65,1.42,2.10),(.30,.25,.22))
            c(f"art_{i}",(.02,.025,y+.06,y+.59,1.49,2.04),(.48+i*.12,.60,.58))
        c("plant_pot",(6.20,6.59,2.54,2.95,0,.43),(.57,.37,.27),collision=True)
        for i in range(16):
            a=i*2*np.pi/16
            r=.18+(i%3)*.07
            self.ball(f"leaf_{i}",(6.39+r*np.cos(a),2.74+r*np.sin(a),.63+(i%5)*.12),.19,(.19,.38+(i%3)*.05,.18),(.75,.56,1.5))
        c("nest_marker",(3.39,3.47,3.03,3.11,-.003,.005),(.90,.36,.16))

    def rgb(self, position, focal, size=(640,480), vertical_fov_deg=48, view_up=(0,0,1)):
        """Actual VTK RGB raster, not a physically calibrated camera."""
        self.window.SetSize(*size)
        camera=self.renderer.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(*view_up)
        camera.SetViewAngle(float(vertical_fov_deg))
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        capture=vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        arr=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
        return np.flipud(arr.reshape(size[1],size[0],3)).copy()

    def render(self,path,position,focal=(3.45,3.1,.85),size=(1500,1000)):
        """Render an external view (use visualization mode)."""
        Image.fromarray(self.rgb(position,focal,size)).save(path)

    def snapshot(self, position, heading_deg=0., pitch_deg=0., size=(640,480),
                 horizontal_fov_deg=80.):
        """Enclosed-scene RGB perspective camera. Yaw 0=+x, 90=+y."""
        if self.mode != "simulation":
            raise ValueError("Snapshots must use enclosed simulation geometry")
        if not 1. <= horizontal_fov_deg < 170.:
            raise ValueError("Perspective horizontal FoV must be 1..170 degrees")
        heading=np.deg2rad(heading_deg)
        pitch=np.deg2rad(pitch_deg)
        direction=np.array([np.cos(pitch)*np.cos(heading),
                            np.cos(pitch)*np.sin(heading),np.sin(pitch)])
        target=np.asarray(position,dtype=float)+direction
        aspect=size[0]/size[1]
        vfov=np.degrees(2*np.arctan(np.tan(np.deg2rad(horizontal_fov_deg)/2)/aspect))
        return self.rgb(position,target,size,vertical_fov_deg=vfov)

    def panorama(self, position, heading_deg=0., size=(360,100),
                 vertical_fov_deg=110., face_size=None):
        """Continuous 360-degree equirectangular RGB panorama from enclosed scene.

        Horizontal FoV is always the full 360 degrees. Set vertical_fov_deg=180
        for the whole sphere or use a narrower band for a visual navigator.
        """
        return render_equirectangular(
            self,position,heading_deg=heading_deg,size=size,
            vertical_fov_deg=vertical_fov_deg,face_size=face_size)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="media")
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    display=Bedroom(mode="visualization")
    simulation=Bedroom(mode="simulation")
    views={
        "bedroom_angle_1.png":((10,-7,6.5),(3.35,3.05,.85)),
        "bedroom_angle_2.png":((-7,-3,5.0),(3.25,3.0,.85)),
        "bedroom_angle_3.png":((9,10,5.8),(3.4,3.0,.75)),
    }
    for filename,(position,focus) in views.items():
        display.render(out/filename,position,focus)
        print(out/filename)
    # All sensory images come from the same eye in the full enclosed scene.
    try:
        from src.camera_models import prophesee_reference_frame
    except ModuleNotFoundError:  # direct execution with python src/bedroom.py
        from camera_models import prophesee_reference_frame
    Image.fromarray(simulation.panorama((3.43,3.06,.8),heading_deg=95)).save(out/"interior_panorama.png")
    print(out/"interior_panorama.png")
    camera=prophesee_reference_frame(simulation,(3.43,3.06,.8),heading_deg=95.)
    Image.fromarray(camera).save(out/"prophesee_84deg_320px.png")
    print(out/"prophesee_84deg_320px.png")
    print(f"Cutaway actors: {len(display.objects)}; simulation actors: {len(simulation.objects)}")
    print(f"Identical collision AABBs: {display.colliders == simulation.colliders}")

if __name__=="__main__":
    main()
