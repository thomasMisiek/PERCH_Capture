"""Continuous equirectangular 360-degree RGB sensor from six VTK cube faces.

All six cameras render from exactly the same position in the fully enclosed
room. Each spherical output pixel samples its own ray in 3D; unlike tiled
perspective images, this has continuous angular sampling and a wrapping seam.
"""
from __future__ import annotations
import numpy as np

# face name, forward, view-up in the room's z-up coordinate system
CUBE_FACES = (
    ('+x', (1, 0, 0), (0, 0, 1)),
    ('-x', (-1, 0, 0), (0, 0, 1)),
    ('+y', (0, 1, 0), (0, 0, 1)),
    ('-y', (0, -1, 0), (0, 0, 1)),
    ('+z', (0, 0, 1), (0, 1, 0)),
    ('-z', (0, 0, -1), (0, 1, 0)),
)

def directions(size, heading_deg=0.0, vertical_fov_deg=110.0):
    """Pixel-center 3D rays, equal angular intervals in azimuth/elevation.

    The center column looks along heading_deg. Horizontal azimuth wraps at
    the two image ends. Size is (width,height) in pixels.
    """
    width, height = map(int, size)
    if width < 2 or height < 2:
        raise ValueError('Both panorama dimensions must be >= 2')
    if not 0. < vertical_fov_deg <= 180.:
        raise ValueError('vertical_fov_deg must be within (0,180]')
    yaw = np.deg2rad(heading_deg) + ((np.arange(width) + .5) / width - .5) * 2 * np.pi
    elevation = (.5 - (np.arange(height) + .5) / height) * np.deg2rad(vertical_fov_deg)
    lat, az = np.meshgrid(elevation, yaw, indexing='ij')
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(az), cos_lat * np.sin(az), np.sin(lat)),axis=-1)

def _bilinear(face, xpos, ypos):
    """Interpolate perspective face colors at floating-point pixel positions."""
    h, w = face.shape[:2]
    xpos = np.clip(xpos, 0., w-1.)
    ypos = np.clip(ypos, 0., h-1.)
    x0 = np.floor(xpos).astype(np.int32)
    y0 = np.floor(ypos).astype(np.int32)
    x1 = np.minimum(x0+1, w-1)
    y1 = np.minimum(y0+1, h-1)
    sx = (xpos-x0)[:,None]
    sy = (ypos-y0)[:,None]
    a = face[y0,x0].astype(np.float32)*(1.-sx) + face[y0,x1].astype(np.float32)*sx
    b = face[y1,x0].astype(np.float32)*(1.-sx) + face[y1,x1].astype(np.float32)*sx
    return np.clip(a*(1.-sy)+b*sy,0,255).astype(np.uint8)

def render_equirectangular(scene,position,heading_deg=0.,size=(360,100),
                           vertical_fov_deg=110.,face_size=None,face_fov_deg=92.):
    """True continuous 360° equirectangular RGB projection.

    Uses overlapping six-direction cubemap faces. Output covers 360° azimuth
    and the given vertical FoV. For full spherical 360x180 choose 180°.
    The underlying VTK camera is not a biologically calibrated insect eye.
    """
    if getattr(scene,'mode',None) != 'simulation':
        raise ValueError('Sensory panoramas require the enclosed simulation scene')
    width, height = map(int,size)
    if face_size is None:
        face_size = max(128,min(1400,int(np.ceil(width/4.))))
    if not 90. < face_fov_deg < 110.:
        raise ValueError('Face FoV must overlap the cube edges (e.g. 92 degrees)')
    rays = directions(size,heading_deg,vertical_fov_deg)
    flat = rays.reshape(-1,3)
    dominant = np.argmax(np.abs(flat),axis=1)
    face_index = np.empty(flat.shape[0],dtype=np.uint8)
    for i,(_,forward,_) in enumerate(CUBE_FACES):
        forward = np.asarray(forward,dtype=np.float64)
        axis = int(np.argmax(np.abs(forward)))
        face_index[(dominant==axis)&(np.sign(flat[:,axis])==np.sign(forward[axis]))] = i
    output = np.zeros((len(flat),3),dtype=np.uint8)
    eye = np.asarray(position,dtype=np.float64)
    extent = np.tan(np.deg2rad(face_fov_deg/2))
    for i,(_,look,up) in enumerate(CUBE_FACES):
        indices = np.flatnonzero(face_index==i)
        if not len(indices):
            continue
        forward = np.asarray(look,dtype=np.float64)
        up = np.asarray(up,dtype=np.float64)
        right = np.cross(forward,up)
        up = np.cross(right,forward)
        face = scene.rgb(eye,eye+forward,size=(face_size,face_size),
                         vertical_fov_deg=face_fov_deg,view_up=up)
        selected = flat[indices]
        depth = selected@forward
        u = (selected@right)/depth/extent
        v = (selected@up)/depth/extent
        px = (u+1)*face_size/2-.5
        py = (1-v)*face_size/2-.5
        output[indices] = _bilinear(face,px,py)
    return output.reshape(height,width,3)


class CachedEquirectangularRenderer:
    """Precompute per-pixel six-face lookup once; render one 360 panorama per pose.

    Identical ray/face mapping to render_equirectangular, but avoids recomputing
    directions, face indices and texture coordinates 101 times in a flight.
    Use (360,100) output with 192-pixel cube faces for stable fine edges.
    """
    def __init__(self, size=(360,100), heading_deg=0.0, vertical_fov_deg=110.0,
                 face_size=192, face_fov_deg=92.0):
        self.size=tuple(map(int,size)); self.face_size=int(face_size)
        self.face_fov_deg=float(face_fov_deg)
        self.heading_deg=float(heading_deg)
        self.vertical_fov_deg=float(vertical_fov_deg)
        if self.face_size<16 or not 90<face_fov_deg<110:
            raise ValueError('Cube-face size must be >=16 and field of view overlap 90 degrees')
        flat=directions(self.size,heading_deg,vertical_fov_deg).reshape(-1,3)
        dominant=np.argmax(np.abs(flat),axis=1)
        extent=np.tan(np.deg2rad(face_fov_deg/2))
        self.faces=[]
        for i,(name,look,up) in enumerate(CUBE_FACES):
            forward=np.asarray(look,dtype=float)
            up=np.asarray(up,dtype=float)
            axis=np.argmax(np.abs(forward))
            ids=np.flatnonzero((dominant==axis)&(np.sign(flat[:,axis])==np.sign(forward[axis])))
            if not len(ids):continue
            right=np.cross(forward,up)
            up=np.cross(right,forward)
            selected=flat[ids]
            depth=selected@forward
            u=(selected@right)/depth/extent
            v=(selected@up)/depth/extent
            px=(u+1)*self.face_size/2-.5
            py=(1-v)*self.face_size/2-.5
            self.faces.append((ids,forward,up,px,py))

    def render(self,scene,position):
        if getattr(scene,'mode',None) != 'simulation':
            raise ValueError('Only enclosed simulation scenes can produce sensory panoramas')
        eye=np.asarray(position,dtype=float)
        out=np.zeros((self.size[0]*self.size[1],3),dtype=np.uint8)
        for ids,forward,up,px,py in self.faces:
            face=scene.rgb(eye,eye+forward,size=(self.face_size,self.face_size),
                           vertical_fov_deg=self.face_fov_deg,view_up=up)
            out[ids]=_bilinear(face,px,py)
        return out.reshape(self.size[1],self.size[0],3)
