"""Physically positioned, inertial correlated-random 3D flying insect agents.

Collision handling uses expanded 3D axis-aligned bounding boxes (AABBs), with a
10 cm clearance from solid room/furniture geometry; not a 6-DOF aerodynamic model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ROOM=np.array([7.,6.2,2.8])
CLEARANCE_M=.10
SPECIES={
    'fly':{'diameter_mm':5.,'speed_m_s':.55,'steer_time_s':.45},
    'mosquito':{'diameter_mm':7.,'speed_m_s':.40,'steer_time_s':.75},
    'hornet':{'diameter_mm':20.,'speed_m_s':.80,'steer_time_s':1.1},
}

@dataclass(frozen=True)
class AABB:
    name:str
    lo:tuple[float,float,float]
    hi:tuple[float,float,float]

    @classmethod
    def from_vtk(cls,name,bounds):
        return cls(name,(float(bounds[0]),float(bounds[2]),float(bounds[4])),
                   (float(bounds[1]),float(bounds[3]),float(bounds[5])))
    def within_margin(self,p,margin):
        return bool(np.all(p>=np.asarray(self.lo)-margin) and np.all(p<=np.asarray(self.hi)+margin))
    def repel(self,p,margin,range_m):
        p=np.asarray(p,float)
        lo=np.asarray(self.lo)-margin;hi=np.asarray(self.hi)+margin
        closest=np.clip(p,lo,hi)
        vec=p-closest
        dist=float(np.linalg.norm(vec))
        if 1e-9<dist<range_m:return (vec/dist)*(range_m-dist)/range_m
        if dist>=range_m:return np.zeros(3)
        # Inside inflated obstacle: push toward closest face from its inside.
        d=np.r_[p-lo,hi-p]
        axis=int(np.argmin(d));result=np.zeros(3)
        result[axis%3]= -1. if axis<3 else 1.
        return result*2.0

class InsectAgent:
    """Inertial direction with correlated stochastic steering and obstacle avoidance.

    Speed is an explicit indoor simulation assumption. Each agent uses its own
    deterministic RNG; direction evolves over time rather than jumping randomly.
    A proposed step is only accepted when its swept segment remains >=0.1m from
    walls and collider AABBs. Rejected motions are steered to valid alternatives.
    """
    def __init__(self,species='mosquito',position=(1.,3.,1.),velocity=None,
                 seed=2026,margin=CLEARANCE_M,obstacles=()):
        if species not in SPECIES:raise ValueError('species must be fly, mosquito, hornet')
        self.species=species;self.diameter_m=SPECIES[species]['diameter_mm']/1000.
        self.speed_m_s=SPECIES[species]['speed_m_s'];self.steer_time_s=SPECIES[species]['steer_time_s']
        self.rng=np.random.default_rng(seed);self.seed=int(seed)
        self.margin=float(margin);self.obstacles=tuple(obstacles)
        self.pos=np.array(position,dtype=float)
        if not self.valid(self.pos):raise ValueError('initial insect position violates 10cm obstacle clearance')
        vel=np.asarray(velocity if velocity is not None else [0.,1.,.15],dtype=float)
        vel/=max(np.linalg.norm(vel),1e-6)
        self.velocity=self.speed_m_s*vel
        self.steering=np.zeros(3,dtype=float)
        self.avoidance_count=0
        self.blocked_count=0
    def valid(self,p):
        if np.any(p<self.margin) or np.any(p>ROOM-self.margin):return False
        return not any(b.within_margin(p,self.margin) for b in self.obstacles)
    def clear_segment(self,a,b):
        # Numerical collision checks including intermediate points: bounded dt<=0.05s.
        return all(self.valid(a+(b-a)*t) for t in np.linspace(0,1,6))
    def step(self,dt=.05,escape_heading=None,escape_turn_rate_rad_s=6.0):
        if not 0<dt<=.1:raise ValueError('0 < dt <= 0.1 sec required')
        alpha=1-np.exp(-dt/self.steer_time_s)
        self.steering=(1-alpha)*self.steering+np.sqrt(2*alpha-alpha*alpha)*self.rng.normal(0,1,3)
        # Avoidance samples prospective next ~0.4 seconds of motion, but never less than 10 cm.
        look=self.pos+.4*self.velocity
        repel=np.zeros(3)
        lower=np.maximum(.0,.35-look);upper=np.maximum(.0,look-(ROOM-.35))
        repel+=lower-upper
        for b in self.obstacles:repel+=b.repel(look,self.margin,.32)
        if np.linalg.norm(repel)>1e-8:self.avoidance_count+=1
        desired=self.velocity/self.speed_m_s + .20*self.steering+3.2*repel
        if escape_heading is not None:
            # Optional idealized threat-escape control. Default behaviour remains
            # unchanged when escape_heading is absent (including RNG draws).
            ev=np.asarray(escape_heading,dtype=float)
            ev=ev/max(float(np.linalg.norm(ev)),1e-10)
            desired=2.5*ev+3.2*repel
        desired/=max(np.linalg.norm(desired),1e-8)
        # Bounded steering rate for inertial motion; avoidance can turn faster.
        blend=min(1.,dt*(1.8 if np.linalg.norm(repel)<.1 else 6.0))
        if escape_heading is not None:
            blend=max(blend,min(1.,dt*float(escape_turn_rate_rad_s)))
        direction=(1-blend)*self.velocity/self.speed_m_s +blend*desired
        direction/=max(np.linalg.norm(direction),1e-8)
        trial=self.pos+direction*self.speed_m_s*dt
        if not self.clear_segment(self.pos,trial):
            self.blocked_count+=1
            # Project out motion into nearby obstacle; try deterministically
            # rotated headings before choosing immobility (never cross solid AABBs).
            options=[]
            for vec in [repel,-self.velocity]+[self.rng.normal(size=3) for _ in range(20)]:
                norm=np.linalg.norm(vec)
                if norm<1e-8:continue
                vec=vec/norm
                goal=self.pos+vec*self.speed_m_s*dt
                if self.clear_segment(self.pos,goal):
                    options.append((float(np.dot(vec,direction)),vec))
            if options:
                direction=max(options,key=lambda x:x[0])[1]
                trial=self.pos+direction*self.speed_m_s*dt
            else:
                direction=np.zeros(3)
                trial=self.pos.copy()
        self.velocity=direction*self.speed_m_s
        self.pos=trial
        return self.pos.copy()

class PerchRobot:
    """Immobile observer at the configured charging base B, facing into the room.

    config/sites.json stores external visualization pose for B (yaw 180,
    pitch -40 toward the base). The on-board optical axis is the reverse:
    yaw 0°, pitch +40° from the base into the room. No locomotion occurs.
    """
    def __init__(self,site=(.30,3.05,.12),yaw_deg=0.,pitch_deg=40.):
        self.position=np.asarray(site,dtype=float)
        self.yaw_deg=float(yaw_deg);self.pitch_deg=float(pitch_deg)
    @property
    def forward(self):
        yaw=np.deg2rad(self.yaw_deg);pitch=np.deg2rad(self.pitch_deg)
        return np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
    @property
    def right(self):
        yaw=np.deg2rad(self.yaw_deg)
        return np.array([np.sin(yaw),-np.cos(yaw),0.])
    @property
    def up(self):return np.cross(self.right,self.forward)
    def spawn(self,distance_m):return self.position+self.forward*float(distance_m)
    def project(self,position,fov_deg=84.,side=320):
        rel=np.asarray(position)-self.position
        depth=float(np.dot(rel,self.forward))
        if depth<=0:return np.nan,np.nan,depth,False
        f=(side/2)/np.tan(np.deg2rad(fov_deg/2))
        u=(side-1)/2+f*np.dot(rel,self.right)/depth
        v=(side-1)/2-f*np.dot(rel,self.up)/depth
        return float(u),float(v),depth,bool(0<=u<side and 0<=v<side)