"""Baseline implementations adapted from SAREnv (MIT; see reference/LICENSE).

Systematic geometry, path splitting, and budget semantics match upstream.
GreedySeeded preserves upstream 8-neighbor myopic rule, but supplies an explicit
RNG and caches local footprints for reproducibility and lower overhead.
These are baseline planners, not GA/NSGA-II and not claimed novel algorithms.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring
from .data import Environment

@dataclass(frozen=True)
class Config:
    num_drones: int = 5
    budget_m: float = 100000.0  # TEAM total, split equally; benchmark convention
    fov_deg: float = 45.0
    altitude_m: float = 80.0
    overlap: float = 0.0
    spacing_m: float = 10.0
    transition_m: float = 50.0
    border_gap_m: float = 15.0

    def __post_init__(self):
        if self.num_drones < 1 or self.budget_m <= 0:
            raise ValueError('Positive drone count and budget required')
        if not 0 < self.fov_deg < 180 or self.altitude_m <= 0:
            raise ValueError('Invalid camera configuration')
        if not 0 <= self.overlap < 1 or self.spacing_m <= 0:
            raise ValueError('Invalid overlap or spacing')

    @property
    def detection_radius(self):
        return self.altitude_m*np.tan(np.radians(self.fov_deg/2))


def trim(paths, budget_per_drone):
    return [substring(p,0,budget_per_drone) if p.length > budget_per_drone else p for p in paths]


def split(path, n):
    if n == 1 or path.is_empty or path.length == 0:
        return [path]
    return [substring(path,i*path.length/n,(i+1)*path.length/n) for i in range(n)]


def spiral(env: Environment, cfg: Config):
    cx, cy = env.center
    step = 2*cfg.detection_radius*(1-cfg.overlap)
    a = step/(2*np.pi)
    theta_max = env.radius/step*2*np.pi
    length = 0.5*a*(theta_max*np.sqrt(1+theta_max**2)+np.log(theta_max+np.sqrt(1+theta_max**2)))
    theta = np.linspace(0,theta_max,max(2,int(length/cfg.spacing_m)))
    radius = np.clip(a*theta,0,env.radius)
    path = LineString(np.column_stack((cx+radius*np.cos(theta), cy+radius*np.sin(theta))))
    return trim(split(path,cfg.num_drones),cfg.budget_m/cfg.num_drones)


def concentric(env: Environment, cfg: Config):
    cx,cy = env.center
    step = 2*cfg.detection_radius*(1-cfg.overlap)
    radius = step
    points = []
    while radius <= env.radius:
        angle = cfg.transition_m/radius
        arc = radius*(2*np.pi-angle)
        theta = np.linspace(0,2*np.pi-angle,max(2,int(arc/cfg.spacing_m)))
        points.extend(zip(cx+radius*np.cos(theta),cy+radius*np.sin(theta)))
        if radius+step <= env.radius:
            points.append((cx+radius+step,cy))
        else:
            theta = np.linspace(2*np.pi-angle,2*np.pi,max(2,int(radius*angle/cfg.spacing_m)))
            points.extend(zip(cx+radius*np.cos(theta),cy+radius*np.sin(theta)))
        radius += step
    path = LineString(points) if len(points)>1 else LineString()
    return trim(split(path,cfg.num_drones),cfg.budget_m/cfg.num_drones)


def pizza(env: Environment, cfg: Config):
    cx,cy = env.center
    step = 2*cfg.detection_radius*(1-cfg.overlap)
    paths = []
    angle = 2*np.pi/cfg.num_drones
    for k in range(cfg.num_drones):
        points = [(cx,cy)]
        radius, direction = step, 1
        while radius <= env.radius:
            lo = k*angle+cfg.border_gap_m/radius
            hi = (k+1)*angle-cfg.border_gap_m/radius
            if lo >= hi:
                radius += step
                continue
            count = max(2,int(radius*(hi-lo)/cfg.spacing_m))
            theta = np.linspace(lo,hi,count) if direction == 1 else np.linspace(hi,lo,count)
            points.extend(zip(cx+radius*np.cos(theta),cy+radius*np.sin(theta)))
            radius += step
            direction *= -1
        paths.append(LineString(points) if len(points)>1 else LineString())
    return trim(paths,cfg.budget_m/cfg.num_drones)


def greedy(env: Environment, cfg: Config, seed=0, random_walk=False):
    p = env.heatmap
    h,w = p.shape
    dx,dy = env.cell_size
    minx,miny,_,_ = env.bounds
    ox,oy = minx+dx/2, miny+dy/2
    cx,cy = env.center
    start = (int(np.clip((cy-miny)/dy,0,h-1)), int(np.clip((cx-minx)/dx,0,w-1)))
    positions = [start]
    for k in range(1,cfg.num_drones):
        angle = 2*np.pi*k/cfg.num_drones
        positions.append((int(np.clip(start[0]+int(min(2,h//10)*np.sin(angle)),0,h-1)),
                          int(np.clip(start[1]+int(min(2,w//10)*np.cos(angle)),0,w-1))))
    rx,ry = int(np.ceil(cfg.detection_radius/dx)),int(np.ceil(cfg.detection_radius/dy))
    offsets = [(r,c) for r in range(-ry,ry+1) for c in range(-rx,rx+1)
               if (r*dy)**2+(c*dx)**2 <= cfg.detection_radius**2]
    cache = {}
    def visible(pos):
        if pos not in cache:
            row,col = pos
            # Preserve upstream tuple-set summation order when scoring.
            cache[pos] = {(row+dr,col+dc) for dr,dc in offsets
                          if 0<=row+dr<h and 0<=col+dc<w}
        return cache[pos]
    observed = set()
    routes = [[pos] for pos in positions]
    for pos in positions:
        observed.update(visible(pos))
    rng = np.random.default_rng(seed)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    iterations = int((cfg.budget_m//dx)//cfg.num_drones)
    for _ in range(iterations):
        for k,(row,col) in enumerate(positions):
            choices = []
            for dr,dc in neighbors:
                r,c = row+dr,col+dc
                if not (0<=r<h and 0<=c<w):
                    continue
                if (ox+c*dx-cx)**2+(oy+r*dy-cy)**2 >= env.radius**2:
                    continue
                gain = 0.0 if random_walk else sum(p[rr,cc] for rr,cc in visible((r,c))-observed)
                choices.append(((r,c),gain))
            if not choices:
                continue
            best,score = max(choices,key=lambda pair: pair[1])
            if score<=0:
                best = choices[int(rng.integers(len(choices)))][0]
            positions[k] = best
            routes[k].append(best)
            observed.update(visible(best))
    paths = [LineString([(ox+c*dx,oy+r*dy) for r,c in route]) if len(route)>1 else LineString()
             for route in routes]
    return trim(paths,cfg.budget_m/cfg.num_drones)

NAMES = ('Spiral','Concentric','Pizza','GreedySeeded','RandomWalkSeeded')

def plan(name, env, cfg, seed=0):
    funcs = {'Spiral':spiral,'Concentric':concentric,'Pizza':pizza}
    if name in funcs:
        return funcs[name](env,cfg)
    if name in ('GreedySeeded','RandomWalkSeeded'):
        return greedy(env,cfg,seed,random_walk=name=='RandomWalkSeeded')
    raise ValueError(f'Unknown planner {name}; choose from {NAMES}')
