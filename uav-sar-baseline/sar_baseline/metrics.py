"""Vectorized SAREnv metrics and a separate simultaneous first-observation score.

The benchmark's raw likelihood sums each observed cell once. Its published
implementation of the discounted score repeats visible mass at each sample and
concatenates drones' distances. We reproduce this explicitly for comparison;
we do NOT interpret it as a probability or physical simultaneous time.
"""
import numpy as np
import shapely
from shapely.ops import unary_union
from .data import Environment
from .planners import Config


def footprint_pairs(xy, env: Environment, radius):
    """Return (sample index, flat cell index) for each visible cell center.

Matches PathEvaluator.get_visible_cells, including integer truncation at borders.
"""
    h,w = env.heatmap.shape
    dx,dy = env.cell_size
    minx,miny,_,_ = env.bounds
    cols = ((xy[:,0]-minx)/dx).astype(int)
    rows = ((xy[:,1]-miny)/dy).astype(int)
    rx,ry = int(np.ceil(radius/dx)),int(np.ceil(radius/dy))
    samples,cells = [],[]
    for dr in range(-ry,ry+1):
        for dc in range(-rx,rx+1):
            rr,cc = rows+dr,cols+dc
            mask = ((rr>=0)&(rr<h)&(cc>=0)&(cc<w)&
                    ((minx+(cc+0.5)*dx-xy[:,0])**2+
                     (miny+(rr+0.5)*dy-xy[:,1])**2 <= radius**2))
            idx = np.flatnonzero(mask)
            samples.append(idx)
            cells.append(rr[idx]*w+cc[idx])
    return np.concatenate(samples),np.concatenate(cells)


def evaluate(paths, env: Environment, cfg: Config, discount=0.999, speed_m_s=10.0, area=True):
    """Exact deterministic expected mass on the benchmark's sampled footprint.

Conditional scores assume the target lies in the selected crop. Raw likelihood
retains the master-map mass, as upstream does. No hidden target is provided to
any planner. No Monte Carlo target sampler is needed for these metrics.
"""
    if not 0 < discount <= 1 or speed_m_s <= 0:
        raise ValueError('Invalid discount or speed')
    flat = env.heatmap.ravel()
    first_distance = np.full(flat.size,np.inf)
    discounted = 0.0
    offset = 0.0
    valid = [p for p in paths if not p.is_empty and p.length>0]
    for path in valid:
        resolution = int(np.ceil(env.meters_per_bin/2))
        distances = np.linspace(0,path.length,int(np.ceil(path.length/resolution))+1)
        xy = shapely.get_coordinates(shapely.line_interpolate_point(path,distances))
        samples,cells = footprint_pairs(xy,env,cfg.detection_radius)
        np.minimum.at(first_distance,cells,distances[samples])
        discounted += float(np.sum(flat[cells]*discount**(distances[samples]+offset)))
        offset += path.length
    covered = np.isfinite(first_distance)
    raw_mass = float(flat[covered].sum())
    # Equal cruise speed, simultaneous starts; never add path lengths between UAVs.
    horizon_m = cfg.budget_m/cfg.num_drones
    capped = np.minimum(first_distance,horizon_m)
    mean_capped_s = float(np.dot(flat,capped)/env.mass/speed_m_s)
    auc = float(np.dot(flat,np.maximum(0,1-capped/horizon_m))/env.mass)
    simultaneous_discount = float(np.sum(flat[covered]*discount**first_distance[covered]))
    length = sum(p.length for p in valid)
    if len(paths) != cfg.num_drones:
        raise ValueError('Planner did not return one path per drone')
    feasible = all(p.length <= horizon_m+1e-6 for p in paths)
    if not feasible:
        raise ValueError('Path exceeds per-drone benchmark budget')
    result = {
        'likelihood_mass':raw_mass,
        'conditional_detection_probability':raw_mass/env.mass,
        'upstream_discounted_score':discounted,
        'simultaneous_first_observation_discounted_mass':simultaneous_discount,
        'conditional_detection_auc':auc,
        'capped_detection_time_s':mean_capped_s,
        'total_path_km':length/1000,
        'max_path_km':max((p.length for p in paths),default=0)/1000,
        'area_covered_km2':float(unary_union([p.buffer(cfg.detection_radius) for p in valid]).area/1e6) if area and valid else 0.0,
        'feasible':feasible,
    }
    checkpoints = np.linspace(0,horizon_m,101)
    curve = np.array([flat[first_distance<=x+1e-8].sum()/env.mass for x in checkpoints])
    return result, checkpoints/speed_m_s, curve
