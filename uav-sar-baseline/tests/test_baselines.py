"""Checks against upstream implementation and analytical probability invariants."""
import importlib.util
import json
import logging
from pathlib import Path
import sys
import types

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from sar_baseline.data import Environment, load_environment
from sar_baseline.download import git_blob_sha
from sar_baseline.metrics import evaluate
from sar_baseline.planners import Config, plan

ROOT=Path(__file__).resolve().parents[1]

def reference_module(relative):
    name='reference_'+Path(relative).stem
    spec=importlib.util.spec_from_file_location(name,ROOT/'reference'/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def tiny():
    p=np.arange(1,401,dtype=float).reshape(20,20)
    p/=p.sum()
    return Environment(0,p,(-300,-300,300,300),(0,0),280,30,'flat','temperate','test')

@pytest.mark.parametrize('name,func',[('Spiral','generate_spiral_path'),('Concentric','generate_concentric_circles_path'),('Pizza','generate_pizza_zigzag_path')])
@pytest.mark.parametrize('drones',[1,5])
def test_geometric_paths_match_reference(tiny,name,func,drones):
    native=reference_module('sarenv/analytics/paths.py')
    c=Config(num_drones=drones,budget_m=2500)
    actual=plan(name,tiny,c)
    expected=getattr(native,func)(center_x=0,center_y=0,max_radius=280,
        fov_deg=c.fov_deg,altitude=c.altitude_m,overlap=c.overlap,
        num_drones=c.num_drones,path_point_spacing_m=c.spacing_m,
        transition_distance_m=c.transition_m,border_gap_m=c.border_gap_m,budget=c.budget_m)
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a.coords,b.coords,atol=1e-8,rtol=1e-12)


def test_metrics_match_reference(tiny):
    native=reference_module('sarenv/analytics/metrics.py')
    cfg=Config(num_drones=2,budget_m=1500)
    paths=[LineString([(-200,0),(200,0),(200,200)]),LineString([(0,-200),(0,200)])]
    actual,_,_=evaluate(paths,tiny,cfg)
    evaluator=native.PathEvaluator(tiny.heatmap,tiny.bounds,gpd.GeoDataFrame(geometry=[]),45,80,30)
    expected=evaluator.calculate_all_metrics(paths,0.999)
    for new,old in [('likelihood_mass','total_likelihood_score'),('upstream_discounted_score','total_time_discounted_score'),('total_path_km','total_path_length'),('area_covered_km2','area_covered')]:
        assert actual[new]==pytest.approx(expected[old],rel=1e-12,abs=1e-12)


def test_overlap_not_double_counted_and_simultaneous_order_invariant(tiny):
    c=Config(num_drones=2,budget_m=1500)
    p=LineString([(-200,0),(200,0)])
    r1,_,_=evaluate([p,LineString()],tiny,c)
    r2,_,_=evaluate([p,p],tiny,c)
    assert r1['likelihood_mass']==pytest.approx(r2['likelihood_mass'])
    q=LineString([(0,-100),(0,250)])
    x,_,_=evaluate([p,q],tiny,c)
    y,_,_=evaluate([q,p],tiny,c)
    assert x['conditional_detection_auc']==pytest.approx(y['conditional_detection_auc'])
    assert x['simultaneous_first_observation_discounted_mass']==pytest.approx(y['simultaneous_first_observation_discounted_mass'])


def test_probability_curve_and_capped_time_identity(tiny):
    c=Config(num_drones=2,budget_m=2500)
    values,t,curve=evaluate(plan('Pizza',tiny,c),tiny,c)
    assert np.all(np.diff(curve)>=-1e-14)
    assert 0<=curve[-1]<=1
    assert curve[-1]==pytest.approx(values['conditional_detection_probability'])
    assert values['capped_detection_time_s']==pytest.approx(t[-1]*(1-values['conditional_detection_auc']))


@pytest.mark.parametrize('name',['Spiral','Concentric','Pizza','GreedySeeded','RandomWalkSeeded'])
def test_budget_and_reproducible_planner(tiny,name):
    cfg=Config(num_drones=3,budget_m=2400)
    a=plan(name,tiny,cfg,seed=19)
    b=plan(name,tiny,cfg,seed=19)
    assert len(a)==3
    assert all(p.length<=800+1e-8 for p in a)
    assert [p.wkb for p in a]==[p.wkb for p in b]


def test_zero_observation_penalized(tiny):
    c=Config(num_drones=2,budget_m=2000)
    scores,t,curve=evaluate([LineString(),LineString()],tiny,c)
    assert scores['conditional_detection_probability']==0
    assert scores['conditional_detection_auc']==0
    assert scores['capped_detection_time_s']==100
    assert np.all(curve==0)


def test_real_loader_matches_upstream_crop():
    sys.path.insert(0,str(ROOT/'reference'))
    logger=types.ModuleType('sarenv.utils.logging_setup')
    logger.get_logger=lambda: logging.getLogger('reference')
    sys.modules['sarenv.utils.logging_setup']=logger
    native=reference_module('sarenv/core/loading.py')
    folder=ROOT/'data/sarenv/1'
    if not folder.exists():
        pytest.skip('Download map 1 to check upstream loader parity')
    meta=json.loads((folder/'metadata.json').read_text())
    loader=object.__new__(native.DatasetLoader)
    loader._master_probability_map=np.load(folder/'heatmap.npy')
    loader._center_point=meta['center_point']
    loader._meter_per_bin=meta['meter_per_bin']
    loader._bounds=meta['bounds']
    loader._climate=meta['climate']
    loader._environment_type=meta['environment_type']
    loader._projected_crs='EPSG:32630'
    loader._master_features_gdf_proj=gpd.GeoDataFrame(geometry=[],crs=loader._projected_crs)
    expected=loader.load_environment('medium')
    actual=load_environment(ROOT/'data/sarenv',1)
    np.testing.assert_array_equal(actual.heatmap,expected.heatmap)
    np.testing.assert_allclose(actual.bounds,expected.bounds,atol=1e-8,rtol=0)
    assert actual.mass<1  # the crop must not be silently normalized


def test_reference_files_are_byte_identical_upstream():
    tree=json.loads((ROOT/'data/sarenv/upstream-tree.json').read_text())
    expected={entry['path']:entry['sha'] for entry in tree['tree'] if entry['type']=='blob'}
    for f in (ROOT/'reference').rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts:
            relative=str(f.relative_to(ROOT/'reference'))
            if relative in expected:
                assert git_blob_sha(f.read_bytes())==expected[relative], relative


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Config(budget_m=0)
    with pytest.raises(ValueError):
        Config(num_drones=0)
