"""Run paired baseline experiments; deterministic methods are evaluated once.

CPU timing is per-run wall time (not a speed benchmark when workers > 1).
Each stochastic method is averaged over seeds within a map before inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import shapely
from .data import load_environment
from .download import REVISION, parse_ids, sha256
from .metrics import evaluate
from .planners import Config, plan


def run_task(args):
    map_id, budget, spec, data_dir, output = args
    env = load_environment(data_dir,map_id,spec['size'])
    cfg = Config(**{k:spec[k] for k in asdict(Config()) if k!='budget_m'},budget_m=budget)
    rows, curves = [],[]
    for name in spec['algorithms']:
        seeds = spec['seeds'] if name.endswith('Seeded') else [0]
        for seed in seeds:
            t = time.perf_counter()
            paths = plan(name,env,cfg,seed)
            plan_s = time.perf_counter()-t
            t = time.perf_counter()
            values,ts,curve = evaluate(paths,env,cfg,spec['discount'],spec['speed_m_s'])
            eval_s = time.perf_counter()-t
            rows.append(dict(dataset=map_id,terrain=env.terrain,climate=env.climate,
                             size=env.size,map_mass=env.mass,algorithm=name,seed=seed,
                             budget_m=budget,num_drones=cfg.num_drones,
                             planning_s=plan_s,evaluation_s=eval_s,**values))
            curves.append(dict(dataset=map_id,budget_m=budget,algorithm=name,seed=seed,
                               time_s=ts.tolist(),probability=curve.tolist()))
            # Keep every route so the report is inspectable without rerunning planners.
            artifact = Path(output)/'routes'/f'map{map_id}_b{budget}_{name}_s{seed}.json'
            artifact.parent.mkdir(parents=True,exist_ok=True)
            artifact.write_text(json.dumps({'dataset':map_id,'config':asdict(cfg),
                'algorithm':name,'seed':seed,
                'paths':[list(p.coords) for p in paths]},separators=(',',':'))+'\n')
    return rows,curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/benchmark.json'))
    parser.add_argument('--data-dir',type=Path,default=Path('data/sarenv'))
    parser.add_argument('--output',type=Path,default=Path('results/full'))
    parser.add_argument('--ids',help='Optional subset, e.g. 1,16,31,46')
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.ids:
        spec['dataset_ids'] = parse_ids(args.ids)
    if not spec['seeds'] or not spec['dataset_ids']:
        raise ValueError('Empty seeds or dataset IDs')
    # Validate all assets before writing any results.
    for i in spec['dataset_ids']:
        load_environment(args.data_dir,i,spec['size'])
    if (args.output/'manifest.json').exists():
        raise FileExistsError('Output already contains a run; choose a new --output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    (args.output/'config.json').write_text(json.dumps(spec,indent=2)+'\n')
    jobs = [(i,b,spec,str(args.data_dir),str(args.output)) for b in spec['budgets_m'] for i in spec['dataset_ids']]
    curves = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'runs.csv').open('w',newline='') as f:
        writer = None
        for rows,newcurves in pool.map(run_task,jobs):
            if writer is None:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            curves.extend(newcurves)
            completed += 1
            print(f'{completed}/{len(jobs)} map-budget jobs; map={rows[0]["dataset"]} budget={rows[0]["budget_m"]}',flush=True)
    (args.output/'curves.json').write_text(json.dumps(curves,separators=(',',':'))+'\n')
    source_hashes = {str(p):sha256(p.read_bytes()) for p in sorted(Path('sar_baseline').glob('*.py'))}
    manifests = {str(i):json.loads((args.data_dir/str(i)/'provenance.json').read_text()) for i in spec['dataset_ids']}
    try:
        hardware = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        hardware = platform.processor()
    manifest = {'started_utc':stamp,'finished_utc':datetime.now(timezone.utc).isoformat(),
                'wall_seconds':time.perf_counter()-start,'workers':args.workers,
                'python':platform.python_version(),'platform':platform.platform(),
                'cpu':hardware,'numpy':np.__version__,'shapely':shapely.__version__,
                'upstream_revision':REVISION,'config':spec,'source_sha256':source_hashes,
                'dataset_provenance':manifests,'run_count':len(curves)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Finished {len(curves)} runs in {manifest["wall_seconds"]:.2f}s',flush=True)

if __name__ == '__main__':
    main()
