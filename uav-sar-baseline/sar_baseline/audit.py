"""Re-run upstream code on cases that differ from its archived CSV.

This diagnostic does not modify the baseline experiment or any numerical result.
It creates a separate audit record and appends an idempotent report appendix.
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import types

import geopandas as gpd
import numpy as np
from .data import load_environment
from .planners import Config,plan
from .metrics import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/full'))
    parser.add_argument('--data-dir',type=Path,default=Path('data/sarenv'))
    args=parser.parse_args()
    comparison_path=args.results/'upstream_comparison.csv'
    if not comparison_path.exists():
        print('No archived comparison for this configuration.');return
    comparison=list(csv.DictReader(comparison_path.open()))
    discrepant=sorted({int(r['dataset']) for r in comparison if float(r['absolute_difference'])>1e-10})
    manifest=json.loads((args.results/'manifest.json').read_text())
    missing=sorted(set(manifest['config']['dataset_ids'])-{int(r['dataset']) for r in comparison})
    sys.path.insert(0,str(Path('reference').resolve()))
    logger=types.ModuleType('sarenv.utils.logging_setup')
    logger.get_logger=lambda:logging.getLogger('upstream-audit')
    sys.modules['sarenv.utils.logging_setup']=logger
    # Use the unchanged upstream loader/planners/evaluator as an independent oracle.
    from sarenv.core.loading import DatasetLoader
    from sarenv.analytics.paths import generate_spiral_path,generate_concentric_circles_path,generate_pizza_zigzag_path
    from sarenv.analytics.metrics import PathEvaluator
    checks=[]
    config=manifest['config']
    cfg=Config(num_drones=config['num_drones'],budget_m=100000,
               fov_deg=config['fov_deg'],altitude_m=config['altitude_m'],overlap=config['overlap'],
               spacing_m=config['spacing_m'],transition_m=config['transition_m'],border_gap_m=config['border_gap_m'])
    for map_id in discrepant:
        folder=args.data_dir/str(map_id)
        meta=json.loads((folder/'metadata.json').read_text())
        loader=object.__new__(DatasetLoader)
        loader._master_probability_map=np.load(folder/'heatmap.npy',allow_pickle=False)
        loader._center_point=meta['center_point'];loader._meter_per_bin=meta['meter_per_bin']
        loader._bounds=meta['bounds'];loader._climate=meta['climate']
        loader._environment_type=meta['environment_type']
        lon,lat=meta['center_point']
        epsg=(32600 if lat>=0 else 32700)+int((lon+180)/6)+1
        loader._projected_crs=f'EPSG:{epsg}'
        # Geometry does not enter crop calculation or likelihood scoring.
        loader._master_features_gdf_proj=gpd.GeoDataFrame(geometry=[],crs=loader._projected_crs)
        original=loader.load_environment(config['size'])
        ours=load_environment(args.data_dir,map_id,config['size'])
        assert np.array_equal(original.heatmap,ours.heatmap)
        np.testing.assert_allclose(original.bounds,ours.bounds,atol=1e-8,rtol=0)
        evaluator=PathEvaluator(original.heatmap,original.bounds,gpd.GeoDataFrame(geometry=[]),cfg.fov_deg,cfg.altitude_m,ours.meters_per_bin)
        for name,func in [('Spiral',generate_spiral_path),('Concentric',generate_concentric_circles_path),('Pizza',generate_pizza_zigzag_path)]:
            native_paths=func(center_x=ours.center[0],center_y=ours.center[1],max_radius=ours.radius,
                fov_deg=cfg.fov_deg,altitude=cfg.altitude_m,overlap=cfg.overlap,
                num_drones=cfg.num_drones,path_point_spacing_m=cfg.spacing_m,
                transition_distance_m=cfg.transition_m,border_gap_m=cfg.border_gap_m,budget=cfg.budget_m)
            native=evaluator.calculate_all_metrics(native_paths,config['discount'])
            scores,_,_=evaluate(plan(name,ours,cfg),ours,cfg,config['discount'],config['speed_m_s'])
            delta=abs(float(native['total_likelihood_score'])-scores['likelihood_mass'])
            assert delta<1e-10
            checks.append({'dataset':map_id,'algorithm':name,'crop_identical':True,
                'native_snapshot_likelihood':float(native['total_likelihood_score']),
                'our_likelihood':scores['likelihood_mass'],'absolute_difference':delta})
    maxdiff=max((r['absolute_difference'] for r in checks),default=0)
    matching=sum(float(r['absolute_difference'])<1e-10 for r in comparison)
    audit={'checks':checks,'archived_comparisons':len(comparison),'archived_matching_at_1e_10':matching,
           'discrepant_maps':discrepant,'maps_missing_from_archived_csv':missing,
           'max_native_snapshot_error':maxdiff,
           'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.results/'native_snapshot_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    heading='\n## 10. Kiểm tra sâu khác biệt với CSV upstream\n'
    appendix=f'''

Có **{matching}/{len(comparison)}** cặp kết quả hình học khớp CSV lưu sẵn với sai số dưới 1e-10.
Các bản đồ có chênh lệch CSV là **{discrepant}**. Những ID thiếu trong CSV lưu sẵn là **{missing}**; chúng vẫn được đánh giá trong thực nghiệm này.

Đã chạy lại trực tiếp loader, ba planner hình học và evaluator **nguyên bản ở cùng commit đã ghim** trên các map khác biệt.
Crop trùng hoàn toàn; sai lệch likelihood lớn nhất giữa mã cài đặt và mã upstream vừa chạy là **{maxdiff:.3e}**.
Do đó không có bằng chứng về sai khác implementation ở những trường hợp đã kiểm tra; kết quả của snapshot hiện tại khác CSV lưu trước.
**Chưa xác định nguyên nhân lịch sử khiến CSV khác snapshot**; không khẳng định lỗi của tác giả, không tự thay số đo để khớp CSV.
File kiểm chứng: `native_snapshot_audit.json`.

Có thể tái lập phụ lục sau lệnh sinh báo cáo:

```bash
.venv/bin/python -m sar_baseline.audit --results results/reproduce
```

Lưu ý diễn giải: tại 100 km, CI chênh lệch Q cuối giữa Greedy và Pizza chứa 0, nên chưa có kết luận rõ về ưu thế Q cuối theo bootstrap này.
Greedy có AUC trung bình cao hơn ở 100 km (tìm được nhiều prior sớm hơn), trong khi Pizza có Q cuối trung bình cao hơn. Đây là hai tiêu chí khác nhau.
'''
    report=args.results/'REPORT_VI.md'
    report.write_text(report.read_text().split(heading)[0]+heading+appendix)
    print(f'Audited {len(checks)} cases; maximum native snapshot difference {maxdiff:.3e}')

if __name__=='__main__':main()
