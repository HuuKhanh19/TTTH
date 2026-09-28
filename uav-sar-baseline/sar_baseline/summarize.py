"""Generate Vietnamese Markdown report, machine-readable summaries and figures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load_environment

LABELS={'Spiral':'Spiral','Concentric':'Concentric','Pizza':'Pizza',
        'GreedySeeded':'Greedy (seeded)','RandomWalkSeeded':'Random Walk (seeded)'}
COLORS={'Spiral':'#3976a8','Concentric':'#74928a','Pizza':'#d49928',
        'GreedySeeded':'#1f8e79','RandomWalkSeeded':'#9b7dba'}
METRICS=['likelihood_mass','conditional_detection_probability','conditional_detection_auc',
         'upstream_discounted_score','capped_detection_time_s','total_path_km',
         'area_covered_km2','planning_s','evaluation_s']

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def grouped_maps(rows,budget,algorithm,metric):
    groups={}
    for row in rows:
        if int(row['budget_m'])==budget and row['algorithm']==algorithm:
            groups.setdefault(int(row['dataset']),[]).append(float(row[metric]))
    return {k:float(np.mean(v)) for k,v in sorted(groups.items())}


def bootstrap(values,repeats,seed):
    a=np.asarray(values)
    rng=np.random.default_rng(seed)
    means=a[rng.integers(len(a),size=(repeats,len(a)))].mean(axis=1)
    return np.percentile(means,[2.5,97.5]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/full'))
    parser.add_argument('--data-dir',type=Path,default=Path('data/sarenv'))
    args=parser.parse_args()
    out=args.results
    manifest=json.loads((out/'manifest.json').read_text())
    cfg=manifest['config']
    rows=list(csv.DictReader((out/'runs.csv').open()))
    if len(rows)!=manifest['run_count']:
        raise ValueError('Incomplete results')
    algorithms=cfg['algorithms']; budgets=cfg['budgets_m']
    summaries=[]
    for budget in budgets:
        for name in algorithms:
            result={'budget_m':budget,'algorithm':name,'n_maps':len(cfg['dataset_ids']),
                    'n_seeds':len(cfg['seeds']) if name.endswith('Seeded') else 1}
            for metric in METRICS:
                a=list(grouped_maps(rows,budget,name,metric).values())
                result[metric+'_mean']=float(np.mean(a))
                result[metric+'_std_across_maps']=float(np.std(a,ddof=1)) if len(a)>1 else 0
                low,high=bootstrap(a,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                result[metric+'_ci_low']=low; result[metric+'_ci_high']=high
            summaries.append(result)
    write_csv(out/'summary.csv',summaries)
    paired=[]
    for budget in budgets:
        base=grouped_maps(rows,budget,'Pizza','conditional_detection_probability')
        for name in algorithms:
            cur=grouped_maps(rows,budget,name,'conditional_detection_probability')
            delta=np.array([cur[i]-base[i] for i in sorted(base)])
            low,high=bootstrap(delta,cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
            paired.append({'budget_m':budget,'algorithm':name,'reference':'Pizza',
                           'mean_difference_pp':100*float(delta.mean()),
                           'ci_low_pp':100*low,'ci_high_pp':100*high,
                           'maps_better':int(np.sum(delta>1e-10)),
                           'maps_tied':int(np.sum(abs(delta)<=1e-10)),
                           'maps_worse':int(np.sum(delta < -1e-10))})
    write_csv(out/'paired_vs_pizza.csv',paired)
    # Compare only deterministic methods and matching publicly archived conditions.
    archived=list(csv.DictReader(Path('reference/results/comparative_metrics_results_n5_budget100000.csv').open()))
    comparisons=[]
    if cfg['num_drones']==5 and cfg['size']=='medium' and 100000 in budgets:
        for row in rows:
            if row['algorithm'] not in ('Spiral','Concentric','Pizza') or int(row['budget_m'])!=100000:
                continue
            ref=next((r for r in archived if int(r['Dataset'])==int(row['dataset']) and r['Algorithm']==row['algorithm']),None)
            if ref is None: continue
            comparisons.append({'dataset':row['dataset'],'algorithm':row['algorithm'],
                'our_likelihood':row['likelihood_mass'],'archived_likelihood':ref['Likelihood Score'],
                'absolute_difference':abs(float(row['likelihood_mass'])-float(ref['Likelihood Score']))})
    if comparisons:
        write_csv(out/'upstream_comparison.csv',comparisons)
    # Basic run-level validation independent of the optimizer's own budget checks.
    assert all(r['feasible']=='True' for r in rows)
    assert all(-1e-12 <= float(r['likelihood_mass']) <= float(r['map_mass'])+1e-12 for r in rows)
    assert all(0<=float(r['conditional_detection_auc'])<=1+1e-12 for r in rows)
    curves=json.loads((out/'curves.json').read_text())
    assert all(np.all(np.diff(c['probability'])>=-1e-12) for c in curves)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure_dir=out/'figures'; figure_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,len(budgets),figsize=(6*len(budgets),4.5),squeeze=False)
    for ax,budget in zip(axes[0],budgets):
        ss=[s for s in summaries if s['budget_m']==budget]
        y=np.array([100*s['conditional_detection_probability_mean'] for s in ss])
        lo=np.array([100*s['conditional_detection_probability_ci_low'] for s in ss])
        hi=np.array([100*s['conditional_detection_probability_ci_high'] for s in ss])
        ax.bar(range(len(ss)),y,color=[COLORS[s['algorithm']] for s in ss],width=.65)
        ax.errorbar(range(len(ss)),y,yerr=[y-lo,hi-y],fmt='none',color='#243444',capsize=4)
        ax.set_xticks(range(len(ss)),[LABELS[s['algorithm']].replace(' (seeded)','') for s in ss],rotation=22,ha='right')
        ax.set_ylim(0,105); ax.set_ylabel('Conditional detection probability (%)')
        ax.set_title(f'Team budget {budget/1000:g} km | {cfg["num_drones"]} UAVs')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'SAREnv: {len(cfg["dataset_ids"])} maps | mean and map-bootstrap 95% intervals')
    fig.tight_layout(); fig.savefig(figure_dir/'comparison.png',dpi=160); plt.close(fig)

    fig,axes=plt.subplots(1,len(budgets),figsize=(6*len(budgets),4.2),squeeze=False)
    for ax,budget in zip(axes[0],budgets):
        for name in algorithms:
            bymap={}
            selected=[c for c in curves if c['budget_m']==budget and c['algorithm']==name]
            for c in selected: bymap.setdefault(c['dataset'],[]).append(c['probability'])
            means=np.array([np.mean(v,axis=0) for v in bymap.values()])
            x=np.array(selected[0]['time_s'])/60
            ax.plot(x,100*means.mean(axis=0),label=LABELS[name],color=COLORS[name],lw=2)
        ax.set_title(f'Team budget {budget/1000:g} km'); ax.set_xlabel('Simultaneous flight time (minutes, assumed 10 m/s)')
        ax.set_ylabel('Conditional detection probability (%)'); ax.set_ylim(0,100); ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('First observation of each cell; repeated coverage counted once')
    fig.tight_layout(); fig.savefig(figure_dir/'detection_curves.png',dpi=160); plt.close(fig)

    example_id=cfg['dataset_ids'][0]; example_budget=budgets[0]
    env=load_environment(args.data_dir,example_id,cfg['size'])
    fig,axes=plt.subplots(1,len(algorithms),figsize=(3.8*len(algorithms),4.2),squeeze=False)
    cx,cy=env.center
    extent=[(env.bounds[0]-cx)/1000,(env.bounds[2]-cx)/1000,(env.bounds[1]-cy)/1000,(env.bounds[3]-cy)/1000]
    for ax,name in zip(axes[0],algorithms):
        ax.imshow(env.heatmap,origin='lower',extent=extent,cmap='YlOrRd',vmax=np.percentile(env.heatmap[env.heatmap>0],99))
        artifact=json.loads((out/'routes'/f'map{example_id}_b{example_budget}_{name}_s0.json').read_text())
        for k,path in enumerate(artifact['paths']):
            a=np.array(path)
            if a.size:
                ax.plot((a[:,0]-cx)/1000,(a[:,1]-cy)/1000,lw=.8,color=plt.cm.tab10(k))
                ax.scatter((a[0,0]-cx)/1000,(a[0,1]-cy)/1000,s=15,marker='s',color=plt.cm.tab10(k))
        ax.set_title(LABELS[name]); ax.set_xlabel('East from center (km)'); ax.set_aspect('equal')
    axes[0,0].set_ylabel('North from center (km)')
    fig.suptitle(f'Example fixed in advance: map {example_id}, budget {example_budget/1000:g} km, seed 0; squares = starts')
    fig.tight_layout(); fig.savefig(figure_dir/'example_routes.png',dpi=160); plt.close(fig)

    def result_table(budget):
        lines=['| Baseline | Q cuối (%) | 95% CI | AUC (%) | Lập kế hoạch (s) |',
               '|---|---:|---:|---:|---:|']
        for s in summaries:
            if s['budget_m']!=budget: continue
            lines.append(f'| {LABELS[s["algorithm"]]} | {100*s["conditional_detection_probability_mean"]:.2f} | '
                         f'[{100*s["conditional_detection_probability_ci_low"]:.2f}; {100*s["conditional_detection_probability_ci_high"]:.2f}] | '
                         f'{100*s["conditional_detection_auc_mean"]:.2f} | {s["planning_s_mean"]:.3f} |')
        return '\n'.join(lines)
    conclusions=[]
    for budget in budgets:
        ss=[s for s in summaries if s['budget_m']==budget]
        best=max(ss,key=lambda s:s['conditional_detection_probability_mean'])
        pair=next(p for p in paired if p['budget_m']==budget and p['algorithm']=='GreedySeeded')
        conclusions.append(f'- Ngân sách đội {budget/1000:g} km: **{LABELS[best["algorithm"]]}** có Q cuối trung bình cao nhất '
                           f'({100*best["conditional_detection_probability_mean"]:.2f}%). Greedy so với Pizza: '
                           f'{pair["mean_difference_pp"]:+.2f} điểm phần trăm, CI [{pair["ci_low_pp"]:+.2f}; {pair["ci_high_pp"]:+.2f}], '
                           f'tốt hơn trên {pair["maps_better"]}/{len(cfg["dataset_ids"])} bản đồ.')
    parity=max((r['absolute_difference'] for r in comparisons),default=float('nan'))
    source_links='''- Paper: [Grøntved và cộng sự (2025), Drones 9(9), 628](https://doi.org/10.3390/drones9090628).
- Dataset/code tác giả: [namurproject/SAREnv](https://github.com/namurproject/SAREnv).
- Bản lưu dataset của trường: [SAREnv v1.0, University of Bristol](https://doi.org/10.5523/bris.2k50yyk57qlrj27r4abckbrsfr). Lần chạy này dùng snapshot GitHub đã ghim, không khẳng định bytes giống bản lưu Bristol.
'''
    text=f'''# Báo cáo baseline multi-UAV SAR trên benchmark SAREnv

**Kết quả chạy thực tế trên CPU; không dùng GPU, không huấn luyện mô hình ảnh.**
Lần chạy bắt đầu: `{manifest['started_utc']}`. Hoàn tất: `{manifest['finished_utc']}`.

## 1. Mục tiêu và phạm vi

Cài đặt và đánh giá các baseline điều phối tìm kiếm cho đội {cfg['num_drones']} UAV trên **{len(cfg['dataset_ids'])} bản đồ đã công bố của SAREnv**.
Chúng tôi sử dụng bản đồ xác suất vị trí người mất tích gốc và quy ước vùng phủ camera của benchmark.
Đây là tái lập baseline trên dữ liệu công bố, chưa phải thuật toán GA/NSGA-II mới hay thực nghiệm UAV ngoài đời.

**Phân biệt p và q:** SAREnv cung cấp prior vị trí không đồng nhất p(x). Trong phép đánh giá này, q=1 khi ô nằm trong vùng nhìn và q=0 ngoài vùng nhìn. Dataset không cung cấp xác suất nhận diện theo hover time.
Vì vậy kết quả này là mốc cơ sở cho đề tài; chưa kiểm chứng mô hình q(x,h) không đồng nhất, hover, che khuất hoặc camera học sâu.

## 2. Nguồn, phiên bản và xử lý dữ liệu

{source_links}
Commit cố định: `{manifest['upstream_revision']}`.
Map IDs: `{','.join(map(str,cfg['dataset_ids']))}`. Kích thước: `{cfg['size']}`.

60 bản đồ của bộ gốc chia thành bốn nhóm: 1–15 đồng bằng/ôn đới, 16–30 miền núi/ôn đới, 31–45 đồng bằng/khô, 46–60 miền núi/khô.
Chúng là bản đồ địa lý kết hợp mô hình xác suất hành vi người mất tích, không phải 60 vụ cứu hộ có vị trí nạn nhân được quan trắc thực tế.
Không chọn bản đồ theo kết quả thuật toán; cấu hình đã được ghi trước khi chạy.

Tải nguyên file `heatmap.npy`, kiểm tra Git blob SHA-1 từ cây repository đã ghim và lưu SHA-256 riêng.
Chỉ trích metadata cuối GeoJSON qua HTTP Range với ETag khớp LFS SHA-256; lưu bytes trích và manifest. Không tải toàn bộ geometry OSM vì các chỉ số ở đây không cần geometry.
Loader cắt hình tròn theo đúng công thức và cách làm tròn của mã gốc, độ phân giải danh nghĩa 30 m/ô. Không giảm độ phân giải.
**Không chuẩn hóa lại crop đầu vào**: khối lượng của vùng medium là một phần của bản đồ master.
Không có training/test split học máy vì đây là đánh giá planner không huấn luyện; các bản đồ dùng để đánh giá với tham số cố định.

## 3. Baseline và thông số

| Baseline | Cách hoạt động | Tính ngẫu nhiên |
|---|---|---|
| Spiral | Xoắn ốc Archimedes, chia đường thành các đoạn bằng độ dài cho đội UAV | Không |
| Concentric | Các vòng tròn đồng tâm nối tiếp, chia đều độ dài | Không |
| Pizza | Mỗi UAV quét một sector hình quạt theo vòng cung zigzag | Không |
| GreedySeeded | Mỗi bước chọn 1 trong 8 ô lân cận có tổng prior chưa quan sát lớn nhất; chia sẻ tập ô đã quan sát giữa UAV | Random khi mọi lựa chọn có gain bằng 0 |
| RandomWalkSeeded | Cùng cơ chế chuyển động và giới hạn của Greedy, chọn ngẫu nhiên láng giềng | Có |

Mã hình học được thích nghi từ SAREnv dưới MIT, giữ attribution tại `reference/LICENSE`.
Greedy giữ quy tắc myopic của upstream, bổ sung RNG có seed và cache footprint. Không xem baseline này là thuật toán mới.
Mỗi kế hoạch trả về tuyến của toàn đội. Chạy tham lam theo vòng lần lượt các UAV như upstream, không giải tối ưu toàn cục.

| Tham số | Giá trị |
|---|---|
| Số UAV | {cfg['num_drones']} |
| Ngân sách tổng chiều dài đội | {', '.join(str(b/1000)+' km' for b in budgets)} |
| Ngân sách mỗi UAV | {', '.join(str(b/1000/cfg['num_drones'])+' km' for b in budgets)} |
| Độ cao / FoV | {cfg['altitude_m']} m / {cfg['fov_deg']}° |
| Bán kính nhìn | khoảng 33,14 m |
| Overlap / waypoint spacing | {cfg['overlap']} / {cfg['spacing_m']} m |
| Mẫu đánh giá dọc đường | không quá 15 m |
| Discount upstream | {cfg['discount']} theo mét |
| Seed stochastic | {cfg['seeds']} |
| Số lần chạy | {manifest['run_count']} |

Giữ nguyên quy ước benchmark: các UAV được đặt tại điểm đầu mỗi đoạn, không tính chi phí triển khai từ cùng depot, không bắt buộc quay về, không có tránh va chạm, pin vật lý, độ dốc bay hay vùng cấm.
Spiral/Concentric có thể bắt đầu ở xa tâm; đây là giới hạn cần giữ trong diễn giải kết quả.
Vận tốc {cfg['speed_m_s']} m/s chỉ là giả định của chỉ số thời gian bổ sung; không ảnh hưởng likelihood gốc.

## 4. Định nghĩa các chỉ số

- **Likelihood mass gốc:** L = tổng p_i trên hợp các ô đã nhìn thấy. Một ô chỉ tính một lần giữa tất cả UAV. So sánh được với cột `Likelihood Score` của repository.
- **Q cuối có điều kiện:** Q = L / M, với M là tổng prior trong crop medium. Diễn giải là xác suất phát hiện nếu người ở trong vùng medium. Không được so trực tiếp Q (%) với raw L hoặc cột `Victims Found (%)` gốc.
- **AUC phát hiện:** tích phân Q(t)/H trong [0,H], tính theo lần đầu từng ô được quan sát; UAV bay đồng thời. AUC cao nghĩa là tìm thấy nhiều khối lượng xác suất sớm hơn.
- **Thời gian phát hiện chặn tại H:** nếu không tìm thấy, tính H; bằng H×(1−AUC). Tránh việc chỉ tính thời gian trên các mục tiêu dễ đã tìm thấy.
- **Upstream discounted score:** giữ đúng mã gốc để đối chiếu: cộng khối lượng nhìn thấy tại mọi mẫu, kể cả lặp lại, discount theo khoảng cách cộng dồn từng UAV. **Không phải xác suất, không phải thời gian thực của đội bay đồng thời.**
- `simultaneous_first_observation_discounted_mass` là chỉ số bổ sung tách biệt, chỉ tính quan sát đầu tiên và dùng khoảng cách riêng mỗi UAV.
- Tổng đường bay, diện tích hợp buffer camera, thời gian planner và evaluator được lưu trong CSV. Diện tích buffer theo upstream có thể gồm phần ngoài vòng tìm kiếm; Q chỉ tính prior trong crop.

Không lấy mẫu nạn nhân ẩn để tính `Victims Found (%)` như mã gốc: metric chính được tính trực tiếp trên bản đồ prior, không có nhiễu Monte Carlo. Chúng tôi không gán nhãn metric thay thế thành kết quả victim sampler của tác giả.

## 5. Kết quả thực nghiệm

Trong mỗi bản đồ, lấy trung bình các seed trước. Sau đó lấy trung bình đều giữa bản đồ, tránh tăng trọng số của phương pháp chạy nhiều seed.
95% CI là percentile bootstrap {cfg['bootstrap_repeats']} lần **theo bản đồ**, seed {cfg['bootstrap_seed']}; thể hiện biến động giữa các kịch bản trong tập benchmark, không chứng minh khả năng tổng quát ngoài đời.

'''
    for budget in budgets:
        text+=f'### Ngân sách tổng đội {budget/1000:g} km\n\n'+result_table(budget)+'\n\n'
    text+='\n'.join(conclusions)+f'''

CI của chênh lệch được bootstrap theo **cặp cùng bản đồ** sau khi trung bình seed. Không kết luận Greedy luôn tốt hơn phương pháp quét đều; các kết luận trên chỉ áp dụng cấu hình đã chạy.

![So sánh baseline](figures/comparison.png)

![Xác suất phát hiện theo thời gian](figures/detection_curves.png)

![Tuyến bay minh họa](figures/example_routes.png)

## 6. Tái lập và đối chiếu mã gốc

Đã đối chiếu {len(comparisons)} cặp kết quả deterministic tại 5 UAV, budget 100 km với file CSV upstream; sai lệch likelihood tuyệt đối lớn nhất **{parity:.3e}**.
Bảng chi tiết: `upstream_comparison.csv`. Đây là đối chiếu các baseline/metric cụ thể, không tuyên bố tái lập toàn bộ paper.
Greedy/Random Walk không kỳ vọng khớp từng số CSV gốc do mã gốc dùng RNG không có seed; bản này công bố 5 seed.

Các kiểm tra tự động gồm: geometry khớp upstream cho 1/5 UAV; evaluator khớp likelihood/discount/area/path length upstream; crop khớp loader gốc; không cộng đôi vùng chồng lấn; tính bất biến khi đổi thứ tự danh sách UAV cho metric đồng thời; đường cong xác suất không giảm; giới hạn đường bay; tái lập seed; trường hợp không quan sát; checksum mã tham chiếu.
Kết quả test lưu ở `../test-results.txt` (chạy lệnh pytest để kiểm tra lại).

## 7. Phần cứng và chi phí tính toán

- CPU: `{manifest['cpu']}`; nền tảng `{manifest['platform']}`.
- Python `{manifest['python']}`, NumPy `{manifest['numpy']}`, Shapely `{manifest['shapely']}`.
- {manifest['workers']} worker processes, không sử dụng GPU.
- Wall-clock cho cả vòng chạy: **{manifest['wall_seconds']:.2f} giây**; không bao gồm tải dữ liệu, cài dependency, sinh báo cáo và test.
- Thời gian mỗi planner là thời gian tường đo trong môi trường có worker song song; không dùng làm benchmark tốc độ phần cứng độc lập.
- Phiên bản dependency chính xác tại `../../requirements-lock.txt`; source hashes, checksum từng map, thời gian và cấu hình tại `manifest.json`.

## 8. Chạy lại

Từ thư mục `uav-sar-baseline`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m sar_baseline.download --ids 1-60
.venv/bin/python -m pytest -q
.venv/bin/python -m sar_baseline.run --config configs/benchmark.json --workers 6 --output results/reproduce
.venv/bin/python -m sar_baseline.summarize --results results/reproduce
```

Đổi `--output` để bảo toàn kết quả cũ. Chạy nhanh: thêm `--ids 1,16,31,46` cho lệnh run.
`runs.csv` chứa từng lần chạy; `summary.csv` chứa trung bình/CI; `paired_vs_pizza.csv` chứa so sánh theo cặp; `curves.json` chứa đường cong; `routes/` chứa mọi tuyến bay.

## 9. Liên hệ đề tài môn học

Baseline này cung cấp dữ liệu, quy trình đánh giá và mốc so sánh trước khi áp dụng GA/NSGA-II trong Chương 2 và 10.
Phần tiếp theo có thể giữ các prior này nhưng thêm q_i(h), hover và chi phí quay về, rồi tối ưu phân công–thứ tự–thời gian quan sát.
Khi đổi các giả định đó phải tạo protocol riêng, chạy lại **tất cả** baseline với cùng ngân sách và không so trực tiếp bảng mới với bảng native này.
Không lấy prior p_i từ SAREnv làm xác suất phát hiện q_i, không coi chưa quan sát là chắc chắn không có người, không diễn giải độ phủ thành số người được cứu thực tế.
'''
    (out/'REPORT_VI.md').write_text(text)
    print(f'Wrote {out}/REPORT_VI.md, summaries and figures; max upstream difference={parity:.3e}')

if __name__=='__main__':
    main()
