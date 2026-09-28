# Multi-UAV SAR: baseline trên benchmark SAREnv

Mã Python chạy CPU để đánh giá **Spiral, Concentric, Pizza, GreedySeeded và RandomWalkSeeded** trên 60 bản đồ công bố trong:

Grøntved và cộng sự (2025), *SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue Using UAVs*, Drones 9(9), 628. [DOI](https://doi.org/10.3390/drones9090628) · [Repository tác giả](https://github.com/namurproject/SAREnv).

**Đọc báo cáo:** [REPORT_VI.md](results/full/REPORT_VI.md).

> Phạm vi: bản đồ xác suất vị trí người mất tích không đồng nhất, camera quan sát nhị phân theo footprint. Chưa thêm hover time, xác suất nhận diện camera không đồng nhất, pin vật lý, quay về depot hoặc tránh va chạm. Kết quả là benchmark lập kế hoạch, không phải thử nghiệm cứu hộ ngoài đời. Chưa cài GA/NSGA-II; năm phương pháp ở đây là mốc để so sánh thuật toán tiến hóa sau này.

## Chạy ngay trên máy hiện tại

Môi trường `.venv` và 60 bản đồ đã được chuẩn bị trong thư mục này. Chạy từ `uav-sar-baseline`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m sar_baseline.run --ids 1,16,31,46 --workers 4 --output results/quick
.venv/bin/python -m sar_baseline.summarize --results results/quick
```

Lệnh run giữ cả hai budget trong config. Chọn một tên thư mục output mới mỗi lần để tránh ghi đè kết quả hoàn chỉnh.

## Cài từ đầu và chạy toàn bộ

Python 3.12 được sử dụng trong lần chạy đã báo cáo. Không cần GPU, CUDA, API key hay dịch vụ trả phí.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m sar_baseline.download --ids 1-60 --workers 6
.venv/bin/python -m pytest -q
.venv/bin/python -m sar_baseline.run --config configs/benchmark.json --workers 6 --output results/reproduce
.venv/bin/python -m sar_baseline.summarize --results results/reproduce
.venv/bin/python -m sar_baseline.audit --results results/reproduce
```

`requirements-lock.txt` lưu đúng phiên bản đã chạy. Có thể dùng `pip install -e '.[test]'` nếu cần môi trường phụ thuộc linh hoạt; khi đó kết quả số có thể thay đổi nhẹ theo thư viện hình học. Lưu phiên bản mới cùng kết quả.

Dữ liệu số cần khoảng 0,5 GB; không tải các ảnh minh họa và toàn bộ geometry OSM. Downloader có cache, retry, kiểm tra Git hash của heatmap và LFS ETag của metadata; không tự sinh bản đồ thay thế khi mạng lỗi.

## Cấu trúc

| File/thư mục | Vai trò |
|---|---|
| `sar_baseline/download.py` | Tải dữ liệu phiên bản cố định, checksum và provenance |
| `sar_baseline/data.py` | Cắt bản đồ medium theo đúng loader gốc, giữ nguyên prior mass |
| `sar_baseline/planners.py` | Năm baseline, chia tuyến nhiều UAV, budget, RNG theo seed |
| `sar_baseline/metrics.py` | Chỉ số gốc và chỉ số phát hiện đồng thời bổ sung |
| `sar_baseline/run.py` | Chạy ghép cặp cùng bản đồ, lưu từng kết quả, từng tuyến và manifest |
| `sar_baseline/audit.py` | Chạy lại mã upstream trên các map có chênh lệch CSV, ghi phụ lục kiểm chứng |
| `sar_baseline/summarize.py` | Tổng hợp, bootstrap theo bản đồ, vẽ hình, sinh báo cáo tiếng Việt |
| `configs/benchmark.json` | 60 map, 5 UAV, tổng budget 100/250 km, seed 0–4 |
| `tests/test_baselines.py` | Kiểm tra đối chiếu mã gốc và các bất biến xác suất/ngân sách |
| `reference/` | Trích mã, license và bảng kết quả nguyên bản từ commit upstream |
| `data/sarenv/` | Heatmap gốc, metadata trích và checksum của từng bản đồ |
| `results/full/` | Kết quả thực nghiệm đầy đủ và báo cáo |

## Cách đọc kết quả

- `runs.csv`: mỗi dòng là một (map, budget, algorithm, seed). Phương pháp deterministic chạy một lần, stochastic chạy 5 seed.
- `summary.csv`: trung bình seed trong mỗi map, sau đó trung bình đều giữa map và CI bootstrap.
- `paired_vs_pizza.csv`: chênh lệch theo cặp bản đồ với Pizza, đơn vị **điểm phần trăm**.
- `upstream_comparison.csv`: so sánh likelihood của ba phương pháp deterministic với CSV gốc ở 100 km.
- `curves.json`: xác suất phát hiện có điều kiện theo thời gian bay đồng thời.
- `routes/*.json`: mọi tuyến đã chạy; tọa độ hệ phẳng UTM, đơn vị mét.
- `manifest.json`: thời gian, môi trường, cấu hình, source hashes và provenance tất cả bản đồ.

`likelihood_mass` giữ đơn vị xác suất trên bản đồ master; `conditional_detection_probability` chia cho khối lượng crop. Không nhầm hai giá trị này với nhau hoặc với `% victims found` từ sampler hình học của SAREnv.

`upstream_discounted_score` tái hiện implementation gốc (cộng lại vùng nhìn, nối khoảng cách các UAV). Nó không phải xác suất hoặc thời gian thực. Các chỉ số `conditional_detection_auc`, `capped_detection_time_s` và `simultaneous_first_observation_discounted_mass` đánh giá riêng lần quan sát đầu tiên khi các UAV bay đồng thời.

## Tái lập và giới hạn

- Snapshot upstream: `0f2d344d17c0b9c18cec2176dd9b171beba727d6`.
- Không dùng vị trí nạn nhân ẩn để lập tuyến. Không tune tham số trên kết quả từng map.
- Không có train/test split học máy; cả 60 map là tập đánh giá planner với cấu hình cố định.
- Không so số lần chạy stochastic như 5 bản đồ độc lập: CI lấy mẫu lại **map** sau khi trung bình seed.
- Native budget là tổng đường đội, chia đều cho UAV. Không tính di chuyển từ depot tới điểm bắt đầu và bay về. Không gọi budget đường bay là mô hình năng lượng pin.
- Không kỳ vọng seed mới của Greedy/Random Walk khớp CSV ngẫu nhiên không seed của tác giả. Ba baseline hình học có kiểm tra đối chiếu trực tiếp.
- Runtime trong CSV được đo khi nhiều process cùng chạy; muốn đo latency độc lập hãy dùng `--workers 1`.
- SAREnv q=1 trong footprint. Nếu thêm q(x,h), hover, depot hoặc ràng buộc vật lý, tạo protocol mới và chạy lại mọi baseline với cùng điều kiện.

## Nguồn và giấy phép

Mã baseline hình học và quy tắc Greedy được thích nghi từ SAREnv (MIT). Giữ thông báo bản quyền tác giả trong [LICENSE](LICENSE) và [reference/LICENSE](reference/LICENSE). Các file trong `reference` là nguyên bản để kiểm chứng; chúng không bị chỉnh sửa.
Dữ liệu địa lý nền của SAREnv xuất phát từ OpenStreetMap; ghi nhận [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) và điều kiện ODbL của dữ liệu nền. Không suy giấy phép MIT cho mọi dữ liệu đầu vào bên thứ ba.

Bản lưu dataset: [SAREnv v1.0 tại University of Bristol](https://doi.org/10.5523/bris.2k50yyk57qlrj27r4abckbrsfr). Nghiên cứu này sử dụng bytes ở snapshot GitHub đã chỉ rõ, không đồng nhất các phiên bản chỉ dựa trên tên.

## Gói chia sẻ

`uav-sar-baseline-source.zip` (ở thư mục cha) chứa source, mã tham chiếu, cấu hình, tests, báo cáo, bảng số và hình. Gói này không chứa `.venv`, heatmap 0,5 GB hoặc toàn bộ JSON tuyến bay. Gói ZIP là file chia sẻ được tạo trong workspace và không được lưu trong Git. Heatmap và tuyến bay không được lưu trong Git; trên máy khác dùng downloader và lệnh run để tái tạo.
