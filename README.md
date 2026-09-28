# TTTH — Multi-UAV Search and Rescue

Project môn Tính toán tiến hóa: nghiên cứu và tối ưu kế hoạch tìm kiếm cứu nạn bằng đội UAV dưới xác suất phát hiện.

Phiên bản hiện tại cài đặt **5 baseline trên 60 bản đồ benchmark SAREnv**: Spiral, Concentric, Pizza, GreedySeeded và RandomWalkSeeded. Đã chạy 1.560 lượt thực nghiệm trên CPU; cấu hình và kết quả chi tiết được lưu để tái lập.

- [Mã nguồn và hướng dẫn cài đặt](uav-sar-baseline/README.md)
- [Báo cáo kết quả tiếng Việt](uav-sar-baseline/results/full/REPORT_VI.md)
- [Cấu hình benchmark](uav-sar-baseline/configs/benchmark.json)
- [Bảng tổng hợp kết quả](uav-sar-baseline/results/full/summary.csv)
- [Cài đặt các baseline](uav-sar-baseline/sar_baseline/planners.py)

## Chạy lại

```bash
cd uav-sar-baseline
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m sar_baseline.download --ids 1-60
.venv/bin/python -m pytest -q
.venv/bin/python -m sar_baseline.run --config configs/benchmark.json --workers 6 --output results/reproduce
.venv/bin/python -m sar_baseline.summarize --results results/reproduce
.venv/bin/python -m sar_baseline.audit --results results/reproduce
```

Heatmap khoảng 0,5 GB được tải bằng downloader; repository lưu metadata, checksum và nguồn dữ liệu. Các tuyến bay chi tiết được sinh lại bằng lệnh `run`. Không cần GPU hay API key.

## Phạm vi

SAREnv cung cấp prior vị trí người mất tích không đồng nhất. Baseline hiện tại giữ mô hình quan sát gốc: phát hiện chắc chắn trong vùng nhìn camera. Hover time, xác suất nhận diện không đồng nhất, GA/NSGA-II và các ràng buộc vật lý là các phần mở rộng tiếp theo.

## Nguồn

Grøntved và cộng sự (2025), *SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue Using UAVs*, Drones 9(9), 628. [Paper](https://doi.org/10.3390/drones9090628) · [Mã nguồn SAREnv](https://github.com/namurproject/SAREnv).

Mã thích nghi từ SAREnv giữ giấy phép và thông báo tác giả tại [LICENSE](uav-sar-baseline/LICENSE).
