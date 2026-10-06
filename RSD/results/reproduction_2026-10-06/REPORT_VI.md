# Báo cáo chạy lại RSD — 06/10/2026

## Lần chạy

- Repo: commit `52cbd20`, với các sửa lỗi chạy được ghi ở cuối báo cáo.
- Môi trường: macOS arm64; Python 3.14.6; NumPy 2.5.2; pandas 3.0.5; Matplotlib 3.11.1.
- Cấu hình trong `main.py`: seed 42, 10 UAV, 50 bước, trạng thái UAV 1 chiều, trường thảm họa 50 ô, `theta = 0.5`.
- Lệnh chạy từ thư mục báo cáo này: `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -u ../../main.py > run.log 2>&1`.
- Kết thúc: exit code 0, cả bốn phương pháp chạy xong; sáu PNG và hai CSV đã được tạo. Các cảnh báo `FigureCanvasAgg is non-interactive` chỉ liên quan đến `plt.show()` khi chạy không có giao diện.

## Kết quả của lần chạy mới

| Phương pháp | Coverage (%) | Mission time (bước) | Energy theo mã | Path length theo mã | Connectivity |
|---|---:|---:|---:|---:|---:|
| Proposed | 2 | 50* | 7.14 | 149.10 | 1.00 |
| Adam | 8 | 50* | 8.45 | 164.13 | 0.42 |
| Javed | 16 | 50* | 17.15 | 246.83 | 0.66 |
| Alawad | 100 | 43 | 53.90 | 492.59 | 0.00 |

\* Giá trị 50 là giới hạn mô phỏng: phương pháp chưa đạt ngưỡng coverage 90%, không phải hoàn thành nhiệm vụ ở bước 50.

Theo mã hiện tại, Proposed có mức tiêu thụ năng lượng và tổng quãng đường thấp nhất, đồng thời giữ kết nối ở cả 50 bước; coverage chỉ 2%, thấp hơn nhiều so với Alawad (100%). Vì vậy không thể diễn giải năng lượng và đường đi thấp như một chiến thắng toàn diện: UAV của Proposed hầu như không khám phá lưới.

Dữ liệu chi tiết: [method_metrics.csv](data/tables/method_metrics.csv), [overall_summary.csv](data/tables/overall_summary.csv), [run.log](run.log). Biểu đồ: [coverage](coverage_plot.png), [energy](energy_plot.png), [path length](path_length_plot.png), [connectivity](connectivity_plot.png), [radar](radar_plot.png), [improvement](improvement_plot.png).

## Giới hạn khi đối chiếu paper

Đây là lần chạy `main.py`, **không tái lập Table 12 của paper**. Table 9 của paper dùng lưới 50 × 50 và 100 bước, còn repo chạy mô hình vị trí 1 chiều trên 50 ô và 50 bước. Các tham số khác cũng khác: bán kính liên lạc 2 so với 12; `theta = 0.5` so với tập giá trị 0–0.1 trong paper. `main.py` không chạy thí nghiệm robustness. Paper báo Proposed đạt coverage 99%, mission time 58, energy 1250, connectivity 0.95 và path length 420; những số này không thể so trực tiếp với bảng ở trên.

Các tệp có sẵn trong `uav_full_pipeline/` không bị thay đổi và không được `main.py` đọc. Bảng `uav_full_pipeline/processed/summary.csv` có số khác với cả lần chạy mới và Table 12. Một số cột của bảng cũ không khớp khi tính lại từ raw CSV bằng hàm metric hiện tại, nên chưa xác nhận được nguồn tạo toàn bộ bảng đó.

Hai lưu ý về metric hiện hành: `compute_energy` lấy năng lượng sau bước đầu làm mốc, nên bỏ qua tiêu hao của bước đầu; các phương pháp được chạy tuần tự trên các mẫu ngẫu nhiên khác nhau, dù toàn bộ lần chạy đã được cố định seed 42. Đây là kết quả của một lần chạy, chưa phải thống kê qua nhiều seed.

## Sửa lỗi để chạy được

- `simulation/simulator.py`: chuyển đoạn pseudocode ở đầu tệp thành docstring hợp lệ; khởi tạo gain bằng 0 cho nhánh baseline khi cập nhật hiệp phương sai.
- `main.py`: tạo ma trận trường thảm họa 50 × 50 riêng với ma trận trạng thái UAV 1 × 1; đặt seed 42 từ cấu hình có sẵn; ghi thêm CSV của cả bốn phương pháp.
- `evaluation/summary_table.py`: tạo thư mục cha trước khi lưu bảng.

`README.md` trong repo thực chất là tệp Word mang đuôi `.md`; hướng dẫn bên trong chỉ yêu cầu NumPy, pandas, Matplotlib và `python main.py`.
