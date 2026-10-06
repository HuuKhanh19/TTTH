# Chạy tham số Table 9 trong mô hình 1D của RSD

Ngày chạy: 06/10/2026. Lệnh chạy từ thư mục `RSD`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 run_table9_sweep.py --seed 42
```

Paper không chọn một giá trị $\theta$ “tốt nhất” duy nhất: [Table 9](../../../Paper/s41598-026-55955-2.md) thử $\theta\in\{0,0.02,0.05,0.1\}$. Table 8 cho thấy $\theta=0.1$ có biên ổn định lớn nhất trong bốn mức, đồng thời gần giới hạn khả dụng nhất. Do đó thí nghiệm này chạy đủ bốn mức thay vì gán một mức là tối ưu cho mọi tiêu chí.

## Tham số thực sự được áp dụng

Mã giữ 10 UAV và dùng $T=100$, $W=0.01I$, $Q=I$, $R=0.1I$, bán kính liên lạc 12, gain consensus 0.15, năng lượng tối thiểu bằng 10% năng lượng ban đầu. Bốn mức $\theta$ được chạy riêng với seed 42. Nhánh này sử dụng công thức hiệu chỉnh Riccati $\bar P=P(I-2\theta WP)^{-1}$ từ phương trình (21) và kiểm tra điều kiện khả dụng trước khi tính gain.

Các giả định kế thừa repo do paper không cho giá trị cụ thể: $E_0=100$ mỗi UAV; $\alpha=0.1$, $\beta=0.01$ cho tiêu hao năng lượng; ma trận $A=B=I$; các trọng số stage cost đều bằng 1. `main.py` vẫn dùng 50 ô **1 chiều**, không phải lưới $50\times50$ của paper. Mã không mô phỏng vật cản 15%, Gaussian ignition, bước thời gian $\Delta t=0.1$ hay động lực học double-integrator 2D. Các phép đo và consensus hiện chưa tác động vào điều khiển Proposed. Bởi vậy đây là **thí nghiệm độ nhạy tham số Table 9 trên mô hình 1D của repo**, không phải tái lập Table 12.

## Kết quả Proposed

| $\theta$ | Coverage (%) | Mission time | Energy theo mã | Path length theo mã | Connectivity | $\min\lambda(I-2\theta WP)$ | $\max\rho(A-BK)$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 100* | 10.7229 | 106.0476 | 1.00 | 1.000000 | 0.090909 |
| 0.02 | 2 | 100* | 10.7230 | 106.0491 | 1.00 | 0.999563 | 0.090876 |
| 0.05 | 2 | 100* | 10.7231 | 106.0514 | 1.00 | 0.998908 | 0.090826 |
| 0.10 | 2 | 100* | 10.7232 | 106.0552 | 1.00 | 0.997817 | 0.090744 |

\* Giá trị `100` nghĩa là chưa đạt ngưỡng coverage 90% sau 100 bước. Tất cả các mức $\theta$ đều thỏa điều kiện khả dụng tính từ ma trận của repo và có bán kính phổ vòng kín dưới 1. Sự khác biệt giữa bốn mức rất nhỏ và không cải thiện coverage. Không có cơ sở chọn một mức là tốt nhất về hiệu quả nhiệm vụ trong mô hình này.

Ở $\theta=0.1$, baseline Alawad đạt coverage 100% tại bước 42, nhưng tiêu thụ năng lượng 108.90 và connectivity 0.19. Proposed chỉ đạt coverage 2%, dù energy 10.72 và connectivity 1.00. Kết quả này cho thấy metric năng lượng thấp không đi kèm với nhiệm vụ khám phá hoàn thành.

Dữ liệu: [kết quả Proposed theo $\theta$](proposed_theta_sweep.csv), [cả bốn phương pháp](all_methods.csv). Lượt chạy $\theta=0.1$: [cấu hình và kiểm tra](theta_0_1/data/tables/run_config.json), [log](theta_0_1/run.log), [biểu đồ coverage](theta_0_1/coverage_plot.png). Mỗi thư mục `theta_*` có hai bảng CSV và sáu PNG.

## Giới hạn đối chiếu với paper

- Paper báo Proposed coverage 99% và mission time 58 bước trong Table 12; mã hiện tại chỉ đếm ô mà tọa độ 1D của UAV đi qua, nên không thể so với coverage trên lưới 2D có vùng quét.
- `compute_energy` dùng năng lượng sau bước đầu làm mốc, bỏ qua tiêu hao bước đầu. `compute_path_length` cộng quãng đường của cả 10 UAV, trong khi paper ghi *average path length per UAV*.
- Bốn phương pháp trong mỗi lượt chạy được mô phỏng tuần tự trên các mẫu ngẫu nhiên khác nhau. Seed 42 cho phép chạy lại cùng kết quả, nhưng chưa tạo các kịch bản ghép cặp hoặc thống kê nhiều seed như hình trong paper.
- Table 8 của paper ghi $\lambda_{\min}(I-2\theta WP)=0.45$ tại $\theta=0$, trong khi biểu thức toán học bằng đúng 1 tại $\theta=0$. Các giá trị điều kiện khả dụng ở bảng trên được tính từ ma trận thực tế của repo.
