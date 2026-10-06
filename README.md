# TTTH — Tính toán tiến hóa cho tìm kiếm cứu nạn bằng UAV

Đồ án môn Tính toán tiến hóa, nghiên cứu phương pháp phối hợp nhiều UAV để tìm kiếm cứu nạn. Kho lưu trữ hiện gồm bản chép Markdown của paper, mã mô phỏng RSD và tài liệu slide.

## Nội dung

- [Paper](Paper/s41598-026-55955-2.md): nội dung paper ở dạng Markdown, công thức LaTeX và 14 hình gốc trong [figures](Paper/figures/).
- [RSD](RSD/): mã mô phỏng phương pháp Proposed và ba baseline Adam, Javed, Alawad. Mã được lấy từ [repository RSD gốc](https://github.com/MohdAsimSayeed/RSD) và có các sửa đổi để chạy, xuất kết quả và thử tham số Table 9.
- [Slide](Slide/1.%20IntroEC.md): ghi chú bài giảng về tính toán tiến hóa và các thuật toán liên quan.

## Chạy RSD

Cần Python 3 cùng `numpy`, `pandas` và `matplotlib`:

```bash
python3 -m pip install numpy pandas matplotlib
cd RSD
python3 run_table9_sweep.py --seed 42
```

Lệnh trên chạy bốn giá trị `theta` trong Table 9 (`0`, `0.02`, `0.05`, `0.1`) với profile `paper-table9-1d`. Kết quả được ghi vào `RSD/results/paper_table9_1d_seed42/`.

Để chạy cấu hình mặc định và giữ kết quả trong một thư mục riêng:

```bash
cd RSD
mkdir -p results/local_default
cd results/local_default
MPLBACKEND=Agg python3 -u ../../main.py --seed 42
```

Các kết quả đã chạy và nhận xét nằm trong [báo cáo cấu hình mặc định](RSD/results/reproduction_2026-10-06/REPORT_VI.md) và [báo cáo quét Table 9](RSD/results/paper_table9_1d_seed42/REPORT_VI.md).

**Phạm vi mô phỏng:** RSD hiện dùng trạng thái UAV một chiều trên 50 ô. Profile Table 9 áp dụng một số tham số của paper nhưng chưa mô phỏng lưới hai chiều, vật cản và động lực học đầy đủ. Các kết quả này không phải phép tái lập Table 12 của paper.
