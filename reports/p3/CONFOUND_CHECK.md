# Kiểm tra nhiễu tuổi/giới — P3 sơ bộ (15/11/2026 16:50)

Nguồn: reports/p1/pads_patients.csv; reports/p3/oof_PD_vs_HC_rf_seed{0,1,2}.csv (P3, nested CV theo bệnh nhân).

| Nhóm | n | Tuổi TB ± SD | Nữ |
|---|---|---|---|
| PD | 276 | 65.4 ± 9.6 | 29 % |
| HC | 79 | 62.9 ± 12.5 | 63 % |
| DD | 114 | 62.4 ± 11.5 | 50 % |

- Chỉ dùng tuổi + giới (hồi quy logistic, CV 5-fold): AUROC PD vs HC = 0.674; PD vs DD = 0.608 → giới lệch mạnh giữa PD và HC, là nguồn nhiễu tiềm năng.
- RF (đặc trưng chuyển động), AUROC PD vs HC phân tầng theo giới:
  - seed 0: nữ 0.892 (n=131, HC=50) | nam 0.877 (n=224, HC=29)
  - seed 1: nữ 0.899 | nam 0.891
  - seed 2: nữ 0.914 | nam 0.886
- RF trong nhóm tuổi 55–75 (n=231, HC=41): AUROC 0.894 (seed 0).
- Tương quan xác suất dự đoán với tuổi: HC r = 0.11, PD r = 0.00.

Kết luận: hiệu năng giữ nguyên khi phân tầng theo giới và giới hạn tuổi → mô hình không chỉ học tuổi/giới.
Trong bài: báo cáo bảng nhân khẩu + AUROC tuổi/giới làm baseline + AUROC phân tầng theo giới (phụ lục).
