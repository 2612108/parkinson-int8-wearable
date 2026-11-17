# B2 — Đo năng lượng bằng đồng hồ USB KEWEISI (16/11/2026)
Bo ESP32-S3 XH-S3E, firmware b2_e2e, mô hình QAT fold0 (lọc nhân quả trên bo). Log: reports/board/B2_energy_run.log.
| Ảnh (giờ điện thoại) | Pha (theo log máy) | V | A | mAh | T (hh:mm khi I ≥ 10 mA) |
|---|---|---|---|---|---|
| 08:53 | IDLE 08:51:43–08:56:43 | 5.11 | 0.00 | 4 | 00:00 |
| 09:01 | INFER 08:56:53–09:01:53 (20 508 lần suy luận, 14.6 ms/lần) | 5.07 | 0.07 | 21 | 00:04 |
| 09:02 | INFER_DUTY từ 09:02:03 | 5.11 | 0.00 | 25 | 00:05 |
| 09:04 | INFER_DUTY | 5.11 | 0.00 | 25 | 00:05 |
Kết luận: INFER ≈ 70 mA tức thời (≈0.36 W → ≈5 mJ/cửa sổ); IDLE và DUTY < 10 mA (dưới độ phân giải). Bộ đếm mAh tăng 21 mAh trong 5 phút INFER (≈250 mA trung bình) — mâu thuẫn với số A tức thời → năng lượng 5–19 mJ/cửa sổ, chỉ mang tính tham khảo; cần đo lại bằng INA219.
