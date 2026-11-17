# Cổng Monipar (mục 5 kế hoạch) — kết luận 15/11/2026 15:40: **QUA**

Nguồn: Zenodo 8104853 v1.0 (CC BY 4.0), 9 file, md5 khớp. Đọc trực tiếp các file .mat.

| Hạng mục | Monipar | PADS | Hệ quả cho kiểm định ngoài |
|---|---|---|---|
| Người | 28: PD 21 (6 supervised + 15 remote), HC 7; 174 lượt (1 lượt/tuần, nhiều tuần) | 469 (PD 276, HC 79, DD 114) | Chỉ làm được PD vs HC; gộp nhiều tuần lên mức người |
| Cảm biến | Chỉ gia tốc 3 trục | Gia tốc + con quay | Mô hình kiểm định ngoài phải là biến thể **chỉ gia tốc** |
| Đơn vị / trọng lực | m/s², **có trọng lực** (|a| trung vị 9.8) | g, **không trọng lực** | Chia 9.81 + lọc thông dải 0.5–20 Hz (loại trọng lực) trước khi đưa vào mô hình |
| Tần số | ~50 Hz, bước 19–20 ms không đều, mốc thời gian ms | 100 Hz không đều | Nội suy theo mốc thời gian lên 50 Hz |
| Cổ tay | Một tay (PD: tay triệu chứng nặng; HC: tay thuận) | Hai tay | Dùng cả hai tay khi huấn luyện; ghi rõ khác biệt |
| Bài tập (nhãn theo từng mẫu) | 1 nghỉ (~4 min/lượt), 2 run tư thế (~30 s), 3 tay lên ngực, 4 gõ ngón, 5 mở-nắm tay, 6 sấp-ngửa, 7 đứng dậy, 8 đi | Relaxed, RelaxedTask, StretchHold, HoldWeight, LiftHold, PointFinger, TouchIndex, TouchNose, DrinkGlas, CrossArms, Entrainment | Ánh xạ chính: Monipar 1 ↔ Relaxed; 2 ↔ StretchHold (± HoldWeight); 3 ↔ TouchNose (gần đúng). 4–8 không có tương đương trực tiếp |
| Nhãn bổ sung | Nhãn run 0–2 theo mẫu (một số lượt), MDS-UPDRS 3.4/3.5/… cho nhóm supervised | — | Có thể thêm phân tích phụ (run), không bắt buộc |

Việc phát sinh:
- P4/P6: thêm biến thể đầu vào **chỉ gia tốc** (acc xyz + |a|, và bất biến |a|, |Δa|) để chuyển giao sang Monipar.
- P6: viết `src/monipar_io.py` (đọc .mat, nội suy theo ms, /9.81, cắt theo nhãn bài tập).
- Hạn chế phải nêu: 28 người, 1 tay, chỉ gia tốc, khác bối cảnh (remote vs supervised).
