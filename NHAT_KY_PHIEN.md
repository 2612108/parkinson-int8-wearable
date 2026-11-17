# Nhật ký phiên — bài JBHI Parkinson

Đầu mỗi phiên: Claude đọc file này + `KE_HOACH_NGHIEN_CUU_Q1_v2.md`. Cuối phiên: cập nhật mục "Trạng thái" và "Đầu vào cho phiên sau".

**Quy trình (chốt 15/11):** máy anh không có GPU → mọi tính toán chạy trên Kaggle.
Claude viết `kaggle/pN/` (notebook + `kernel-metadata.json`) và `ANTIGRAVITY_PN.md`.
Claude Code chạy trong terminal ở thư mục này (đọc `CLAUDE.md`) tự đẩy notebook lên, theo dõi, chỉ kéo `reports/` về `reports/pN/`. `ANTIGRAVITY_PN.md` là danh sách bước, Claude Code tự làm theo.
Dữ liệu thô nằm trong output của kernel `parkinson-p1-audit`; các phiên sau gắn qua `kernel_sources`.
Không ghi token Kaggle vào repo.

## Trạng thái
| Phiên | Nội dung | Trạng thái |
|---|---|---|
| P1 | Tải PADS + Monipar, audit | Kaggle notebook chạy >2 h không ra → 13:37 chuyển sang tải + audit trên máy (Antigravity), sau đó tạo Kaggle Dataset riêng tư `nguynthqunhlu/parkinson-pads-monipar-raw`. Định dạng PADS đã xác nhận: 7 cột (Time, acc xyz [g], gyro xyz [rad/s]), 100 Hz, 1024/2048 mẫu/bài, 2 cổ tay. DOI PADS 10.13026/m0w9-zx22 |
| P2 | Khóa gap (Scopus/Xplore) + đọc 5 bài JBHI → bảng related work, `COMPLIANCE.md` | Xong phần không cần tài khoản (reports/p2). Chờ anh: Scopus/IEEE Xplore |
| P3 | Pipeline không rò rỉ + baseline đặc trưng (RF/XGB), so với bài gốc PADS | **Xong.** PD vs HC RF BA 0.827±0.011, AUROC 0.906; PD vs DD RF AUROC 0.747. Rò rỉ theo cửa sổ thổi phồng +0.11/+0.19. CI đã sửa: `reports/p3/ci_fixed.csv` |
| P4 | DL baselines + DS-TCN-MIL, nested CV → bảng RQ1 | **Xong (1 seed).** PD vs HC DS-TCN AUROC 0.856; PD vs DD DL 0.635–0.685 < RF 0.747. CI p4a–c chưa tính lại (thiếu oof trên máy) |
| P5 | PTQ/QAT INT8 + hồ sơ triển khai TFLM x86 bit-exact | **Xong** (p56a raw, p56b inv; Kaggle CPU). Sai lệch D2: depthwise stride (xem PROTOCOL_P56.md) |
| B0 | Dựng ESP-IDF + TFLM, kiểm tra bo XH-S3E | **Xong 13:02.** ESP32-S3 QFN56 rev v0.2, flash 16 MB, PSRAM 8 MB, /dev/ttyACM0, ESP-IDF v5.3.2 |
| B1 | Đo latency/arena/flash + bit-exact trên bo, dữ liệu PADS phát lại (sau P5) | **B1-real xong 16/11 01:09:** ptq/qat/qat_sr fold0 đều 50/50 bit-exact; ESP-NN 11.5 ms, ANSI 70.3 ms, arena 21.2 kB, app ~449 kB (`reports/board/B1_REAL_SUMMARY.txt`). Trước đó: đường ống xong (`firmware/tools/run_bench.sh`). Mô hình giả không dilation: 50/50 bit-exact, 9.8 ms ESP-NN / 61.6 ms ANSI. Có dilation: 0/50 → cấm dilation |
| P6 | Dịch chuyển mô phỏng + kiểm định ngoài Monipar → **cổng JBHI/BSPC** | **Cổng: H3 ủng hộ (raw) → JBHI.** Monipar xong 16/11 03:55 (p56m v3): zero-shot dao động mạnh giữa biến thể (acc 0.49–0.95, KTC rộng 0.3–0.4), chỉ mô tả, không kết luận |
| P7 | Thống kê, ablation, hình/bảng cuối | **Kiểm định chính xong** (`scripts/p7_p56_stats.py`, `reports/p7/`): H1 không chứng minh (ΔAUROC −0.063 [−0.115; −0.013]); H2 không khác (+0.008); H3 ủng hộ +0.133 [+0.062; +0.207], p_Holm 0.0015. Seed 1–2 xong 05:50: H3 ủng hộ ở 2/3 seed (TB +0.100 ± 0.048); H1 âm ở cả 3 (PTQ −0.053 ± 0.018). Còn: Hình 1, Hình 3 |
| P8 | Viết Methods + Results vào IEEEtran | |
| P9 | Viết Intro, Related work, Discussion, Abstract; cắt về ≤ 8 trang | |
| P10 | Gói nộp: cover letter, 4 phản biện, checklist JBHI | |

## Đầu vào cho phiên sau (P1 → P2)
- `reports/p1/`: `audit_pads.md`, `audit_monipar.md`, `env.md`, `STATUS.txt`
- Quyết định cổng mục 5: Monipar có bài nghỉ + giữ tay không
- (song song) anh gửi kết quả 3 chuỗi Scopus ở mục 9 của kế hoạch, dạng CSV export

## Đầu vào cho phiên sau (P7 → P8, 16/11)
- Kết quả chính: `reports/p7/hyp_PD_vs_HC_raw.json`; bo: `reports/board/B1_REAL_SUMMARY.txt`; Monipar: `reports/p56m/` khi xong.
- Đóng góp viết lại: PTQ thông thường làm giảm AUROC; QAT khắc phục với chi phí trên bo bằng 0; ngẫu nhiên hóa cảm biến chỉ giúp ở quay lớn → không phải đóng góp chính.
- Việc còn: seed 1–2 cho raw (cam kết trong đề cương); kéo oof p4a–c để sửa CI; viết Results/Methods vào paper/main.tex; phản biện mù ở P9.
