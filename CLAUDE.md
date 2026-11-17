# CLAUDE.md — dự án bài báo Parkinson (IEEE JBHI)

Tác giả: Lại Quang Tuyến (Quang-Tuyen Lai), HCMUTE. Trả lời anh Tuyến bằng tiếng Việt, xưng "em", gọi "anh".

## Đọc trước khi làm gì
1. `KE_HOACH_NGHIEN_CUU_Q1_v2.md` — kế hoạch đã chốt (gap, RQ, dữ liệu, giao thức, tạp chí JBHI, ngân sách 8 trang).
2. `NHAT_KY_PHIEN.md` — đang ở phiên nào, đầu vào/đầu ra mỗi phiên. **Cuối mỗi phiên phải cập nhật file này.**

## Cách làm việc
- Anh Tuyến muốn Claude **tự làm liên tục từ P1 đến khi ra bài báo**, làm tới đâu báo kết quả tới đó (tóm tắt ngắn: số liệu chính, quyết định, việc cần anh làm).
- Chỉ dừng hỏi anh khi: (a) cần tài khoản/quyền mà Claude không có (Scopus trường, Synapse/LRS, ORCID, chữ ký), (b) gặp cổng quyết định trong kế hoạch (cổng Monipar ở mục 5, cổng JBHI/BSPC ở mục 6d) — trình bày số liệu và đề xuất, (c) thao tác không hoàn tác được (nộp bài, xóa dữ liệu).
- Máy này KHÔNG có GPU. Mọi tính toán nặng chạy trên **Kaggle** qua Kaggle CLI (token đã có ở `~/.kaggle/access_token`; nếu CLI cần thì `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)`). Không bao giờ in hay ghi token vào file trong repo.
- Mỗi phiên N: notebook ở `kaggle/pN/` (+ `kernel-metadata.json`, `is_private: true`), `kaggle kernels push`, theo dõi `kaggle kernels status`, chỉ kéo báo cáo nhỏ về `reports/pN/` (không kéo dữ liệu thô).
- Dữ liệu thô nằm trong output kernel `<user>/parkinson-p1-audit`; phiên sau gắn qua `kernel_sources` (hoặc tạo Kaggle Dataset riêng tư nếu tiện hơn).
- Dùng GPU Kaggle cho P4–P6; P1–P3 chạy CPU.
- **Bo ESP32-S3** (Sparkle IoT XH-S3E, N16R8) cắm sẵn qua USB-C. Việc build/flash firmware giao cho **Antigravity**: Claude viết code vào `firmware/` và hướng dẫn vào `ANTIGRAVITY_BOARD.md`, rồi báo anh Tuyến giao cho Antigravity; đọc kết quả ở `reports/board/`. Bo chỉ chạy dữ liệu PADS phát lại, không đo người.

## Quy tắc khoa học (không thương lượng)
- Chia theo bệnh nhân; mọi chuẩn hóa/chọn siêu tham số/ngưỡng/calibration PTQ chỉ từ tập train. Không rò rỉ.
- Không bịa số liệu, không làm đẹp kết quả; kết quả âm tính vẫn báo cáo. Mọi con số trong bài phải truy được về file trong `reports/`.
- Ghi seed, phiên bản thư viện, cấu hình vào `reports/pN/`.
- Tài liệu tham khảo phải có DOI và đã đọc tóm tắt; trích 3–5 bài JBHI (mục 6c).
- Viết thẳng vào template `IEEEtran`, ≤ 8 trang bài chính. Khai báo dùng AI trong Acknowledgment.

## Hợp đồng đầu ra P5 → bo (để Antigravity nạp ngay)
- Lưu mô hình INT8 (vào/ra int8) ở `reports/p5/models/<tên>.tflite`; 50+ cửa sổ test float32 `(N, T, C)` ở `reports/p5/bench_inputs.npy`.
- Đường ống bo đã sẵn: `firmware/tools/tflite_to_c.py` → `firmware/b1_bench/` → `firmware/tools/check_serial.py` (xem `ANTIGRAVITY_B1.md`).
- **QUY TẮC BẮT BUỘC (đo trên bo 15/11, reports/board/B1_DIAG.txt):** KHÔNG dùng `DepthwiseConv1D`/`Conv1D` với `dilation_rate>1`. TFLite tách dilation thành SPACE_TO_BATCH_ND/BATCH_TO_SPACE_ND; trên TFLM (cả ESP-NN lẫn ANSI C) đầu ra **sai lệch tới 106–113 LSB, 0/50 bit-exact**, và chậm gấp đôi (19.4 ms so với 9.8 ms). Bản không dilation: **50/50 bit-exact** cả ESP-NN và ANSI. Muốn trường tiếp nhận rộng thì dùng stride/pooling hoặc xếp chồng nhiều khối kernel lớn hơn. Trước khi nộp mô hình cho bo, chạy `firmware/tools/tflite_to_c.py` và kiểm tra danh sách op không có SPACE_TO_BATCH_ND/BATCH_TO_SPACE_ND.
- ESP-NN tăng tốc 6.3× so với ANSI C trên mô hình giả không dilation (9.8 ms so với 61.6 ms/cửa sổ 4 s). Báo cáo cả hai trong bài.

## Quy trình viết + phản biện thử (bắt buộc, theo skill của anh Tuyến — bản sao ở `_ref/skills/`)
- `_ref/skills/journal-paper-submission/SKILL.md` và `_ref/skills/paper-mock-review/SKILL.md` (+ `references/`).
- Đề cương định trước cho P5/P6: `PROTOCOL_P56.md` (chốt 15/11 23:10, trước khi xem kết quả). Không đổi giả thuyết/ngưỡng sau khi thấy số; mọi thay đổi ghi vào "Nhật ký sửa đổi".
- Văn phong: tiếng Anh Mỹ, câu 15–25 từ, tách câu > 30 từ. Kết luận chỉ theo tiêu chí chính; báo cáo kết quả âm tính; CI chứa 0 ≠ tương đương.
- Trước khi báo "sẵn sàng nộp": chạy tác tử phản biện mù (định nghĩa ở `_ref/skills/paper-mock-review/references/paper-reviewer-agent.md`), kiểm lại từng nhận xét, lưu `paper/revision/mock_review_<ngày>.md`, lặp đến khi ≥ Minor revision.

## Bản thảo
Khung bài JBHI đã có ở `paper/` (IEEEtran, biên dịch được, 3 trang khung). P8–P9 điền vào `paper/main.tex`, thay mọi `\TBD{}` bằng số từ `reports/`. Chạy `python paper/fetch_bib.py` để sinh `refs.bib` từ DOI (máy này ra Internet được); thêm DOI PhysioNet của PADS vào `paper/refs_doi.txt`.

## Lộ trình phiên
Xem bảng trong `NHAT_KY_PHIEN.md` (P1 → P10).
