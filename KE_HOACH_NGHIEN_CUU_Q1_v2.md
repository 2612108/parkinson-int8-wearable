# Kế hoạch nghiên cứu v2 — Parkinson trên cảm biến cổ tay, mô hình lượng tử hóa cho vi điều khiển

*Bản rà soát ngày 15/11/2026 (cập nhật lần 2: chốt tạp chí đích là IEEE JBHI). Thay thế bản v1. Nguyên tắc: toàn bộ thực nghiệm làm trên máy tính, không đo trên người.*

## 1. Kết luận rà soát (đọc phần này trước)

| Hạng mục v1 | Kết quả kiểm chứng | Xử lý ở v2 |
|---|---|---|
| ALAMEDA có raw signal | Sai. Chỉ có CSV 92 đặc trưng + 4 nhãn run (5.8 MB) | Bỏ khỏi nhánh deep learning; chỉ dùng tùy chọn ở mức đặc trưng |
| BIOCLITE (Zenodo, 2023, 50 Hz) | Không tìm thấy dữ liệu công bố; mới có bài protocol 2025 | Thay bằng **Monipar** (cùng nhóm tác giả, đã có trên Zenodo) |
| mPower để test zero-shot | Điện thoại, bài đi bộ, không phải cổ tay | Bỏ |
| "Tremor detection" trên PADS | PADS chỉ có nhãn chẩn đoán PD / DD / HC, không có nhãn run theo đoạn | Đổi bài toán chính thành phân loại bệnh ở mức bệnh nhân |
| Gap 1: chưa ai làm cross-dataset | Sai. Đã có ít nhất 3 bài 2025–2026 (xem mục 2) | Thu hẹp gap |
| Gap 3: chưa ai chạy trên ESP32 | Sai một phần. Nhiều hệ ESP32 đo run; đã có TinyML PD trên Arduino Nano BLE | Thu hẹp gap |
| Kỳ vọng accuracy > 92% | Không thực tế: bài gốc PADS chỉ đạt balanced accuracy khoảng 79% (PD vs HC) và 69% (PD vs DD) khi chỉ dùng tín hiệu chuyển động | Bỏ bảng kỳ vọng; báo cáo theo số thực đo được |
| BSPC là Q1 | Scimago/Scopus: Q1. JCR theo JIF: Q2 (33/130, Engineering Biomedical); theo JCI: Q1 | Giữ BSPC nếu khoa tính theo Scimago; xem mục 6 |
| Sensors (MDPI), JNER | Open access thu phí | Loại |

## 2. Tài liệu đã có (để không làm trùng)

**Trên PADS (469 người: 276 PD, 79 HC, 114 DD theo trường `condition`/`label` của bộ dữ liệu — audit 15/11; bản trước ghi 291/99 là sai; Apple Watch, 2 cổ tay, 100 Hz, acc + gyro; gia tốc đã loại trọng lực):**
- Varghese et al., npj Parkinson's Disease 2024 — bài gốc, nested CV, ML cổ điển + DL.
- MFAM (arXiv 2510.10558) — mạng đối kháng đa tần + attention MIL, PD vs DD.
- Self-supervised dual-channel cross-attention (arXiv 2604.18372) — có đo suy luận 48 ms trên **Raspberry Pi**, tự nhận hạn chế là chỉ dùng một dataset.
- CNN–Transformer + blockchain (Egyptian Informatics Journal 2026) — tự nhận cần nén/lượng tử hóa để chạy trên thiết bị biên.
- Một bản tái lập trên GitHub cho thấy pipeline cũ bị rò rỉ dữ liệu và kết quả tụt khi sửa → rò rỉ là vấn đề có thật trên PADS.

**Cross-dataset cho run/triệu chứng vận động (đã có):**
- Context-agnostic ML (2026): huấn luyện trên Levodopa Response Study, kiểm tra trên ALAMEDA, dùng Random Forest trên đặc trưng.
- Rest tremor & dyskinesia (2026): MJFF-LRS + Verily Study Watch.
- ParaDigMa (npj PD 2025): PD@Home → Personalized Parkinson Project, mã nguồn mở.

**TinyML cho Parkinson (đã có):** găng tay Arduino Nano 33 BLE với mô hình lượng tử hóa trên dữ liệu tự thu; TinyML cho freezing of gait; lượng tử hóa cho giọng nói PD.

**Lượng tử hóa dưới domain shift:** đã nghiên cứu cho ảnh (PTQ khá bền với domain shift) và HAR nói chung; một bài HAR 2026 ghi nhận kết quả INT8 trên máy tính không dự báo đúng kết quả trên vi điều khiển.

## 3. Khoảng trống thật (đã thu hẹp)

Chưa tìm thấy công trình nào trả lời câu hỏi sau cho tín hiệu quán tính cổ tay của bệnh nhân Parkinson:

> **Mô hình đủ nhỏ cho vi điều khiển, sau khi lượng tử hóa INT8, có giữ được khả năng tổng quát hóa và độ tin cậy xác suất khi đổi thiết bị/đoàn hệ không?**

Cụ thể còn trống ba điểm:
1. Các bài trên PADS đều đánh giá trong một dataset; chưa có kiểm định ngoài trên thiết bị khác.
2. Các bài cross-dataset dùng mô hình đặc trưng hoặc mô hình FP32, không xét lượng tử hóa và ngân sách bộ nhớ vi điều khiển.
3. Các bài TinyML PD dùng dữ liệu tự thu nhỏ, không có đánh giá ở mức bệnh nhân trên dữ liệu lâm sàng công khai, không xét domain shift.

*Lưu ý trung thực:* mới rà bằng tìm kiếm web. Trước khi viết câu "to the best of our knowledge" cần tra thêm Scopus/IEEE Xplore với từ khóa ở mục 9.

## 4. Đề tài đề xuất

**Tên tạm:** *Quantization under device shift: a leakage-free benchmark and a shift-robust tiny network for wrist-inertial Parkinson's disease assessment*

**Câu hỏi nghiên cứu**
- RQ1. Dưới giao thức không rò rỉ, ở mức bệnh nhân, mô hình ≤ 50 kB đạt bao nhiêu so với mô hình lớn trên PADS (PD vs HC, PD vs DD)?
- RQ2. PTQ và QAT làm thay đổi thế nào độ chính xác, độ hiệu chuẩn (ECE) và mức đồng thuận FP32–INT8 khi dữ liệu bị dịch chuyển (đổi thiết bị, tần số lấy mẫu, dải đo, nhiễu, hướng đeo)?
- RQ3. Huấn luyện QAT kết hợp ngẫu nhiên hóa cảm biến có thu hẹp được khoảng tụt đó không?

**Đóng góp dự kiến**
- C1. Benchmark không rò rỉ trên PADS + kiểm định ngoài zero-shot trên Monipar (đồng hồ khác, chỉ gia tốc kế).
- C2. Nghiên cứu hệ thống đầu tiên về tương tác giữa lượng tử hóa và domain shift cho bài toán này, gồm cả dịch chuyển thật (Monipar, hai thiết bị của LRS) và dịch chuyển mô phỏng có kiểm soát.
- C3. Mạng nhỏ bền với dịch chuyển: khối vào bất biến hướng đeo + sensor-randomization + QAT + gộp attention-MIL lên mức bệnh nhân.
- C4. Hồ sơ triển khai tính trên máy: kích thước flash, tensor arena, MACs, kết quả INT8 bit-exact bằng TFLite Micro bản x86.

## 5. Dữ liệu

| Bộ | Vai trò | Thiết bị / tín hiệu | Nhãn | Truy cập |
|---|---|---|---|---|
| **PADS** | Chính: huấn luyện + CV | Apple Watch S4, 2 cổ tay, acc + gyro, 100 Hz, 11 bài 10–20 s | PD 276 / HC 79 / DD 114 (469 người) | PhysioNet, mở, CC BY-NC-SA 4.0 |
| **Monipar** | Kiểm định ngoài zero-shot | Smartwatch thương mại, acc 3 trục, 8 bài tập theo MDS-UPDRS III | 21 PD + 7 HC, có MDS-UPDRS | Zenodo 8104853, mở |
| **MJFF Levodopa Response Study** | Nhánh run (tùy chọn nhưng nên có) | GENEActiv + Pebble ở hai cổ tay, chỉ gia tốc kế | Điểm run 0–4 theo từng bài, bác sĩ chấm | Synapse, phải xin quyền và theo điều khoản MJFF |
| ALAMEDA | Tùy chọn, mức đặc trưng | 92 đặc trưng GENEActiv | Có/không run | Zenodo 10782573 |

Hạn chế phải nêu trong bài: Monipar nhỏ (28 người) và không có gyro nên nhánh kiểm định ngoài chỉ dùng gia tốc kế; hai thiết bị trong LRS đeo ở hai tay khác nhau (tay nặng và tay nhẹ) nên chênh lệch không thuần do thiết bị.

**Cổng quyết định sau tuần 1:** nếu Monipar không khớp bài tập với PADS (cần ít nhất tư thế nghỉ và giữ tay) thì nhánh kiểm định ngoài chuyển sang LRS và bài toán run.

## 6. Tạp chí (không mất phí, không cần open access)

| Vai trò | Tạp chí | Chỉ mục | Chi phí đường Traditional | Ghi chú |
|---|---|---|---|---|
| **Đích (đã chốt)** | **IEEE Journal of Biomedical and Health Informatics (JBHI)** | SCIE; Q1 theo dữ liệu trích dẫn mở. Cần xác nhận trên Clarivate Master Journal List | 0 nếu ≤ 8 trang; trang 9–10 là 250 USD/trang; từ trang 11 là 350 USD/trang | Phạm vi hợp: có số chuyên đề 2025 về domain adaptation/generalization, có bài TinyML cho MCU năm 2026 |
| Dự phòng 1 | Biomedical Signal Processing and Control | SCIE; Scimago Q1; JCR-JIF Q2 | 0 | Dùng nếu bài nghiêng về nghiên cứu thực nghiệm (xem mục 6d) |
| Dự phòng 2 | IEEE Sensors Journal | SCIE | 0 nếu ≤ 8 trang; 175 USD/trang vượt | |

### 6a. Yêu cầu của JBHI (đọc từ trang hướng dẫn tác giả, 15/11/2026)

- [ ] Template IEEE hai cột; nộp PDF; hình và bảng nằm trong bài.
- [ ] **Bài chính ≤ 8 trang** để không mất phí. Giới hạn cứng 14 trang **tính cả phụ lục**, áp dụng cả khi sửa bài → phải chừa chỗ cho thí nghiệm reviewer yêu cầu thêm.
- [ ] Abstract ≤ 250 từ, đủ: mục tiêu, phương pháp, kết quả chính, kết luận, một câu về ý nghĩa y sinh. **Không viết tắt**, không chú thích, không trích dẫn, không công thức.
- [ ] Mọi tác giả có ORCID và dùng email cơ quan hoặc email IEEE (không Gmail). Không được thêm tác giả sau khi nộp.
- [ ] Cover letter dán vào ô văn bản: nêu điểm mới, ý nghĩa với JBHI, cam kết chưa nộp nơi khác.
- [ ] Đề xuất 4 người phản biện, khác cơ quan với mọi tác giả, không xung đột lợi ích.
- [ ] Mẫu đồng ý tác giả có chữ ký tay (bài nhiều tác giả), tải lên dạng tài liệu bổ trợ.
- [ ] Dữ liệu công khai: trích dẫn nguồn và URL (PADS, Monipar). Không cần IRB riêng nhưng phải nói rõ là dữ liệu công khai đã ẩn danh.
- [ ] Khai báo dùng AI trong mục Acknowledgment (theo quy định chung của IEEE — đọc lại IEEE Author Center trước khi nộp).
- [ ] Tối thiểu 3 phản biện; phản biện biết tên tác giả.
- [ ] Không yêu cầu highlights hay graphical abstract.

### 6b. Ngân sách 8 trang

| Phần | Trang |
|---|---|
| Abstract + Introduction (gap, đóng góp) | 1.0 |
| Related work (bảng so sánh thay cho văn xuôi) | 0.75 |
| Dữ liệu + giao thức đánh giá | 1.0 |
| Phương pháp (mô hình, QAT, mô phỏng dịch chuyển) | 1.75 |
| Kết quả: 3 bảng + 3 hình | 2.25 |
| Thảo luận + hạn chế + kết luận | 0.75 |
| Tài liệu tham khảo (khoảng 40) | 0.5–0.75 |

Hệ quả: bài chính chỉ giữ **hai bài toán trên PADS, kiểm định ngoài Monipar, và thí nghiệm dịch chuyển**. Ablation chi tiết, cửa sổ 2/4/8 s, nhánh run trên LRS đưa vào phụ lục (tối đa khoảng 4 trang).

### 6c. Bài JBHI ứng viên để trích dẫn (3–5 bài, chưa đọc tóm tắt)

Tìm qua Crossref lọc theo ISSN 2168-2194. Phải đọc tóm tắt từng bài trước khi viết câu trích dẫn.

| Bài | DOI | Dùng ở đâu |
|---|---|---|
| Leveraging Unlabelled Data in Multiple-Instance Learning … Parkinsonian Tremor in Free-Living Conditions (2023) | 10.1109/jbhi.2023.3267095 | Related work: MIL cho PD |
| Cross-Modal Federated TinyML for MCU-based Internet of Medical Things (2026) | 10.1109/jbhi.2026.3679191 | Related work: TinyML trên MCU |
| Few-Shot Learning for Cross-Domain Human Activity Recognition Using Wearable Sensors (2026) | 10.1109/jbhi.2026.3713678 | Domain shift cảm biến đeo |
| Cross-Domain Human Activity Recognition via Domain Adaptation and Fused Attention (2025) | 10.1109/jbhi.2025.3528008 | Domain shift cảm biến đeo |
| Multi-Feature Fusion and Compressed Bi-LSTM for Memory-Efficient Heartbeat Classification on Wearable Devices (2026) | 10.1109/jbhi.2026.3669719 | Nén mô hình cho thiết bị đeo |

### 6d. Rủi ro khi nộp JBHI và cách xử lý

| Rủi ro | Mức | Xử lý |
|---|---|---|
| JBHI đòi đóng góp phương pháp, không chỉ benchmark | Cao | C3 phải cho cải thiện có ý nghĩa thống kê dưới dịch chuyển. **Cổng tuần 5–6:** nếu QAT + ngẫu nhiên hóa cảm biến không thu hẹp khoảng tụt so với PTQ thường (kiểm định cặp, p < 0.05 sau hiệu chỉnh), chuyển bài sang dạng nghiên cứu thực nghiệm và nộp BSPC |
| Kiểm định ngoài quá nhỏ (Monipar: 21 PD + 7 HC) | Cao | Báo cáo khoảng tin cậy bootstrap, không kết luận mạnh; bổ sung LRS nếu được cấp quyền; nêu rõ trong Limitations |
| Dịch chuyển mô phỏng không phải dịch chuyển thật | Trung bình | Tham số lấy từ datasheet; đối chiếu chiều hướng với dịch chuyển thật PADS → Monipar |
| Không có số đo trên vi điều khiển | Trung bình | Chỉ khẳng định "tương thích ngân sách MCU, kiểm chứng bit-exact bằng TFLite Micro"; không ghi ESP32-S3 trong tên bài; ghi là hạn chế. Phương án nửa ngày trên board vẫn để ngỏ |
| PADS mất cân bằng (276 PD / 79 HC / 114 DD) | Trung bình | Balanced accuracy, AUROC, trọng số lớp; không dùng accuracy thô |
| Trùng với bài 2026 đã đo trên Raspberry Pi | Thấp | Khác biệt: lượng tử hóa INT8, ngân sách MCU, kiểm định ngoài, phân tích dịch chuyển |

## 7. Phương pháp

**Giao thức đánh giá (phần quyết định chất lượng bài)**
- Chia theo bệnh nhân, nested CV 5×5 phân tầng theo nhãn; hai cổ tay và mọi bài tập của một người luôn cùng một fold.
- Chuẩn hóa, chọn siêu tham số, chọn ngưỡng, calibration set của PTQ: chỉ lấy từ tập huấn luyện.
- Chỉ số: balanced accuracy, AUROC, sensitivity/specificity ở mức bệnh nhân; khoảng tin cậy 95% bằng bootstrap theo bệnh nhân; kiểm định cặp (DeLong, McNemar) có hiệu chỉnh đa so sánh; ECE và Brier cho hiệu chuẩn.
- Báo cáo cả kết quả âm tính và cả bản "rò rỉ" để cho thấy mức thổi phồng.

**Tiền xử lý:** lấy lại mẫu về 50 Hz, lọc thông dải 0.5–20 Hz, cửa sổ 4 s chồng 50% (thử thêm 2 s và 8 s), gộp cửa sổ → bài tập → bệnh nhân.

**Mô hình so sánh:** đặc trưng phổ + Random Forest/XGBoost; 1D-CNN; DS-CNN; TCN nhỏ; CNN-LSTM; một mô hình lớn làm trần trên (InceptionTime hoặc Transformer nhỏ). Mô hình đề xuất: DS-TCN ≤ 30k tham số, khối vào bất biến hướng (độ lớn vector + thành phần sau khi quay về trục trọng lực), attention-MIL.

**Lượng tử hóa:** FP32, PTQ INT8 (per-channel), QAT INT8, thêm biến thể activation 16-bit. Một framework duy nhất: TensorFlow/Keras + TFMOT + TFLite.

**Dịch chuyển mô phỏng (thay cho đo trên phần cứng):**
- Tần số lấy mẫu 100 → 50 → 25 Hz, jitter thời gian lấy mẫu.
- Dải đo và độ phân giải ADC: ±2/±8/±16 g, 12–16 bit, có bão hòa.
- Nhiễu trắng và bias theo datasheet của IMU phổ biến (MPU6050, BMI270, LSM6DS3).
- Quay trục ngẫu nhiên (đeo lệch), đổi tay trái/phải, mất kênh gyro.
Mỗi yếu tố quét theo mức độ, vẽ đường cong suy giảm cho FP32, PTQ, QAT.

**Hồ sơ triển khai trên máy:** kích thước `.tflite`, tensor arena đo bằng TFLite Micro biên dịch cho x86, MACs, số phép toán được hỗ trợ bởi ESP-NN; kiểm tra INT8 bit-exact giữa TFLite và TFLM.

**Về số đo trên board:** không đưa vào kế hoạch chính. Phương án dự phòng không cần người đeo: nạp `model.h`, cho board chạy trên dữ liệu phát lại từ PADS, đọc thời gian suy luận — khoảng nửa ngày. Nếu không làm thì tiêu đề và abstract chỉ nói "microcontroller-class budget", không ghi ESP32-S3.

## 8. Lộ trình (khoảng 10 tuần, chỉ trên máy tính)

| Tuần | Việc | Sản phẩm |
|---|---|---|
| 1 | Tải PADS, Monipar; nộp đơn xin LRS trên Synapse; audit (số người, bài tập, nhãn, tần số) | Bảng audit + quyết định ở cổng mục 5 |
| 1–2 | Tra Scopus/IEEE Xplore để khóa gap; đọc tóm tắt các bài JBHI ứng viên (mục 6c) | Bảng related work, `COMPLIANCE.md` |
| 2–3 | Pipeline dữ liệu không rò rỉ + baseline đặc trưng | Kết quả tái lập gần bài gốc PADS |
| 3–5 | Các mô hình DL + mô hình đề xuất, nested CV | Bảng kết quả chính (RQ1) |
| 5–6 | PTQ/QAT, hồ sơ triển khai | Bảng FP32 vs INT8 |
| 6–8 | Thí nghiệm dịch chuyển mô phỏng + kiểm định ngoài Monipar/LRS | Hình đường cong suy giảm (RQ2, RQ3) |
| 8–9 | Ablation, thống kê, hình | Bộ hình/bảng hoàn chỉnh |
| 9–10 | Viết thẳng vào template `IEEEtran` (≤ 8 trang + phụ lục), cover letter, 4 phản biện đề xuất, mẫu đồng ý có chữ ký | Gói nộp JBHI |

## 9. Việc cần anh Tuyến quyết định hoặc làm

1. Xác nhận JBHI trên Clarivate Master Journal List (SCIE, quartile JCR mới nhất).
2. Đăng ký tài khoản Synapse và nộp yêu cầu truy cập Levodopa Response Study (cần tên và email cơ quan).
3. Tra Scopus bằng tài khoản trường với các chuỗi: `"PADS" AND Parkinson AND smartwatch`; `Parkinson AND (TinyML OR microcontroller OR quantiz*) AND (wearable OR inertial)`; `Parkinson AND wearable AND ("cross-dataset" OR "external validation" OR "domain shift")`.
4. ~~Xác nhận bỏ hẳn hay giữ phương án đo trên board.~~ **Đã chốt 15/11: GIỮ.** Bo Sparkle IoT XH-S3E (ESP32-S3 N16R8) cắm sẵn trên máy; Antigravity build/flash (`ANTIGRAVITY_BOARD.md`). Đo latency, arena, flash, bit-exact trên dữ liệu PADS phát lại — không đo người. Khi có số đo thật, tiêu đề/abstract được phép ghi ESP32-S3; rủi ro "không có số đo trên vi điều khiển" ở mục 6d coi như xử lý xong.

## 10. Nguồn

- PADS: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/ ; bài gốc https://doi.org/10.1038/s41531-023-00625-7
- Monipar: https://zenodo.org/records/8104853 ; https://doi.org/10.3389/fneur.2023.1326640
- Levodopa Response Study: https://doi.org/10.1038/s41597-021-00830-0 ; https://www.michaeljfox.org/news/levodopa-response-study
- ALAMEDA tremor: https://zenodo.org/records/10782573
- BioClite protocol: https://www.researchprotocols.org/2025/1/e72820/PDF
- MFAM: https://arxiv.org/abs/2510.10558
- SSL cross-attention trên PADS: https://arxiv.org/abs/2604.18372
- CNN–Transformer trên PADS: https://www.sciencedirect.com/science/article/pii/S1110866526001519
- Bản tái lập có sửa rò rỉ: https://github.com/aniketkhetan/PADS-Parkinsons-Detection
- Context-agnostic ML: https://pmc.ncbi.nlm.nih.gov/articles/PMC13660283
- ParaDigMa: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12246068/
- Rest tremor & dyskinesia: https://www.researchgate.net/publication/408147667
- Găng tay TinyML: https://journals.sagepub.com/doi/10.1177/14613484251414903
- Nén mô hình dưới distribution shift: https://arxiv.org/abs/2401.12014
- µBi-ConvLSTM (INT8 trên MCU): https://arxiv.org/abs/2602.06523
- BSPC chỉ mục: https://www.letpub.com/journal-selector/journal/1175 ; https://researchjournalrank.com/journal/biomedical-signal-processing-and-control
- JBHI hướng dẫn tác giả và phí trang: https://embs.org/jbhi/prepare-and-submit-your-manuscript
- Crossref (lọc ISSN JBHI 2168-2194 / 2168-2208): https://api.crossref.org/works
