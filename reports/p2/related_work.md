# P2 — Related work và khóa gap (15/11/2026)

Nguồn tra: Crossref API, OpenAlex API (tóm tắt), Semantic Scholar API, web search, trang arXiv/PMC/GitHub.
**Chưa tra Scopus/IEEE Xplore** (cần tài khoản trường) — câu "to the best of our knowledge" chưa được phép viết cho tới khi anh gửi kết quả 3 chuỗi Scopus.

## A. 5 bài JBHI (ISSN 2168-2194) — DOI đã xác minh qua Crossref, đã đọc tóm tắt

| # | Bài | DOI | Tóm tắt (đã đọc) | Dùng ở đâu |
|---|---|---|---|---|
| J1 | Papadopoulos & Delopoulos, "Leveraging Unlabelled Data in MIL Problems for Improved Detection of Parkinsonian Tremor in Free-Living Conditions", JBHI 27(7):3569–3578, 2023 | 10.1109/JBHI.2023.3267095 | Kết hợp semi-supervised (Virtual Adversarial Training) với MIL; phát hiện run PD từ gia tốc tay ngoài đời thực, nhãn thô ở mức người | Related work MIL; biện minh gộp attention-MIL lên mức bệnh nhân |
| J2 | Ibrar et al., "Cross-Modal Federated TinyML for MCU-based Internet of Medical Things", JBHI 2026 (early access) | 10.1109/JBHI.2026.3679191 | TinyCFL: federated + TinyML trên MCU, dung hợp đa phương thức trung gian, đánh giá UP-Fall | Related work TinyML y sinh trên MCU |
| J3 | Zheng et al., "Few-Shot Learning for Cross-Domain HAR Using Wearable Sensors", JBHI 2026 (early access) | 10.1109/JBHI.2026.3713678 | CFSH-Net nhẹ, wavelet; thích nghi cross-position/user/dataset bằng few-shot, 6 benchmark HAR | Domain shift cảm biến đeo (vị trí, người, dataset) |
| J4 | Zhu et al., "Cross-Domain HAR via Domain Adaptation and Fused Attention", JBHI 29(8):5394–5404, 2025 | 10.1109/JBHI.2025.3528008 | TCA-BiLSTM: MMD/TCA căn chỉnh dữ liệu từ cảm biến ở vị trí khác nhau; DSADS, PAMAP2 | Domain shift; đối lập: họ cần dữ liệu đích, ta zero-shot |
| J5 | Nikandish, He, Haghi, "Multi-Feature Fusion and Compressed Bi-LSTM for Memory-Efficient Heartbeat Classification on Wearable Devices", JBHI 30(10):8897–8906, 2026 | 10.1109/JBHI.2026.3669719 | Bi-LSTM nén, mô hình 84k–1.25M; PTQ INT8 làm accuracy tụt (bản lớn INT8 88.4% so với 96.4% FP32) | Nén/PTQ cho thiết bị đeo; dẫn chứng PTQ có thể tụt mạnh |

## B. Công trình trên PADS

| Bài | Nguồn | Giao thức | Kết quả (theo nguồn) | Khác biệt với ta |
|---|---|---|---|---|
| Varghese et al. 2024, npj PD | 10.1038/s41531-023-00625-7 (+ đính chính 10.1038/s41531-024-00710-5) | Nested CV, ML cổ điển + DL | Bal. acc ≈ 0.76 (PD vs HC), ≈ 0.75 (PD vs DD) theo bảng so sánh trong repo tái lập; kế hoạch ghi ≈79%/69% chỉ dùng chuyển động — **P3 phải đọc lại bảng gốc để chốt con số** | Một dataset, FP32 |
| Khetan, repo tái lập 2025 | github.com/aniketkhetan/PADS-Parkinsons-Detection (không DOI → chỉ dùng làm bằng chứng phụ/footnote) | Rò rỉ do augment trước khi chia (≈150/188 hàng test có bản gần trùng); bản sạch chia theo người | Sạch: PD vs HC 0.789 ± 0.059; PD vs DD 0.638 ± 0.019 | Bằng chứng rò rỉ có thật |
| Zannat 2026, arXiv 2604.18372 v2 | 10.48550/arXiv.2604.18372 | Không nêu giao thức chia trong tóm tắt | Acc 93.2% / AUROC 0.963 (HC vs PD); 90.9% / 0.960 (PD vs DD); 124.5 ms/cửa sổ trên Raspberry Pi 4 | **Sửa kế hoạch:** bản v2 ghi 124.5 ms (không phải 48 ms). Không lượng tử hóa, không MCU, một dataset; số cao bất thường so với nested CV gốc → nêu cẩn trọng |
| Zhao et al. 2025, MFAM, arXiv 2510.10558 | 10.48550/arXiv.2510.10558 | PADS PD vs DD + dữ liệu riêng | Tóm tắt không có số | MIL + đa tần; FP32 |
| CNN–Transformer + blockchain, Egyptian Informatics J. 2026 | S1110866526001519 (DOI cần lấy ở P9) | — | — | Tự nhận cần nén cho biên |
| medRxiv 2026 latent motor score | 10.64898/2026.09.03.26362173 | preprint | — | Chỉ PADS |

## C. TinyML / MCU cho Parkinson

| Bài | DOI | Ghi chú |
|---|---|---|
| Parikh et al. 2026, găng tay, J. Low Freq. Noise Vib. Active Control | 10.1177/14613484251414903 | TFLite lượng tử hóa trên Arduino Nano BLE; dữ liệu tự thu; subject-independent, **không kiểm định ngoài** |
| TinyML nhận dạng hoạt động bệnh nhân PD (Ömer Halisdemir Üniv. 2022) | 10.28948/ngumuh.1144946 | Tạp chí trong nước, chỉ dẫn chứng phụ |

## D. Kết luận khóa gap (tạm thời)
Các tìm kiếm web/Crossref **không** tìm thấy công trình nào đồng thời: (1) tín hiệu quán tính cổ tay PD, (2) mô hình INT8 cho MCU, (3) kiểm định ngoài / domain shift. Ba điểm trống ở mục 3 của kế hoạch vẫn đứng. Trạng thái: **chờ Scopus + IEEE Xplore** để nâng thành khẳng định.

## E. Cập nhật cần đưa vào kế hoạch
- Latency bài SSL: 124.5 ms trên RPi 4 (v2), không phải 48 ms.
- Monipar (từ bài gốc PMC10750794): Tickwatch S2, 50 Hz, chỉ acc; 8 bài, trong đó **(1) run nghỉ 30 s, ngồi, tay trên tay ghế** và **(2) run tư thế 30 s, tay duỗi ngang ngực**; đồng hồ đeo bên tay triệu chứng nặng hơn (PD) hoặc tay thuận (HC); 21 PD + 7 HC, 174 lượt thu, 2–9 tuần/người. → Về giấy tờ, **cổng mục 5 qua** (có nghỉ + giữ tay). Ánh xạ dự kiến: Monipar rest ↔ PADS Relaxed1/2; postural ↔ PADS StretchHold; finger tapping ↔ PADS TouchIndex/PointFinger (cần audit dữ liệu thật để xác nhận).
- Monipar có nhiều lượt/người theo tuần → gộp lên mức bệnh nhân, không coi mỗi lượt là một mẫu độc lập.
