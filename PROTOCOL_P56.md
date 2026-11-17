# Đề cương phân tích định trước — P5/P6 (lượng tử hóa dưới dịch chuyển, kiểm định ngoài)

**Chốt lúc:** 15/11/2026 23:10 (UTC+7), TRƯỚC khi xem bất kỳ kết quả nào của p56a/p56b (đang chạy trên Kaggle CPU, đẩy 22:35).
Kết quả P3/P4 đã được xem trước thời điểm này; chúng KHÔNG nằm trong các giả thuyết dưới đây.
Mọi thay đổi sau thời điểm này ghi vào mục "Nhật ký sửa đổi" kèm lý do; phân tích không có trong đề cương này được gọi là *thăm dò* trong bài.

## Đóng băng (md5)
| File | md5 |
|---|---|
| scripts/p56_quant_shift.py | 2559ddf923799f297e2f3f9f87a4d7e7 |
| src/evalkit.py | 3c7e7a6c45f588448e5d622259660566 |
| src/features.py | 78c715a21028c0936e2e6f01f73303fe |
| src/pads_io.py | 9b352769b4ec5c6d206e200b93385863 |
| src/monipar_io.py | 6e23bc08c6d404ce51bcaf6cf00eb5a3 |
| kaggle/p56a/p56a.ipynb | f2e9d7095a02c969c848f18668761eec |
| kaggle/p56b/p56b.ipynb | d5f0d7112eff3f4689e93c49a6803d53 |

Siêu tham số, mức dịch chuyển (`SHIFTS`), cách chọn nhiệt độ/ngưỡng (trên validation sạch, giữ nguyên dưới dịch chuyển) cố định theo script trên.

## Thiết lập
- Bài toán chính: PD vs HC trên PADS (276/79), nested 5-fold theo bệnh nhân (seed 0), 20 % train làm validation.
- Mô hình: DS-TCN ~22.7k tham số, đầu vào `raw` (chính) và `inv` (phụ).
- Biến thể: fp32, fp32_sr, ptq, ptq_sr, qat, qat_sr.
- Tập dịch chuyển định trước **S** = 14 điều kiện khác `clean` trong `SHIFTS` (fs50, fs25, jitter10, jitter30, adc4g12b, adc2g10b, adc2g8b, noise_lo, noise_hi, rot15, rot45, rot180, mirror, nogyro).
- Mọi so sánh dùng dự đoán mức bệnh nhân gộp 5 fold (`patient_preds_*.csv`), bootstrap 2000 lần theo bệnh nhân, phân tầng theo nhãn, **cặp** (cùng bộ bệnh nhân cho hai biến thể).

## Giả thuyết và tiêu chí
**H1 — Không thua kém của INT8 khi không dịch chuyển (tiêu chí chính 1).**
ptq so với fp32, đầu vào raw, điều kiện clean. ΔAUROC = AUROC(ptq) − AUROC(fp32).
Không thua kém nếu cận dưới khoảng tin cậy 95 % hai phía của ΔAUROC > **−0.02**.
*Căn cứ ngưỡng:* sai số chuẩn bootstrap của AUROC ở P4 (cùng dữ liệu, cùng kiến trúc) là 0.024–0.026; −0.02 nhỏ hơn một sai số chuẩn, nghĩa là mất mát do lượng tử hóa phải nhỏ hơn độ bất định vốn có giữa các lần lấy mẫu.

**H2 — Lượng tử hóa làm suy giảm thêm dưới dịch chuyển (tiêu chí chính 2).**
Đại lượng: suy giảm trung bình D(v) = mean_{s∈S}[AUROC(v,clean) − AUROC(v,s)].
So sánh D(ptq) − D(fp32), raw. Ủng hộ H2 nếu khoảng tin cậy 95 % > 0. Nếu chứa 0: báo cáo "không thấy khác biệt", KHÔNG viết là "tương đương".

**H3 — Phương pháp đề xuất bền hơn (tiêu chí chính 3; cổng JBHI/BSPC ở mục 6d kế hoạch).**
Đại lượng: AUROC trung bình trên S, Ā(v) = mean_{s∈S} AUROC(v,s).
So sánh Ā(qat_sr) − Ā(ptq), raw. Ủng hộ H3 nếu khoảng tin cậy 95 % > 0.
- Nếu H3 được ủng hộ → giữ JBHI, đóng góp phương pháp C3.
- Nếu không → bài chuyển dạng nghiên cứu thực nghiệm (BSPC), báo cáo H3 âm tính thẳng thắn.

Ba tiêu chí chính kiểm định với hiệu chỉnh Holm (α = 0.05).

## Tiêu chí phụ (mô tả, không kiểm định suy diễn trừ khi ghi)
- Theo từng điều kiện s: ΔAUROC(ptq − fp32), ΔAUROC(qat_sr − ptq) với khoảng tin cậy, hiệu chỉnh Holm trong họ 14 so sánh.
- Phân tách đóng góp: fp32_sr vs fp32 (vai trò ngẫu nhiên hóa), qat vs ptq (vai trò QAT).
- Hiệu chuẩn: ECE, Brier; đồng thuận quyết định INT8 ↔ FP32 tương ứng.
- Đầu vào inv so với raw: chỉ mô tả (đã thấy ở P4 rằng inv kém hơn khi sạch).
- Monipar (21 PD/7 HC, zero-shot, chỉ gia tốc): AUROC, BA với CI 95 %; chỉ mô tả, KHÔNG kết luận tương đương hay vượt trội vì cỡ mẫu nhỏ; báo cáo riêng hiện tượng ngưỡng PADS không chuyển được nếu xảy ra.
- Bo ESP32-S3: latency, arena, flash, bit-exact /50 cho ptq và qat_sr fold 0.

## Hạn chế đã biết trước khi có kết quả
- Một seed cho P5/P6 (giới hạn tính toán). Nếu còn thời gian: chạy lại seed 1, 2 cho raw với cùng script, báo cáo trung bình ± SD — quyết định chạy thêm KHÔNG phụ thuộc vào kết quả seed 0.
- Dịch chuyển mô phỏng không thay thế thiết bị thật; Monipar là dịch chuyển thật duy nhất.

## Nhật ký sửa đổi
- **D1 — 2026-11-15 23:59 (giờ VN), TRƯỚC khi có bất kỳ kết quả p56 nào** (p56a, p56b đều RUNNING lúc 23:57).
  Lỗi: `bootstrap_ci` chỉ nhận một ngưỡng vô hướng; P3/P4 gọi với trung vị ngưỡng các fold trong khi ước lượng điểm dùng ngưỡng riêng từng fold → CI của sens/spec/BA có thể không chứa ước lượng điểm (thấy ở p4d cnn1d, cnn_lstm). AUROC/ECE không bị ảnh hưởng.
  Sửa: `bootstrap_ci` nhận thêm mảng ngưỡng theo từng bệnh nhân (vô hướng thì hành vi y hệt cũ). md5 mới `src/evalkit.py` = 7470888ac228db5794bde2ad34c2aadd (cũ 3c7e7a6c…).
  Ảnh hưởng đến P56: không đổi hành vi — p56_quant_shift.py chỉ gọi `bootstrap_ci` với ngưỡng vô hướng (Monipar); phân tích H1–H3 (P7) sẽ truyền mảng ngưỡng theo fold. Kernel p56a/p56b đang chạy dùng bản cũ, kết quả thô (patient_preds) không phụ thuộc hàm này.
  P3/P4: tính lại CI bằng `tools/recompute_ci.py` từ các file oof_*.csv, ghi `ci_fixed.csv`; số liệu trong bài lấy từ đó.
- **D2 — 2026-11-16 ~00:50, phát hiện SAU khi kéo kết quả p56.** Mã thực thi trên Kaggle (bản `scripts/p56_quant_shift.py` lưu trong output kernel, md5 84207c660051520238ddaa7e69d21873) KHÁC bản đóng băng (2559ddf9…) ở đúng một chỗ: trong khối DS-TCN, bước nhảy giảm mẫu nằm ở DepthwiseConv2D (7,1) stride (s,1) thay vì ở pointwise 1×1. Số tham số như nhau (22 737), không dilation; mọi phần khác (dữ liệu, chia fold, biến thể, dịch chuyển, hiệu chỉnh, chỉ số) giống hệt. Nguyên nhân: phiên bản kernel đẩy lên Kaggle được tạo từ bản script trước lần sửa cho GPU (lỗi quy trình, không có chủ ý). Xử lý: báo cáo kết quả của mã đã chạy thật; mô tả kiến trúc trong bài theo mã đã chạy; ghi rõ sai lệch này trong phần Phương pháp/Phụ lục. Không chạy lại để "chọn" bản tốt hơn. Bản đóng băng cũ được giữ nguyên trong repo để đối chiếu.
- **D3 — cùng thời điểm.** Phần Monipar KHÔNG chạy: script dừng ở `find_dir` ("không thấy Monipar dưới /kaggle/input") sau khi đã ghi xong toàn bộ kết quả PADS. Sẽ chạy riêng bằng chính mã đã thực thi (84207c…) với `--skip_cv --monipar`, sau khi kiểm tra đường dẫn dataset v2; Monipar vẫn chỉ là tiêu chí mô tả.
- **D4 — ghi chú phân tích (không đổi tiêu chí).** Script phân tích `scripts/p7_p56_stats.py` viết sau khi có dữ liệu nhưng thực hiện đúng định nghĩa H1–H3 ở trên (AUROC trên dự đoán bệnh nhân gộp 5 fold, bootstrap cặp phân tầng 2000, seed 0). p-value bootstrap = 2·min(tỉ lệ Δ* ≤ ngưỡng, tỉ lệ Δ* ≥ ngưỡng) với ngưỡng −0.02 cho H1 và 0 cho H2/H3; Holm trên 3 p. "Ủng hộ" yêu cầu cả tiêu chí CI lẫn p_Holm < 0.05. AUROC trung bình theo fold (từ shift_results.csv) chỉ báo cáo mô tả.
- **D3b — 2026-11-16 02:15.** Lượt p56m v1 lỗi môi trường (thiếu `tensorflow-model-optimization`, quên ô pip) và cho thấy dataset Kaggle chỉ chứa `pads/` (Monipar chưa từng được tải lên). v2: thêm ô pip giống p56a, tải Monipar trực tiếp từ Zenodo 8104853 trong notebook (kiểm md5). Script và src giữ nguyên md5 đã chạy (84207c66…, evalkit 3c7e7a6c…).
- **D3c — 2026-11-16 03:05.** p56m v2 tải Monipar thành công (md5 Zenodo khớp) nhưng huấn luyện lỗi TF CPU "only supports equal length strides" ở DepthwiseConv2D stride (2,1) (cùng lỗi từng gặp trên GPU; p56a không gặp, có thể do khác đường thực thi CPU). v3: chỉ trong notebook p56m, đổi stride depthwise (s,1) → (s,s). Vì chiều rộng tensor = 1 và padding "same", out_w = ⌈1/s⌉ = 1 và pad_w = 0, nên phép tính **giống hệt về toán học** (cùng trọng số, cùng đầu ra). Không đổi gì khác.
- **D5 — 2026-11-16 05:50. Seed 1, 2 (đã cam kết).** Chạy bằng mã đã chạy ở p56a với depthwise (s,s) ≡ (s,1) (D3c). Kết quả (pooled AUROC, cùng phân tích P7): H1 Δ = −0.063 / −0.064 / −0.033 (seed 0/1/2), không seed nào chứng minh được không thua kém; H2 không ủng hộ ở seed nào; H3 = +0.133 (ủng hộ) / +0.045 [−0.005; 0.093] (không) / +0.122 [0.054; 0.195] (ủng hộ), trung bình +0.100 ± 0.048. Kết luận chính vẫn theo seed 0 như đề cương; bài báo cáo đầy đủ cả ba seed, ghi rõ H3 lặp lại ở 2/3 seed.
- **D6 — 2026-11-16 07:20, sau phản biện thử vòng 1 (paper/revision/mock_review_20261116.md), TRƯỚC khi có kết quả các thí nghiệm dưới đây. Tất cả là THĂM DÒ, không phải tiêu chí chính.**
  (a) Phát hiện: AUROC trên dự đoán gộp 5 fold bị kéo thấp do mỗi fold có temperature/thang xác suất riêng (FP32 gộp 0.762 so với trong fold 0.879). Bổ sung phân tích độ nhạy dùng AUROC trong từng fold (3 seed × 5 fold). Kết quả này đã tính và ghi trong mock_review_20261116.md: H3 trong fold +0.038 [−0.004; 0.080] (không ý nghĩa); PTQ − FP32 −0.066 [−0.095; −0.036]; QAT − FP32 +0.003. Bài sẽ báo cáo CẢ hai cách tính và nói rõ H3 không vững.
  (b) Thí nghiệm bổ sung (script scripts/p56x_quant_extra.py md5 e9da14eeb1a0f073a62f3a98cb525cbb, seed 0, raw): E1 PTQ hiệu chuẩn 2000 cửa sổ (ptq_c2k); E2 đối chứng FP32 fine-tune cùng lịch QAT không fake-quant (fp32_ft, ptq_ft); E3 kiến trúc CNN thường (--arch cnn); E4 PD vs DD với DS-TCN. Kernel: p56x1 (PD vs HC, DS-TCN), p56x2 (PD vs DD), p56x3 (PD vs HC, CNN).
- **D7 — 2026-11-16 08:50 (thăm dò, trước khi có kết quả).** Đo đầu-cuối trên bo (B2) cho thấy firmware dùng lọc nhân quả, còn huấn luyện dùng lọc zero-phase. tools/causal_eval.py trên fold 0 (QAT): AUROC 0.890 (zero-phase) → 0.816 (nhân quả, chạy liên tục) / 0.833 (nhân quả, reset mỗi cửa sổ); 82 % quyết định trùng. Bổ sung kernel p56x4: cùng thiết kế seed 0 PD vs HC raw DS-TCN nhưng huấn luyện và đánh giá với lọc nhân quả (--causal; script md5 2c811cb060ff87ea1f476bc63dfe77c0), để đo chi phí của việc khớp tiền xử lý với thiết bị và cung cấp mô hình đúng cho bo.
- **D8 — 2026-11-16 11:00.** Sau phản biện thử vòng 2: mọi số trong fold tính trên 15 đơn vị fold–seed (seed 0–2) với corrected resampled t (Nadeau–Bengio 2003, tỉ lệ test/train 0.25) và TOST ±0.02 cho "không khác"; lần chạy lặp seed 0 (p56x1) chỉ dùng ở mục thăm dò. p56x4 (lọc nhân quả): FP32 0.871 vs 0.879 (zero-phase), QAT 0.869, PTQ −0.034 (một fold −0.179).
- **D10 — 2026-11-16 14:00, phát hiện khi trả lời phản biện thử BSPC vòng 1.** Khi sửa stride depthwise (s,1)→(s,s) ở D3c, lệnh `BatchNormalization` sau DepthwiseConv2D bị đưa nhầm vào chú thích. Hệ quả: seed 0 tiền định (p56a, 22 737 tham số) có BN sau depthwise, còn seed 1, 2 (p56s1/s2), mọi lượt p56m, p56x1–4, p56c0–2, p56q dùng biến thể KHÔNG có BN sau depthwise (21 841 tham số). Phát hiện bằng so `params` trong fold_meta.jsonl. Xử lý: (i) báo cáo trung thực trong bài; (ii) khôi phục BN (scripts/p56x_quant_extra_r1.py, md5 cb26378cbd1fce20c6c9db9010be5f68) và chạy lại toàn bộ thí nghiệm ba seed với kiến trúc đúng: p56z0–2 (zero-phase, --extra --diag) và p56k0–2 (nhân quả, --extra --diag); (iii) seed 0 tiền định (p56a) vẫn là phân tích chính; seed 1, 2 cũ được thay bằng p56z1/z2 cho phân tích lặp lại, kết quả cũ được báo cáo ở phụ lục. Quyết định chạy lại không phụ thuộc vào kết quả.
- **D11 — 2026-11-16 14:00 (thăm dò).** Phân tích độ nhạy cho H1–H3: chuẩn hóa thứ hạng điểm trong từng fold trước khi gộp (pooled_ranknorm). Kết quả trên p56a/s1/s2: FP32 gộp 0.762/0.843/0.746 → chuẩn hóa 0.879/0.862/0.881 (= trung bình trong fold). H1: −0.088/−0.059/−0.050 (không thua kém không được chứng minh); H2 ≈ 0; H3: +0.058 (p thô 0.036) / +0.043 (0.042) / +0.013 (0.59) → không đạt sau Holm ở seed nào. Kết luận: H3 ở phân tích gộp gốc phụ thuộc cách gộp.
