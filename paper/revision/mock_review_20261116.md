# Phản biện thử vòng 1 — 16/11/2026 (tác tử độc lập, general-purpose + định nghĩa paper-reviewer)

Đầu vào cho tác tử: tên tạp chí (IEEE JBHI, regular paper), manuscript.tex (đã bỏ chú thích), PDF + ảnh trang, ghi chú hướng dẫn tác giả JBHI. Không kể quá trình làm.

**Khuyến nghị: Reject (khuyến khích nộp lại như bài mới).**

## Lỗi lớn và kết quả kiểm lại

| # | Nhận xét | Kiểm lại | Đúng/sai |
|---|---|---|---|
| M1 | FP32 trong Bảng IV (0.762) mâu thuẫn DS-TCN Bảng III (0.856) | Đúng. Hai số khác cách tính: Bảng III = P4 (mã Conv1D, AUROC gộp oof); Bảng IV = P56 AUROC gộp 5 fold sau temperature scaling riêng từng fold. AUROC trong từng fold của FP32 P56 là 0.879 (seed 0). Gộp fold làm lệch thang xác suất giữa các fold → AUROC gộp thấp giả tạo. | **Đúng — nghiêm trọng** |
| M2 | QAT lẫn với huấn luyện thêm (QAT 0.857 > FP32 0.762) | Theo AUROC trong fold: QAT − FP32 = +0.003 (KTC −0.011; +0.017), không vượt cha. "QAT hơn FP32" là ảo ảnh do gộp fold. Nhận xét về nhiễu "fine-tune thêm" vẫn hợp lệ về thiết kế. | Đúng một phần |
| M3 | PTQ yếu (300 cửa sổ hiệu chuẩn) | Đúng, chưa thử cỡ/phương pháp hiệu chuẩn | Đúng |
| M4 | RF và mạng khác không được lượng tử/dịch chuyển/nạp bo; PD vs DD không xét lượng tử | Đúng | Đúng |
| M5 | Abstract/Conclusion vượt kết quả chính | Đúng ("not safe" dựa trên H1 không kết luận; "QAT removes the loss" là hậu kiểm) | Đúng |
| M6 | p của H1 sai kiểu kiểm định; 50 Hz pha loãng; bỏ qua phương sai seed | p H1 = 2·min(tỉ lệ ≤ −0.02, ≥ −0.02) là kiểm định hai phía quanh ngưỡng, không phải p non-inferiority một phía → sửa. 13 dịch chuyển: H3 không đổi (+0.038 vs +0.038 trong fold). | Đúng |
| M7 | Hứa sens/spec/Brier nhưng không báo; RQ không định nghĩa; đề cương không có dấu thời gian | Đúng | Đúng |
| M8 | Chọn mean pooling dựa trên kết quả outer fold; Monipar nhiễu bối cảnh | Đúng (ghi là lựa chọn thiết kế hậu kiểm); trọng lực Monipar bị lọc bằng band-pass 0.5 Hz (chưa ghi trong bài) | Đúng |
| M9 | Latency chưa gồm tiền xử lý; không đo năng lượng | Đúng | Đúng |
| M10 | Tính mới hạn chế: QAT > PTQ đã biết | Hợp lý | Đúng |
| M11 | Mã "theo yêu cầu" chưa đủ | Theo quyết định của anh Tuyến; JBHI không bắt buộc | Ghi nhận |

## Phân tích bổ sung sau phản biện (AUROC trong từng fold, 3 seed × 5 fold = 15 đơn vị, KTC t)
| Đối chiếu | Trung bình [KTC 95%] | Theo seed 0/1/2 |
|---|---|---|
| PTQ − FP32, sạch | −0.066 [−0.095; −0.036] | −0.087 / −0.060 / −0.050 |
| Suy giảm thêm do lượng tử (H2) | +0.001 [−0.008; +0.010] | ≈ 0 |
| QAT+SR − PTQ, TB 14 dịch chuyển (H3) | +0.038 [−0.004; +0.080] | +0.057 / +0.043 / +0.013 |
| QAT − FP32, sạch | +0.003 [−0.011; +0.017] | |
| QAT − PTQ, TB dịch chuyển | +0.068 [+0.040; +0.095] | +0.089 / +0.066 / +0.048 |
| PTQ+SR − FP32+SR, sạch | −0.007 [−0.024; +0.010] | |
| QAT+SR − FP32, TB dịch chuyển | −0.029 [−0.052; −0.006] | |

Kết luận: mất mát do PTQ là chắc chắn; QAT khôi phục đúng mức FP32; ngẫu nhiên hóa cảm biến KHÔNG cải thiện trung bình (làm giảm AUROC sạch); H3 có ý nghĩa ở AUROC gộp chủ yếu vì cách gộp fold.
