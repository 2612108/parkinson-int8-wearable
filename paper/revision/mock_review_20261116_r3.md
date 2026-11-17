# Phản biện thử vòng 3 — 16/11/2026 (tác tử mới)
**Khuyến nghị: Reject (khuyến khích làm lại đáng kể).** Ba vòng liên tiếp đều Reject cho JBHI.

| # | Nhận xét | Kiểm lại | Xử lý |
|---|---|---|---|
| M1 | Phân tích xác nhận (AUROC gộp) bị chính tác giả thừa nhận là sai thiết kế → kết luận dựa trên hậu kiểm; đề cương không công khai | Đúng | Chỉ sửa được bằng một nghiên cứu xác nhận MỚI (seed mới, chỉ số trong fold định trước, đề cương công khai OSF) |
| M2 | 16×8 và 2000 cửa sổ chỉ chạy ở bối cảnh nhân quả, nơi mất mát PTQ không có ý nghĩa; QAT "reliable" khi chưa chứng minh tương đương; ngưỡng ±0.02 chưa biện minh | Đúng | Sửa câu Discussion ("in the causal runs"); cần chạy 16×8 + CLE/bias correction/AdaRound trong bối cảnh zero-phase |
| M3 | Tính mới hạn chế (PTQ hỏng ở depthwise, QAT sửa — đã biết; gộp điểm CV đã biết) | Đúng | Không sửa được bằng văn bản |
| M4 | RF thắng và cũng chạy được trên bo; so sánh lệch (seed, cách gộp, hai bản cài đặt) | Đúng | Cần làm lại benchmark thống nhất + RF một cổ tay/một bài |
| M5 | Nhiều lựa chọn sau khi xem kết quả trên cùng fold | Đúng | Đã ghi là thăm dò; cần xác nhận trên seed mới |
| M6 | Thiếu bảng nhân khẩu, chưa phân tầng mạng theo tuổi/giới | Đúng | Cần bổ sung |
| M7 | Mâu thuẫn nội bộ (seed, latency 14.6 vs 12.6, Brier, footnote b, "one recording") | Đúng | ĐÃ SỬA |
| M8 | Monipar | Đúng | Đã ghi (✓)ᵃ mô tả |
| M9 | Mã/firmware/trọng số chưa công bố | Ghi nhận | Theo quyết định của anh Tuyến |

Kết luận: các lỗi trình bày đã sửa; các lỗi còn lại cần một nghiên cứu xác nhận mới và vẫn vướng tính mới ở mức JBHI.
