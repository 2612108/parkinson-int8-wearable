# Phản biện thử vòng 2 — 16/11/2026 (tác tử mới, cùng định nghĩa paper-reviewer)
**Khuyến nghị: Reject (khuyến khích nộp lại).**

| # | Nhận xét chính | Kiểm lại | Đã sửa (bản 16/11 ~11:00) |
|---|---|---|---|
| M1 | Abstract/Conclusion dựa trên phân tích trong fold không có trong đề cương; dùng nó gạt H3 đã được ủng hộ ở seed 0 | Đúng | Abstract/Conclusion mở đầu bằng kết quả định trước (H1 không đạt mọi seed, H2 null, H3 đạt 2/3 seed); phân tích trong fold ghi rõ là thăm dò |
| M2 | "Under device shift" không được ủng hộ (H2 null) | Đúng | Bỏ khỏi tên bài; nói rõ mất mát xảy ra trên dữ liệu sạch, không tăng dưới dịch chuyển |
| M3 | Bảng giả thuyết bị lặp (tab:hyp hai lần), p mâu thuẫn | Đúng (bảng cũ còn sót) | Xóa bảng cũ |
| M4 | Trộn 15 và 20 đơn vị; số không khớp bảng | Đúng | Mọi số trong fold tính lại trên 15 đơn vị (seed 0–2); lần chạy lặp seed 0 chỉ ở mục thăm dò |
| M5 | "Matched" không có ngưỡng; CI t xem fold độc lập | Đúng | Dùng corrected resampled t (Nadeau–Bengio); TOST ±0.02: tương đương chưa được chứng minh (p = 0.12); viết "no detectable loss" |
| M6 | Tính mới; thiếu CLE/bias correction/AdaRound; PTQ của FP32 fine-tune chỉ mất 0.015 | Đúng | Ghi nhận trong Results + Limitations; chưa chạy các phương pháp PTQ nâng cao |
| M7 | RF chưa lượng tử/nạp bo; "screening"; chưa phân tầng tuổi/giới | Đúng | Đổi "screening" → "detection"; thêm vào Limitations; chưa làm thí nghiệm |
| M8 | Monipar không diễn giải được nhưng Bảng I đánh dấu Ext. ✓ | Đúng | Đổi thành (✓) "descriptive only" |
| M9 | Mã "theo yêu cầu"; đề cương chưa công bố | Ghi nhận | Theo quyết định của anh Tuyến (chia sẻ khi tạp chí yêu cầu); đề cương có thể nộp làm tài liệu bổ sung |
| M10 | Bit-exact ít cửa sổ; năng lượng 5–19 mJ quá rộng; nhân quả chỉ 1 seed | Đúng | Bỏ con số mJ; giữ dòng đo được; nhân quả 1 seed ghi trong Limitations |

Nhỏ: bỏ macro \TBD/\NOTE; sửa caption Bảng IV; Guo/Ilse → bản ICML; {IMU}; pages "--". Chưa: tách vài câu > 30 từ, phóng to Hình 2.

Đánh giá: các lỗi trình bày/thống kê đã sửa. Các điểm còn lại cần thí nghiệm mới (PTQ nâng cao, RF trên bo, nhân quả 3 seed) và quyết định đóng góp → xem đề xuất chọn tạp chí.
