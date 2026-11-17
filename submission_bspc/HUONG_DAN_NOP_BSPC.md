# Gói nộp BSPC – hướng dẫn và bảng kiểm

Bài: *Eight-bit quantization of a tiny wrist-inertial network for Parkinson's disease detection: calibration sensitivity, evaluation pitfalls and microcontroller validation*
Tạp chí: Biomedical Signal Processing and Control (Elsevier), loại bài Research article / Full length article.

## File trong thư mục này
| File | Tải lên mục | Ghi chú |
|---|---|---|
| `overleaf_bspc.zip` | — | Mở Overleaf → New Project → **Upload Project**, chọn compiler pdfLaTeX, Recompile. Đã thử biên dịch lại từ zip: 30 trang, không lỗi, không thiếu trích dẫn |
| `manuscript.pdf` | Manuscript | Nếu hệ thống đòi file nguồn, tải `main.tex`, `refs.bib` và các `Figure_*.pdf` (LaTeX source files) |
| `Figure_1.pdf`, `Figure_2.pdf`, `Figure_3.pdf` | Figure | Dạng vector |
| `highlights.docx` | Highlights | 5 ý, 69–82 ký tự |
| `supplementary_material.pdf` | Supplementary material | Bảng S1, mục S2–S3 |
| `cover_letter.docx` (và .pdf) | Cover letter | **Điền ngày nộp** ở dòng đầu |

## Anh cần tự làm khi nộp
1. **Declaration of interest:** tạo bằng Elsevier declarations tool, chọn "no competing interests", rồi tải file Word đó lên. Nội dung phải khớp câu trong bài.
2. **Xác nhận CRediT** với thầy Việt (mục "CRediT authorship contribution statement" trong bài).
3. **Tác giả liên hệ:** thầy Ngô Bá Việt (vietnb@hcmute.edu.vn). Nếu anh là người bấm nộp, nhớ chọn thầy là Corresponding author ở bước Authors và nhập ORCID cho cả hai người.
4. **Mở Guide for Authors** của BSPC bằng trình duyệt của anh một lần để xác nhận các mục đánh dấu (?) trong `COMPLIANCE_BSPC.md`: số từ của full paper, số bảng/hình, giới hạn từ khóa (bài đang có 7). Trang này chặn công cụ tự động nên em không đọc trực tiếp được.
5. **Đọc lại toàn bài** trước khi nộp. Bản khai báo AI trong bài nói rằng tác giả đã đọc, chỉnh sửa và chịu trách nhiệm toàn bộ nội dung.
6. **Người phản biện đề xuất** (nếu hệ thống hỏi): chỉ đề xuất người không cùng cơ quan và không có xung đột lợi ích.

## Code và dữ liệu
Repo GitHub công khai (từ 17/11/2026): https://github.com/2612108/parkinson-int8-wearable — code, kết quả từng fold, firmware, đề cương có ngày. Bài và cover letter đã ghi link này. Khi bài được nhận, nên tạo bản Zenodo có DOI.

## Trạng thái phản biện mô phỏng
Bảy vòng phản biện mù độc lập đều cho Major revision; điểm vòng 7 là 3/3/2/3. Những yêu cầu chưa làm có thể gặp lại ở phản biện thật:
- kiểm lại cách cắt trên kiến trúc khác và seed mới;
- đối chiếu bộ ước lượng MSE/KL với bản tham chiếu;
- phân tích cửa sổ nào tạo dải cực trị;
- AUROC một cổ tay / một bài;
- công khai code trong lúc phản biện.

Nên chuẩn bị sẵn cho vòng sửa bài thật.
