# paper/ — bản thảo JBHI

- `main.tex`: khung bài theo ngân sách 8 trang (mục 6b kế hoạch). `\TBD{}` (đỏ) = chờ số liệu thật; `\NOTE{}` (xanh) = ghi chú nội bộ. Bản nộp phải không còn `\TBD`/`\NOTE`.
- `refs_doi.txt` → `python paper/fetch_bib.py` → `refs.bib` (BibTeX lấy từ DOI). `refs.bib` hiện tại chỉ là bản tạm để biên dịch.
- Biên dịch: `cd paper && latexmk -pdf main.tex`.

## Trước khi nộp (P10)
1. Tải gói template JBHI chính thức từ IEEE Template Selector (template-selector.ieee.org → Transactions/Journals → IEEE Journal of Biomedical and Health Informatics → LaTeX). Nếu gói dùng `ieeecolor` + style riêng JBHI thì đổi dòng `\documentclass` theo gói đó; nội dung giữ nguyên.
2. Đếm trang ≤ 8 (bài chính), tổng ≤ 14 kể cả phụ lục.
3. Abstract ≤ 250 từ, không viết tắt; kiểm `COMPLIANCE.md` (reports/p2) từng mục.
4. Tác giả: Quang-Tuyen Lai (thứ nhất), Ba-Viet Ngo (liên hệ, vietnb@hcmute.edu.vn, ORCID 0000-0002-8379-957X) — đổi 16/11 theo anh Tuyến.
