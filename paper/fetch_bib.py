"""Sinh refs.bib từ refs_doi.txt qua doi.org (content negotiation, Crossref/DataCite).
Chạy trên máy có Internet:  python paper/fetch_bib.py
- Đổi khóa BibTeX thành khóa trong refs_doi.txt.
- Bảo vệ chữ hoa trong title bằng {} (IEEE style hạ chữ thường).
- DOI nào lỗi → in ra, KHÔNG tự bịa entry. PADSdata: nếu DOI PhysioNet sai, sửa theo trang dataset.
"""
import pathlib, re, sys, urllib.request

HERE = pathlib.Path(__file__).parent
PROTECT = ["Parkinson", "Parkinsonian", "TinyML", "MCU", "IoT", "LSTM", "Bi-LSTM", "HAR", "INT8", "ESP32",
           "PADS", "MIL", "CNN", "Transformer", "Smartwatch", "MDS-UPDRS", "Levodopa", "Internet of Medical Things"]


def fetch(doi):
    req = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def fix(bib, key):
    bib = re.sub(r"@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)

    def prot(m):
        t = m.group(2)
        for w in sorted(PROTECT, key=len, reverse=True):
            t = re.sub(rf"(?<![{{\w]){re.escape(w)}(?![\w}}])", "{" + w + "}", t)
        return m.group(1) + t + m.group(3)
    return re.sub(r"(title\s*=\s*\{)(.*?)(\},?\s*\n?)", prot, bib, count=1, flags=re.S)


out, bad = [], []
for line in (HERE / "refs_doi.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    key, doi = line.split()[:2]
    try:
        out.append(fix(fetch(doi).strip(), key))
        print("ok ", key, doi)
    except Exception as e:
        bad.append((key, doi, str(e)))
        print("LỖI", key, doi, e)
(HERE / "refs.bib").write_text("\n\n".join(out) + "\n")
if bad:
    (HERE / "refs_LOI.txt").write_text("\n".join(f"{k}\t{d}\t{e}" for k, d, e in bad))
    sys.exit(1)
