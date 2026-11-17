"""Phiên 1: audit PADS và Monipar. Không giả định cứng cấu trúc thư mục: tự dò, ghi lại những gì thấy.

Kết quả: reports/audit_pads.md, reports/audit_monipar.md
"""
import collections, json, pathlib, re, html
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REP = ROOT / "reports"; REP.mkdir(exist_ok=True)


def inventory(base: pathlib.Path, md: list):
    files = [p for p in base.rglob("*") if p.is_file()]
    by_ext = collections.Counter(p.suffix.lower() or "(none)" for p in files)
    size = sum(p.stat().st_size for p in files)
    md += [f"- Tổng file: {len(files)}, dung lượng {size/1e6:.1f} MB",
           "- Theo phần mở rộng: " + ", ".join(f"`{k}`: {v}" for k, v in by_ext.most_common())]
    dirs = collections.Counter(str(p.parent.relative_to(base)) for p in files)
    md.append("- Thư mục (top 15 theo số file):")
    md += [f"  - `{d}`: {n}" for d, n in dirs.most_common(15)]
    return files


def read_numeric(path: pathlib.Path):
    """Đọc file bảng số, tự đoán dấu phân cách và header."""
    for sep in [",", ";", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", nrows=200000)
            if df.shape[1] >= 2:
                if not np.issubdtype(df.dtypes.iloc[0], np.number):
                    pass
                return df, sep
        except Exception:
            continue
    return None, None


def fs_from_time(t: np.ndarray):
    d = np.diff(t[np.isfinite(t)])
    if len(d) == 0 or np.mean(d > 0) < 0.95:  # không đơn điệu tăng → không phải cột thời gian
        return None
    d = d[d > 0]
    if len(d) == 0:
        return None
    med = np.median(d)
    # đơn vị: s, ms hoặc ns
    for scale, unit in [(1, "s"), (1e-3, "ms"), (1e-9, "ns")]:
        fs = 1 / (med * scale)
        if 5 <= fs <= 2000:
            return fs, unit, float(np.std(d) / med)
    return None


# ---------------------------------------------------------------- PADS
def audit_pads():
    base = RAW / "pads"
    md = ["# Audit PADS", ""]
    if not base.exists():
        md.append("**Chưa có dữ liệu** — chạy scripts/01_download.sh"); return md
    md.append("## Kho file"); inventory(base, md)

    # bệnh nhân
    pj = sorted(base.rglob("patient_*.json"))
    md += ["", f"## Bệnh nhân ({len(pj)} file JSON)"]
    if pj:
        rows = [json.loads(p.read_text()) for p in pj]
        pdf = pd.json_normalize(rows)
        md.append(f"- Cột: {', '.join(map(str, pdf.columns))}")
        for c in ["condition", "gender", "handedness"]:
            if c in pdf:
                md.append(f"- `{c}`: " + ", ".join(f"{k}={v}" for k, v in pdf[c].value_counts(dropna=False).items()))
        for c in ["age", "age_at_diagnosis", "height", "weight"]:
            if c in pdf:
                s = pd.to_numeric(pdf[c], errors="coerce")
                md.append(f"- `{c}`: mean {s.mean():.1f} ± {s.std():.1f}, thiếu {s.isna().sum()}")
        pdf.to_csv(REP / "pads_patients.csv", index=False)
        md.append("- Đã ghi `reports/pads_patients.csv`")

    # file_list.csv của bản preprocessed (nếu có)
    for fl in base.rglob("file_list.csv"):
        df = pd.read_csv(fl)
        md += ["", f"## `{fl.relative_to(base)}`", f"- shape {df.shape}; cột: {', '.join(df.columns)}",
               "```", df.head(5).to_string(), "```"]
        for c in df.columns:
            if df[c].nunique() <= 10:
                md.append(f"- `{c}`: " + ", ".join(f"{k}={v}" for k, v in df[c].value_counts().items()))

    # chuỗi thời gian: tên dạng {id}_{Task}_{Wrist}.txt
    ts = [p for p in base.rglob("*.txt") if re.match(r"^\d+_.+_(Left|Right)Wrist", p.stem)]
    md += ["", f"## Chuỗi thời gian thô ({len(ts)} file)"]
    if ts:
        meta = pd.DataFrame([dict(zip(["subject", "task", "wrist"], p.stem.split("_", 2)), path=str(p)) for p in ts])
        meta["subject"] = meta["subject"].astype(int)
        md.append(f"- Số người có tín hiệu: {meta.subject.nunique()}")
        md.append("- Bài tập: " + ", ".join(f"{k} ({v})" for k, v in meta.task.value_counts().items()))
        md.append("- Cổ tay: " + ", ".join(f"{k}={v}" for k, v in meta.wrist.value_counts().items()))
        per = meta.groupby("subject").size()
        md.append(f"- File/người: min {per.min()}, median {per.median():.0f}, max {per.max()}")
        # mẫu 30 file để đo tần số, số cột, độ dài, biên độ
        stats = []
        for p in meta.sample(min(30, len(meta)), random_state=0).path:
            df, sep = read_numeric(pathlib.Path(p))
            if df is None:
                continue
            arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
            fs = fs_from_time(arr[:, 0])
            stats.append(dict(file=pathlib.Path(p).name, n=len(arr), cols=arr.shape[1],
                              fs=None if fs is None else round(fs[0], 2),
                              t_unit=None if fs is None else fs[1],
                              jitter=None if fs is None else round(fs[2], 4),
                              acc_absmax=float(np.nanmax(np.abs(arr[:, 1:4]))) if arr.shape[1] >= 4 else None,
                              gyro_absmax=float(np.nanmax(np.abs(arr[:, 4:7]))) if arr.shape[1] >= 7 else None))
        if stats:
            sdf = pd.DataFrame(stats)
            md += ["", "Mẫu 30 file (cột 0 giả định là thời gian, 1–3 acc, 4–6 gyro — **cần anh xác nhận với readme của PADS**):",
                   "```", sdf.to_string(index=False), "```"]
            first = pathlib.Path(meta.path.iloc[0])
            md += ["", f"5 dòng đầu `{first.name}`:", "```", "".join(first.open().readlines()[:5]), "```"]
        meta.to_csv(REP / "pads_timeseries_index.csv", index=False)

    # observation JSON: mô tả bài tập
    obs = sorted(base.rglob("observation_*.json"))
    if obs:
        o = json.loads(obs[0].read_text())
        md += ["", f"## Observation JSON ({len(obs)} file) — ví dụ `{obs[0].name}`", "```json",
               json.dumps(o, indent=1, ensure_ascii=False)[:2500], "```"]
    readmes = [p for p in base.rglob("*") if p.name.lower().startswith("readme")]
    for r in readmes[:2]:
        md += ["", f"## `{r.relative_to(base)}` (2000 ký tự đầu)", "```", r.read_text(errors="ignore")[:2000], "```"]
    return md


# ---------------------------------------------------------------- Monipar
def audit_monipar():
    base = RAW / "monipar"
    md = ["# Audit Monipar", ""]
    if not base.exists():
        md.append("**Chưa có dữ liệu** — chạy scripts/01_download.sh"); return md
    rec = base / "_zenodo_record.json"
    if rec.exists():
        r = json.loads(rec.read_text())
        m = r.get("metadata", {})
        desc = re.sub(r"<[^>]+>", " ", html.unescape(m.get("description", "")))
        md += ["## Zenodo", f"- Tiêu đề: {m.get('title')}", f"- Ngày: {m.get('publication_date')}",
               f"- License: {m.get('license')}", f"- Phiên bản: {m.get('version')}",
               "- File: " + ", ".join(f"{f['key']} ({f.get('size',0)/1e6:.1f} MB)" for f in r.get("files", [])),
               "", "Mô tả:", "```", re.sub(r"\s+", " ", desc)[:3000], "```", ""]
    md.append("## Kho file")
    files = inventory(base, md)
    tab = [p for p in files if p.suffix.lower() in {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json", ".mat"}
           and p.name != "_zenodo_record.json"]
    md += ["", f"## Danh sách file dữ liệu ({len(tab)}), 80 file đầu", "```"]
    md += [str(p.relative_to(base)) for p in tab[:80]]
    md.append("```")
    # gợi ý tên bài tập từ tên file
    tokens = collections.Counter(t.lower() for p in tab for t in re.split(r"[_\-\s\.]+", p.stem) if t and not t.isdigit())
    md.append("- Token hay gặp trong tên file: " + ", ".join(f"{k}({v})" for k, v in tokens.most_common(40)))
    # mẫu nội dung
    md += ["", "## Mẫu nội dung (10 file)"]
    for p in tab[:10]:
        md.append(f"### `{p.relative_to(base)}`")
        try:
            if p.suffix.lower() in {".xlsx", ".xls"}:
                x = pd.read_excel(p, sheet_name=None, nrows=5)
                for s, d in x.items():
                    md += [f"sheet `{s}` shape≥{d.shape}", "```", d.head().to_string(), "```"]
                continue
            if p.suffix.lower() == ".json":
                md += ["```json", p.read_text(errors="ignore")[:1200], "```"]; continue
            if p.suffix.lower() == ".mat":
                md.append("(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)"); continue
            df, sep = read_numeric(p)
            if df is None:
                md += ["```", "".join(p.open(errors="ignore").readlines()[:5]), "```"]; continue
            md.append(f"- sep `{sep}`, shape {df.shape}, cột: {list(df.columns)[:12]}")
            num = df.apply(pd.to_numeric, errors="coerce")
            for c in num.columns[:2]:
                fs = fs_from_time(num[c].to_numpy(float))
                if fs:
                    md.append(f"- cột `{c}` giống thời gian: fs ≈ {fs[0]:.2f} Hz (đơn vị {fs[1]}, jitter {fs[2]:.3f})")
            md += ["```", df.head(5).to_string(), "```"]
        except Exception as e:
            md.append(f"lỗi đọc: {e}")
    return md


if __name__ == "__main__":
    for name, fn in [("audit_pads.md", audit_pads), ("audit_monipar.md", audit_monipar)]:
        txt = "\n".join(fn()) + "\n"
        (REP / name).write_text(txt)
        print(f"== {name} ==\n{txt[:1500]}\n...")
