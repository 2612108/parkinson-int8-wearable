"""Đóng gói src/*.py vào một notebook Kaggle: mỗi module một ô %%writefile, rồi các lệnh chạy.
Dùng: python kaggle/make_nb.py --out kaggle/p3/p3.ipynb --mods features evalkit pads_io p3_baseline --run "python p3_baseline.py --out /kaggle/working"
"""
import argparse, json, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--mods", nargs="+"); ap.add_argument("--run", nargs="+"); ap.add_argument("--pre", default="")
a = ap.parse_args()
src = pathlib.Path(__file__).resolve().parents[1] / "src"
cells = [dict(cell_type="markdown", metadata={}, source=[f"# {pathlib.Path(a.out).stem} — sinh tự động từ src/ bởi kaggle/make_nb.py"])]
code = lambda s: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.splitlines(keepends=True))
cells.append(code("%cd /kaggle/working\n!mkdir -p src reports\n" + a.pre))
for m in a.mods:
    cells.append(code(f"%%writefile src/{m}.py\n" + (src / f"{m}.py").read_text()))
for r in a.run:
    cells.append(code(f"!cd src && {r} 2>&1 | tee -a ../reports/run.log"))
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"), language_info=dict(name="python")), nbformat=4, nbformat_minor=4)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(a.out, "w"), indent=1, ensure_ascii=False)
print("wrote", a.out, len(cells), "cells")
