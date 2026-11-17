"""Đóng gói một script của dự án thành notebook Kaggle (nhúng src/ + script, gắn dataset, GPU tuỳ chọn).

python tools/make_kaggle_nb.py --session p4 --script scripts/p4_dl.py --args "--pads /kaggle/input --out reports/p4" \
    --dataset nguynthqunhlu/parkinson-pads-monipar-raw --gpu --user nguynthqunhlu
→ kaggle/p4/{p4.ipynb, kernel-metadata.json}. Đẩy: kaggle kernels push -p kaggle/p4
Notebook tự giải nén mọi .zip trong /kaggle/input (nếu Kaggle chưa giải nén) vào /kaggle/working/input.
"""
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cell(src, md=False):
    if md:
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--script", required=True)
    ap.add_argument("--args", default=""); ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--kernel_source", action="append", default=[])
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--user", required=True)
    ap.add_argument("--pip", default="xgboost joblib")
    a = ap.parse_args()
    s = a.session; out = ROOT / "kaggle" / s; out.mkdir(parents=True, exist_ok=True)
    cells = [cell(f"# {s.upper()} — `{a.script}`\nSinh tự động bởi tools/make_kaggle_nb.py. Kết quả ở /kaggle/working/reports/.", md=True),
             cell(f"%cd /kaggle/working\n!mkdir -p src scripts reports\n!pip -q install {a.pip}\n!nvidia-smi || true")]
    for f in sorted((ROOT / "src").glob("*.py")):
        body = f.read_text()
        if not body.strip():          # %%writefile không chấp nhận thân rỗng
            body = "# (trống)\n"
        cells.append(cell(f"%%writefile src/{f.name}\n" + body))
    sp = ROOT / a.script
    cells.append(cell(f"%%writefile {a.script}\n" + sp.read_text()))
    cells.append(cell(
        "import pathlib, zipfile\n"
        "for z in pathlib.Path('/kaggle/input').rglob('*.zip'):\n"
        "    dst = pathlib.Path('/kaggle/working/input') / z.stem\n"
        "    if not dst.exists():\n"
        "        print('giải nén', z); zipfile.ZipFile(z).extractall(dst)\n"
        "!ls /kaggle/input /kaggle/working/input 2>/dev/null | head -30"))
    args = a.args.replace("--pads /kaggle/input", "--pads /kaggle/input")
    cells.append(cell(f"!python {a.script} {args} 2>&1 | tee reports/{s}_run.log | grep -vE 'oneDNN|cpu_feature' | tail -n 80\n"
                      f"# nếu dữ liệu nằm trong file zip đã giải nén thì chạy lại với /kaggle/working/input"))
    cells.append(cell(f"import os\nif not os.path.exists('reports/{s}/run_info.json'):  # script không tìm thấy dữ liệu → thử thư mục đã giải nén\n"
                      f"    !python {a.script} {args.replace('/kaggle/input', '/kaggle/working/input')} 2>&1 | tee reports/{s}_run2.log | tail -n 80"))
    cells.append(cell("!ls -laR reports | head -60"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / f"{s}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = {"id": f"{a.user}/parkinson-{s}", "title": f"parkinson-{s}", "code_file": f"{s}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": bool(a.gpu), "enable_tpu": False,
            "enable_internet": True, "dataset_sources": a.dataset, "competition_sources": [],
            "kernel_sources": a.kernel_source}
    if a.gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(out, json.dumps(meta))


if __name__ == "__main__":
    main()
