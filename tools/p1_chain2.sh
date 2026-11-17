#!/usr/bin/env bash
# Chuỗi 2 (15:20): dừng P3 bản cũ → tải Monipar → cập nhật Kaggle Dataset (thêm Monipar) → audit lại → P3 bản mới.
#   setsid nohup bash tools/p1_chain2.sh > reports/p1/chain2.log 2>&1 < /dev/null &
cd "$HOME/Desktop/PARKINSON" || exit 1
S=reports/p1/STATUS.txt
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$S"; }
PY=.venv/bin/python

log "chain2: dừng P3 cũ (loader chưa nội suy theo Time)"
pkill -f "p3_basel[i]ne.py" ; sleep 2
[ -d reports/p3 ] && mv reports/p3 "reports/p3_old_$(date +%H%M)"

log "Monipar: cài requests/tqdm vào .venv rồi tải"
uv pip install --python "$PY" requests tqdm >/dev/null 2>&1 || "$PY" -m pip install -q requests tqdm
"$PY" - <<'PY' >> reports/p1/download_monipar.log 2>&1
import hashlib, json, pathlib, requests
out = pathlib.Path("data/raw/monipar"); out.mkdir(parents=True, exist_ok=True)
rec = requests.get("https://zenodo.org/api/records/8104853", timeout=60).json()
(out / "_zenodo_record.json").write_text(json.dumps(rec, indent=2))
for f in rec["files"]:
    name, url, want = f["key"], f["links"]["self"], f.get("checksum", "")
    dst = out / name
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    ok = (not want.startswith("md5:")) or hashlib.md5(dst.read_bytes()).hexdigest() == want[4:]
    print(name, dst.stat().st_size, "md5 OK" if ok else "md5 SAI")
PY
(cd data/raw/monipar && for z in *.zip; do [ -e "$z" ] && unzip -n -q "$z" -d "${z%.zip}"; done)
log "Monipar xong: $(du -sh data/raw/monipar | cut -f1), $(find data/raw/monipar -type f | wc -l) file"

log "audit lại (có Monipar)"
"$PY" scripts/02_audit.py > reports/p1/audit_stdout.log 2>&1
cp reports/audit_pads.md reports/audit_monipar.md reports/p1/ 2>/dev/null

log "Kaggle Dataset: thêm phiên bản có Monipar"
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
.venv/bin/kaggle datasets version -p data/raw --dir-mode zip -m "add Monipar" >> reports/p1/dataset_create.log 2>&1
log "dataset version rc=$?: $(tail -1 reports/p1/dataset_create.log)"
unset KAGGLE_API_TOKEN

log "P3 bản mới (3 seed)"
mkdir -p reports/p3
"$PY" scripts/p3_baseline.py --pads data/raw/pads --out reports/p3 --seeds 0 1 2 > reports/p3_run.log 2>&1
log "P3 xong rc=$?"
log "HẾT CHUỖI 2"
