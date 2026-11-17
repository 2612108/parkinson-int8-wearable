#!/usr/bin/env bash
# Phiên 1 (v2): tải PADS từ bucket S3 công khai của PhysioNet (AWS Open Data, không cần tài khoản)
# và Monipar (Zenodo 8104853). Chạy lại được.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw/pads" "$ROOT/data/raw/monipar" "$ROOT/reports"
SECONDS=0

echo "== PADS (S3 physionet-open) =="
pip -q install awscli >/dev/null 2>&1 || true
AWS="python -m awscli"
BASE="s3://physionet-open/parkinsons-disease-smartwatch"
$AWS s3 ls --no-sign-request "$BASE/" | head
# bucket có thể chứa thư mục phiên bản 1.0.0/ hoặc dữ liệu ngay ở gốc
if $AWS s3 ls --no-sign-request "$BASE/1.0.0/" >/dev/null 2>&1; then SRC="$BASE/1.0.0/"; else SRC="$BASE/"; fi
echo "nguồn: $SRC"
$AWS s3 sync --no-sign-request --only-show-errors "$SRC" "$ROOT/data/raw/pads/"
echo "PADS xong sau ${SECONDS}s, $(find "$ROOT/data/raw/pads" -type f | wc -l) file"
if [ -f "$ROOT/data/raw/pads/SHA256SUMS.txt" ]; then
  (cd "$ROOT/data/raw/pads" && sha256sum -c SHA256SUMS.txt --quiet > "$ROOT/reports/pads_sha256.log" 2>&1) \
    && echo "PADS checksum OK" || echo "PADS checksum CÓ LỖI, xem reports/pads_sha256.log"
fi

echo "== Monipar =="
cd "$ROOT"
PY_BIN="$ROOT/.venv/bin/python"; [ -x "$PY_BIN" ] || PY_BIN=python3
"$PY_BIN" - <<'PY'
import hashlib, json, pathlib, requests
from tqdm import tqdm
out = pathlib.Path("data/raw/monipar"); out.mkdir(parents=True, exist_ok=True)
rec = requests.get("https://zenodo.org/api/records/8104853", timeout=60).json()
(out / "_zenodo_record.json").write_text(json.dumps(rec, indent=2))
for f in rec["files"]:
    name, url = f["key"], f["links"]["self"]
    dst = out / name
    want = f.get("checksum", "")
    if dst.exists() and want.startswith("md5:") and hashlib.md5(dst.read_bytes()).hexdigest() == want[4:]:
        print("đã có:", name); continue
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh, tqdm(total=f.get("size"), unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk); bar.update(len(chunk))
    if want.startswith("md5:"):
        print(name, "md5 OK" if hashlib.md5(dst.read_bytes()).hexdigest() == want[4:] else "md5 SAI")
PY
cd "$ROOT/data/raw/monipar"
for z in *.zip; do [ -e "$z" ] && unzip -n -q "$z" -d "${z%.zip}"; done || true
echo "Tổng thời gian tải: ${SECONDS}s"
