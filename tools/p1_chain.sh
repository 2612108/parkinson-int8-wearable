#!/usr/bin/env bash
# Chuỗi P1→P3→Dataset chạy nền trên máy anh Tuyến:  nohup bash tools/p1_chain.sh > reports/p1/chain.log 2>&1 &
cd "$HOME/Desktop/PARKINSON" || exit 1
S=reports/p1/STATUS.txt; mkdir -p reports/p1 reports/p3
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$S"; }

log "C: tải PADS + Monipar qua S3/Zenodo"
bash scripts/01_download_s3.sh >> reports/p1/download_s3.log 2>&1; rc=$?
NP=$(ls data/raw/pads/patients/patient_*.json 2>/dev/null | wc -l)
NT=$(ls data/raw/pads/movement/timeseries/*.txt 2>/dev/null | wc -l)
log "C xong rc=$rc patients=$NP timeseries=$NT; $(du -sh data/raw/pads data/raw/monipar | tr '\n' ' ')"

log "D: audit"
.venv/bin/python scripts/02_audit.py > reports/p1/audit_stdout.log 2>&1; rc=$?
cp reports/audit_pads.md reports/audit_monipar.md reports/pads_patients.csv reports/pads_timeseries_index.csv reports/p1/ 2>/dev/null
log "D xong rc=$rc"

log "F: tạo Kaggle Dataset riêng tư (trước P3 để GPU P4 chạy song song)"
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
cat > data/raw/dataset-metadata.json <<'EOF'
{"title":"parkinson-pads-monipar-raw","id":"nguynthqunhlu/parkinson-pads-monipar-raw","licenses":[{"name":"other"}],"isPrivate":true}
EOF
.venv/bin/kaggle datasets create -p data/raw --dir-mode zip >> reports/p1/dataset_create.log 2>&1; rc=$?
log "F xong rc=$rc: $(tail -2 reports/p1/dataset_create.log | tr '\n' ' ')"
unset KAGGLE_API_TOKEN

log "E: P3 baseline (3 seed)"
.venv/bin/python scripts/p3_baseline.py --pads data/raw/pads --out reports/p3 --seeds 0 1 2 > reports/p3_run.log 2>&1; rc=$?
log "E xong rc=$rc"
log "HẾT CHUỖI"
