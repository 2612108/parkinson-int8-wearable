# PARKINSON — Quantization under device shift (JBHI)

Code cho bài báo theo kế hoạch v2 (`KE_HOACH_NGHIEN_CUU_Q1_v2.md`). Làm theo phiên; nhật ký ở `NHAT_KY_PHIEN.md`.

## Cấu trúc
```
data/raw/pads/        PADS (PhysioNet)          -- không đưa lên git
data/raw/monipar/     Monipar (Zenodo 8104853)  -- không đưa lên git
scripts/              script chạy theo phiên
src/                  thư viện dùng chung
reports/              kết quả, bảng audit, log gửi lại cho Claude
```

## Phiên 1 — tải dữ liệu và audit
```bash
cd ~/Desktop/PARKINSON
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/01_download.sh          # ~1 GB, có thể chạy lại, tự nối tiếp
python scripts/02_audit.py           # ghi reports/audit_pads.md, reports/audit_monipar.md
python scripts/00_env_report.py      # ghi reports/env.md (CPU/GPU/RAM, phiên bản thư viện)
```
Gửi lại cho Claude 3 file trong `reports/`.
