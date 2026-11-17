#!/usr/bin/env bash
# R1 (phản biện BSPC vòng 1): đẩy 6 kernel p56z0-2 (zero-phase) + p56k0-2 (nhân quả), BN sau depthwise đã khôi phục (D10).
cd ~/Desktop/PARKINSON || exit 1
for n in p56z0 p56z1 p56z2 p56k0 p56k1 p56k2; do
  echo "== $n $(date -Iseconds)"; kaggle kernels push -p kaggle/$n 2>&1 | tail -3
done
