"""Tính lại CI bootstrap theo bệnh nhân cho các file oof_*.csv (cột y, p, thr theo fold).
Dùng: python tools/recompute_ci.py reports/p3 reports/p4a reports/p4b reports/p4c reports/p4d
Ghi reports/<dir>/ci_fixed.csv. Không đụng results.json gốc."""
import sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evalkit import metrics, bootstrap_ci  # noqa: E402

for d in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "oof_*.csv"))):
        D = pd.read_csv(f); y, p, t = D.y.to_numpy(), D.p.to_numpy(), D.thr.to_numpy()
        m = metrics(y, p, t); ci = bootstrap_ci(y, p, thr=t, n_boot=2000)
        r = dict(file=os.path.basename(f), n=len(y))
        for k in ("bal_acc", "auroc", "sens", "spec", "ece"):
            r[k] = round(float(m[k]), 4); r[k + "_lo"] = round(ci[k][0], 4); r[k + "_hi"] = round(ci[k][1], 4)
            if not (r[k + "_lo"] - 1e-9 <= r[k] <= r[k + "_hi"] + 1e-9):
                print("CANH BAO: diem ngoai CI", f, k)
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(d, "ci_fixed.csv"), index=False)
        print(d); print(pd.DataFrame(rows)[["file", "bal_acc", "bal_acc_lo", "bal_acc_hi", "auroc", "auroc_lo", "auroc_hi", "sens", "sens_lo", "sens_hi"]].to_string(index=False))
