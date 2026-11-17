"""P7 — Phân tích P56 đúng PROTOCOL_P56.md (H1–H3 + tiêu chí phụ).
python scripts/p7_p56_stats.py reports/p56a/patient_preds_PD_vs_HC_raw.csv [--out reports/p7] [--B 2000]
Bootstrap cặp theo bệnh nhân, phân tầng theo nhãn, seed 0. AUROC trên dự đoán bệnh nhân gộp 5 fold (đúng đề cương)."""
import argparse, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata

S = ["fs50", "fs25", "jitter10", "jitter30", "adc4g12b", "adc2g10b", "adc2g8b", "noise_lo", "noise_hi",
     "rot15", "rot45", "rot180", "mirror", "nogyro"]
V = ["fp32", "ptq", "qat", "fp32_sr", "ptq_sr", "qat_sr"]


def auc_mat(y, P):
    """AUROC cho từng cột của P (n, k) bằng hạng (Mann–Whitney)."""
    n1 = y.sum(); n0 = len(y) - n1
    R = np.apply_along_axis(rankdata, 0, P)
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds"); ap.add_argument("--out", default="reports/p7")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--margin", type=float, default=-0.02)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(a.preds, index_col=0, dtype={0: str}); y = D.y.to_numpy().astype(int)
    cols = [f"{s}|{v}" for s in ["clean"] + S for v in V]; P = D[cols].to_numpy(float)
    ci = {c: k for k, c in enumerate(cols)}
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; rng = np.random.default_rng(0)
    A0 = auc_mat(y, P)
    AB = np.empty((a.B, len(cols)))
    for b in range(a.B):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        AB[b] = auc_mat(y[idx], P[idx])

    def g(A, s, v): return A[..., ci[f"{s}|{v}"]]
    def D_(A, v): return np.mean([g(A, "clean", v) - g(A, s, v) for s in S], 0)
    def Abar(A, v): return np.mean([g(A, s, v) for s in S], 0)
    def summ(pt, bs, null=0.0):
        lo, hi = np.percentile(bs, [2.5, 97.5]); p = min(1.0, 2 * min((bs <= null).mean(), (bs >= null).mean()))
        return dict(est=float(pt), lo=float(lo), hi=float(hi), p=float(max(p, 1 / a.B)))

    H = {"H1_dAUROC_clean_ptq_minus_fp32": summ(g(A0, "clean", "ptq") - g(A0, "clean", "fp32"),
                                                 g(AB, "clean", "ptq") - g(AB, "clean", "fp32"), a.margin),
         "H2_D_ptq_minus_D_fp32": summ(D_(A0, "ptq") - D_(A0, "fp32"), D_(AB, "ptq") - D_(AB, "fp32")),
         "H3_Abar_qatsr_minus_Abar_ptq": summ(Abar(A0, "qat_sr") - Abar(A0, "ptq"), Abar(AB, "qat_sr") - Abar(AB, "ptq"))}
    for k, h in zip(H, holm([h["p"] for h in H.values()])):
        H[k]["p_holm"] = float(h)
    H["H1_dAUROC_clean_ptq_minus_fp32"]["supported"] = H["H1_dAUROC_clean_ptq_minus_fp32"]["lo"] > a.margin
    H["H2_D_ptq_minus_D_fp32"]["supported"] = H["H2_D_ptq_minus_D_fp32"]["lo"] > 0
    H["H3_Abar_qatsr_minus_Abar_ptq"]["supported"] = H["H3_Abar_qatsr_minus_Abar_ptq"]["lo"] > 0
    # Holm phải đồng thuận với CI: yêu cầu cả hai
    for k in H:
        H[k]["supported_holm"] = bool(H[k]["supported"] and H[k]["p_holm"] < 0.05)
    # phụ: AUROC từng ô + CI; ΔAUROC theo từng s (ptq−fp32, qat_sr−ptq) Holm trong họ 14
    cell = pd.DataFrame([dict(shift=s, variant=v, auroc=g(A0, s, v), lo=np.percentile(g(AB, s, v), 2.5),
                              hi=np.percentile(g(AB, s, v), 97.5)) for s in ["clean"] + S for v in V])
    per = []
    for name, (va, vb) in {"ptq-fp32": ("ptq", "fp32"), "qat_sr-ptq": ("qat_sr", "ptq"),
                           "fp32_sr-fp32": ("fp32_sr", "fp32"), "qat-ptq": ("qat", "ptq")}.items():
        rows = [dict(contrast=name, shift=s, **summ(g(A0, s, va) - g(A0, s, vb), g(AB, s, va) - g(AB, s, vb))) for s in S]
        for r, h in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = float(h)
        per += rows
    summary = {v: dict(clean=float(g(A0, "clean", v)), Abar=float(Abar(A0, v)), D=float(D_(A0, v))) for v in V}
    res = dict(file=a.preds, n=int(len(y)), n_pd=int(y.sum()), B=a.B, margin=a.margin, hypotheses=H, summary=summary)
    tag = pathlib.Path(a.preds).stem.replace("patient_preds_", "")
    (out / f"hyp_{tag}.json").write_text(json.dumps(res, indent=1))
    cell.to_csv(out / f"auroc_cells_{tag}.csv", index=False); pd.DataFrame(per).to_csv(out / f"per_shift_{tag}.csv", index=False)
    print(json.dumps(res, indent=1))
    print(pd.DataFrame(per).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
