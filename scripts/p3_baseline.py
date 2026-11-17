"""P3 — Baseline đặc trưng thủ công trên PADS, giao thức không rò rỉ ở mức bệnh nhân.

python scripts/p3_baseline.py --pads <thư mục chứa movement/ và patients/> --out reports/p3 [--seeds 0 1 2] [--quick]

Bài toán: PD vs HC, PD vs DD (nhị phân, mức bệnh nhân).
- Đặc trưng mỗi bản ghi (bài × cổ tay): lấy lại mẫu 100→50 Hz, lọc 0.5–20 Hz, 12 đặc trưng × 8 kênh
  (acc xyz, gyro xyz, |acc|, |gyro|) = 96. Vector bệnh nhân = nối theo thứ tự cố định (bài, cổ tay).
- Nested CV: ngoài 5-fold phân tầng theo bệnh nhân (lặp nhiều seed); trong 5-fold chọn siêu tham số theo AUROC,
  ngưỡng chọn trên dự đoán OOF của tập trong (chỉ train). Impute median fit trên train.
- Đối chứng rò rỉ: cùng đặc trưng mức cửa sổ 4 s, chia ngẫu nhiên theo cửa sổ (rò rỉ) vs chia theo bệnh nhân.
Kết quả: results.json, oof_<task>_<model>_seed<k>.csv, leakage.json, run_info.json.
"""
import argparse, json, pathlib, sys, time, platform
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import bandpass, resample, segment_features, FEAT_NAMES  # noqa: E402
from src.evalkit import metrics, bootstrap_ci, youden_threshold  # noqa: E402
from src.pads_io import load_txt  # noqa: E402

FS_IN, FS = 100, 50
WIN, HOP = 4 * FS, 2 * FS


def find_root(p):
    p = pathlib.Path(p)
    for cand in [p] + list(p.rglob("movement")):
        base = cand if (cand / "movement").exists() else cand.parent
        if (base / "movement").is_dir() and (base / "patients").is_dir():
            return base
    raise SystemExit(f"không tìm thấy movement/ và patients/ dưới {p}")


def label_of(cond):
    c = str(cond).strip().lower()
    if c.startswith("healthy"):
        return "HC"
    if c.startswith("parkinson"):
        return "PD"
    return "DD"


def prep(x):
    x = resample(x, FS_IN, FS)
    return bandpass(x, FS).astype(np.float32)


def subject_records(base, sid):
    obs = json.loads((base / "movement" / f"observation_{sid}.json").read_text())
    out = []
    for sess in obs["session"]:
        for rec in sess["records"]:
            out.append((sess["record_name"], rec["device_location"], base / "movement" / rec["file_name"]))
    return out


def process_subject(base, sid):
    rec_feats, win_rows = {}, []
    for task, wrist, f in subject_records(base, sid):
        if not f.exists():
            continue
        x = prep(load_txt(f))
        rec_feats[(task, wrist)] = segment_features(x, FS)
        for s in range(0, x.shape[1] - WIN + 1, HOP):
            win_rows.append((task, wrist, segment_features(x[:, s:s + WIN], FS)))
    return sid, rec_feats, win_rows


def build(base, quick=False, n_jobs=-1):
    pats = []
    for pj in sorted((base / "patients").glob("patient_*.json")):
        d = json.loads(pj.read_text())
        pats.append(dict(sid=str(d["id"]).zfill(3), condition=d.get("condition"), label=label_of(d.get("condition")),
                         age=d.get("age"), gender=d.get("gender")))
    P = pd.DataFrame(pats)
    P = P[[(base / "movement" / f"observation_{s}.json").exists() for s in P.sid]].reset_index(drop=True)
    if quick:
        P = P.groupby("label").head(25).reset_index(drop=True)
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(process_subject)(base, s) for s in P.sid)
    keys = sorted({k for _, rf, _ in res for k in rf})
    nf = len(FEAT_NAMES) * 8
    X = np.full((len(res), len(keys) * nf), np.nan, np.float32)
    W = []
    for i, (sid, rf, wr) in enumerate(res):
        for j, k in enumerate(keys):
            if k in rf:
                X[i, j * nf:(j + 1) * nf] = rf[k]
        for task, wrist, v in wr:
            W.append((sid, task, wrist, v))
    return P, X, keys, W


# ------------------------------------------------------------------ models
def make_model(name, params, seed):
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=seed, **params)
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                            eval_metric="logloss", n_jobs=-1, random_state=seed, tree_method="hist", **params)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), clf)


GRIDS = {"rf": {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]},
         "xgb": {"max_depth": [3, 5], "min_child_weight": [1, 5]}}


def fit_predict(name, params, Xtr, ytr, Xte, seed):
    m = make_model(name, params, seed)
    kw = {}
    if name == "xgb":
        pos = ytr.mean()
        m.set_params(xgbclassifier__scale_pos_weight=(1 - pos) / max(pos, 1e-6))
    m.fit(Xtr, ytr, **kw)
    return m.predict_proba(Xte)[:, 1]


def nested_cv(name, X, y, seed, n_outer=5, n_inner=5):
    oof = np.zeros(len(y)); thr_used = np.zeros(len(y)); chosen = []
    for k, (tr, te) in enumerate(StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(X, y)):
        inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed + 100).split(X[tr], y[tr]))
        best, best_auc, best_oof = None, -1, None
        for params in ParameterGrid(GRIDS[name]):
            io = np.zeros(len(tr))
            for itr, ite in inner:
                io[ite] = fit_predict(name, params, X[tr][itr], y[tr][itr], X[tr][ite], seed)
            auc = roc_auc_score(y[tr], io)
            if auc > best_auc:
                best, best_auc, best_oof = params, auc, io
        thr = youden_threshold(y[tr], best_oof)
        oof[te] = fit_predict(name, best, X[tr], y[tr], X[te], seed)
        thr_used[te] = thr
        chosen.append(dict(fold=k, params=best, inner_auc=float(best_auc), thr=thr))
    yh = (oof >= thr_used).astype(int)
    m = metrics(y, oof, thr=0.5)
    m["bal_acc"] = float(balanced_accuracy_score(y, yh))
    m["sens"] = float(((yh == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    m["spec"] = float(((yh == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return oof, thr_used, m, chosen


# ------------------------------------------------------------------ leakage demo
def leakage(W, P, task_pair, seed=0):
    lab = dict(zip(P.sid, P.label))
    rows = [(s, v) for s, t, w, v in W if lab.get(s) in task_pair]
    if not rows:
        return {}
    sids = np.array([s for s, _ in rows]); Xw = np.stack([v for _, v in rows])
    y = np.array([1 if lab[s] == "PD" else 0 for s in sids])
    out = {}
    for split in ["random_window", "by_patient"]:
        if split == "random_window":
            folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(Xw, y)
        else:
            folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xw, y, groups=sids)
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = fit_predict("rf", {"max_features": "sqrt", "min_samples_leaf": 1}, Xw[tr], y[tr], Xw[te], seed)
        out[split] = dict(window_bal_acc=float(balanced_accuracy_score(y, p >= 0.5)),
                          window_auroc=float(roc_auc_score(y, p)), n_windows=int(len(y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads", required=True); ap.add_argument("--out", default=str(ROOT / "reports/p3"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"])
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = find_root(a.pads)
    P, X, keys, W = build(base, quick=a.quick)
    P.to_csv(out / "patients_used.csv", index=False)
    info = dict(n_patients=len(P), label_counts=P.label.value_counts().to_dict(), n_records=len(keys),
                records=[f"{t}|{w}" for t, w in keys], n_features=int(X.shape[1]),
                nan_frac=float(np.isnan(X).mean()), n_windows=len(W), t_features_s=round(time.time() - t0, 1),
                python=platform.python_version(), seeds=a.seeds, quick=a.quick)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    results = {}
    for pair, name_pair in [(("PD", "HC"), "PD_vs_HC"), (("PD", "DD"), "PD_vs_DD")]:
        idx = P.label.isin(pair).to_numpy()
        y = (P.label[idx] == "PD").astype(int).to_numpy(); Xp = X[idx]
        for mname in a.models:
            per_seed = []
            for sd in a.seeds:
                oof, thr, m, chosen = nested_cv(mname, Xp, y, sd)
                pd.DataFrame(dict(sid=P.sid[idx].to_numpy(), y=y, p=oof, thr=thr)).to_csv(
                    out / f"oof_{name_pair}_{mname}_seed{sd}.csv", index=False)
                if sd == a.seeds[0]:
                    m["ci95"] = bootstrap_ci(y, oof, thr=thr)
                m["chosen"] = chosen; m["seed"] = sd
                per_seed.append(m)
                print(name_pair, mname, sd, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
            agg = {k: (float(np.mean([s[k] for s in per_seed])), float(np.std([s[k] for s in per_seed])))
                   for k in ["bal_acc", "auroc", "sens", "spec", "ece", "brier"]}
            results[f"{name_pair}/{mname}"] = dict(mean_sd=agg, per_seed=per_seed, n=int(len(y)), n_pos=int(y.sum()))
        results[f"{name_pair}/leakage_demo"] = leakage(W, P, pair)
        print(name_pair, "leakage", results[f"{name_pair}/leakage_demo"])
    info["t_total_s"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print("xong", info["t_total_s"], "s")


if __name__ == "__main__":
    main()
