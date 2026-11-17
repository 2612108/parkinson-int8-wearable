"""P3: pipeline không rò rỉ + baseline đặc trưng (RF/XGB), nested CV 5x5 theo bệnh nhân, và bản rò rỉ có chủ đích.
Chạy: python p3_baseline.py --out /kaggle/working"""
import argparse, json, pathlib, platform, time, warnings
import numpy as np, pandas as pd
import sklearn, scipy
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import features as F, evalkit as E, pads_io as P

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/kaggle/working"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
NT = 30 if a.smoke else 500
OUT = pathlib.Path(a.out); REP = OUT / "reports" / "p3"; REP.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)
SEEDS = [int(s) for s in a.seeds.split(",")]
t0 = time.time()

# ------------------------------------------------------------ dữ liệu
npz = CACHE / "pads_50hz.npz"
if not npz.exists():
    root = P.find_pads_root(); print("PADS root:", root)
    P.build_cache(root, npz)
idx, pats, segs = P.load_cache(npz)
print(idx.shape, pats.group.value_counts().to_dict(), f"{time.time()-t0:.0f}s")
TASKS = sorted(idx.task.unique()); WR = ["LeftWrist", "RightWrist"]
REST_POST = [t for t in TASKS if t.lower().startswith("relaxed") and "task" not in t.lower()] + [t for t in TASKS if t.lower() == "stretchhold"]

def subj_features(acc_only=False, tasks=None):
    tasks = tasks or TASKS
    nf = len(F.FEAT_NAMES) * (4 if acc_only else 8)
    subs = sorted(pats.subject)
    M = np.full((len(subs), len(tasks) * 2 * nf), np.nan, np.float32)
    pos = {s: i for i, s in enumerate(subs)}
    for k, r in idx.iterrows():
        if r.task not in tasks: continue
        seg = segs[k][:3] if acc_only else segs[k]
        j = (tasks.index(r.task) * 2 + WR.index(r.wrist)) * nf
        M[pos[r.subject], j:j + nf] = F.segment_features(seg, P.FS_OUT)
    return np.array(subs), M

FEATSETS = {"accgyro_all": subj_features(False), "acc_restpost": subj_features(True, REST_POST)}
print("featsets", {k: v[1].shape for k, v in FEATSETS.items()}, "REST_POST", REST_POST, f"{time.time()-t0:.0f}s")
grp = pats.set_index("subject").group

def task_xy(subs, name):
    neg = {"PD_vs_HC": "HC", "PD_vs_DD": "DD"}[name]
    g = grp.loc[subs].to_numpy()
    m = np.isin(g, ["PD", neg])
    return m, (g[m] == "PD").astype(int)

# ------------------------------------------------------------ mô hình
def models(seed, pos_w):
    return {
        "RF": (make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=NT, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
               [{"randomforestclassifier__max_features": mf, "randomforestclassifier__min_samples_leaf": ml} for mf in ["sqrt", 0.05] for ml in [1, 3]]),
        "XGB": (make_pipeline(SimpleImputer(strategy="median"), xgb.XGBClassifier(n_estimators=30 if a.smoke else 300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3, scale_pos_weight=pos_w, n_jobs=4, random_state=seed, eval_metric="logloss", tree_method="hist")),
                [{"xgbclassifier__max_depth": d, "xgbclassifier__min_child_weight": w} for d in [2, 4] for w in [1, 5]]),
    }

def inner_select(est, grid, X, y, seed):
    """Chọn siêu tham số + ngưỡng CHỈ bằng CV trong tập train."""
    skf = StratifiedKFold(5, shuffle=True, random_state=1000 + seed)
    best = (-1, None, None)
    for g in (grid[:1] if a.smoke else grid):
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            m = clone(est).set_params(**g).fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y, oof)
        if auc > best[0]: best = (auc, g, oof)
    return best[1], E.youden_threshold(y, best[2]), best[0]

rows, oof_rows = [], []
for fs_name, (subs, M) in FEATSETS.items():
    for task in ["PD_vs_HC", "PD_vs_DD"]:
        m, y = task_xy(subs, task); X = M[m]; sid = subs[m]
        for seed in SEEDS:
            for mname in ["RF", "XGB"]:
                p = np.zeros(len(y)); thr_used = np.zeros(len(y)); chosen = []
                for k, (tr, te) in enumerate(E.outer_folds(y, 5, seed)):
                    est, grid = models(seed, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))[mname]
                    g, thr, iauc = inner_select(est, grid, X[tr], y[tr], seed)
                    mdl = clone(est).set_params(**g).fit(X[tr], y[tr])
                    p[te] = mdl.predict_proba(X[te])[:, 1]; thr_used[te] = thr; chosen.append(dict(fold=k, params=g, thr=thr, inner_auc=iauc))
                yh = (p >= thr_used).astype(int)
                met = E.metrics(y, p, 0.5); met["bal_acc_thr"] = float(sklearn.metrics.balanced_accuracy_score(y, yh))
                ci = E.bootstrap_ci(y, p, 0.5, n_boot=1000, seed=seed)
                rows.append(dict(featset=fs_name, task=task, model=mname, seed=seed, protocol="clean", **met,
                                 **{f"{k}_ci": v for k, v in ci.items()}, chosen=json.dumps(chosen, default=str)))
                oof_rows += [dict(featset=fs_name, task=task, model=mname, seed=seed, subject=int(s), y=int(t), p=float(q), thr=float(h)) for s, t, q, h in zip(sid, y, p, thr_used)]
                print(fs_name, task, mname, seed, {k: round(float(met[k]), 3) for k in ["bal_acc", "bal_acc_thr", "auroc", "ece"]}, f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------ bản RÒ RỈ có chủ đích (chỉ để đo mức thổi phồng)
# L1: chuẩn hóa + chọn 200 đặc trưng trên TOÀN BỘ dữ liệu trước CV; ngưỡng chọn trên tập test.
subs, M = FEATSETS["accgyro_all"]
for task in ["PD_vs_HC", "PD_vs_DD"]:
    m, y = task_xy(subs, task)
    Xl = SimpleImputer(strategy="median").fit_transform(M[m]); Xl = StandardScaler().fit_transform(Xl)
    Xl = SelectKBest(f_classif, k=200).fit_transform(Xl, y)
    for seed in SEEDS:
        p = np.zeros(len(y))
        for tr, te in E.outer_folds(y, 5, seed):
            p[te] = RandomForestClassifier(NT, class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(Xl[tr], y[tr]).predict_proba(Xl[te])[:, 1]
        thr = E.youden_threshold(y, p)  # rò rỉ: ngưỡng tối ưu trên test
        met = E.metrics(y, p, thr)
        rows.append(dict(featset="accgyro_all", task=task, model="RF", seed=seed, protocol="leaky_L1_globalselect_testthr", **met))
        print("LEAKY L1", task, seed, round(met["bal_acc"], 3), round(met["auroc"], 3), flush=True)

# L2: cửa sổ 4 s/50% làm mẫu, chia ngẫu nhiên theo cửa sổ (cùng người ở cả train và test) so với chia theo nhóm người.
W, H = 4 * P.FS_OUT, 2 * P.FS_OUT
wf, wy, wg = [], [], []
for k, r in idx.iterrows():
    g = grp.loc[r.subject]
    for s0 in range(0, segs[k].shape[1] - W + 1, H):
        wf.append(F.segment_features(segs[k][:, s0:s0 + W], P.FS_OUT)); wy.append(g); wg.append(r.subject)
wf, wy, wg = np.array(wf, np.float32), np.array(wy), np.array(wg)
print("windows", wf.shape, f"{time.time()-t0:.0f}s", flush=True)
for task, neg in [("PD_vs_HC", "HC"), ("PD_vs_DD", "DD")]:
    m = np.isin(wy, ["PD", neg]); Xw, yw, gw = np.nan_to_num(wf[m]), (wy[m] == "PD").astype(int), wg[m]
    for proto, splitter in [("leaky_L2_window_random", StratifiedKFold(5, shuffle=True, random_state=0).split(Xw, yw)),
                            ("clean_window_grouped", StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xw, yw, gw))]:
        p = np.zeros(len(yw))
        for tr, te in splitter:
            p[te] = RandomForestClassifier(NT, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=0).fit(Xw[tr], yw[tr]).predict_proba(Xw[te])[:, 1]
        met = E.metrics(yw, p, 0.5)
        rows.append(dict(featset="window_accgyro", task=task, model="RF", seed=0, protocol=proto, **met))
        print(proto, task, round(met["bal_acc"], 3), round(met["auroc"], 3), f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------ ghi kết quả
res = pd.DataFrame(rows); res.to_csv(REP / "p3_results_all.csv", index=False)
pd.DataFrame(oof_rows).to_csv(REP / "p3_oof_predictions.csv", index=False)
num = ["bal_acc", "bal_acc_thr", "auroc", "sens", "spec", "ece", "brier"]
summ = res.groupby(["protocol", "featset", "task", "model"])[[c for c in num if c in res]].agg(["mean", "std"]).round(3)
summ.to_csv(REP / "p3_summary.csv")
counts = pats.group.value_counts().to_dict()
md = ["# P3 — Baseline đặc trưng, nested CV 5×5 theo bệnh nhân", "",
      f"- Số người theo nhóm: {counts}; số bản ghi (subject×task×wrist): {len(idx)}; bài tập: {TASKS}",
      f"- Seeds outer: {SEEDS}; CI 95% bootstrap theo bệnh nhân (1000 lần) trong `p3_results_all.csv`",
      "- clean: imputer/siêu tham số/ngưỡng chỉ từ tập train (inner 5-fold). bal_acc ở ngưỡng 0.5; bal_acc_thr ở ngưỡng chọn trong inner CV.",
      "- leaky_L1: chuẩn hóa + chọn đặc trưng trên toàn bộ dữ liệu, ngưỡng tối ưu trên test. leaky_L2: cửa sổ chia ngẫu nhiên (cùng người ở train/test). clean_window_grouped: cửa sổ chia theo người.",
      "", "```", summ.to_string(), "```"]
(REP / "p3_summary.md").write_text("\n".join(md) + "\n")
env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgb.__version__,
           seeds=SEEDS, fs=P.FS_OUT, bandpass=[0.5, 20], window_s=4, overlap=0.5, outer=5, inner=5, runtime_s=round(time.time() - t0))
(REP / "p3_config_env.json").write_text(json.dumps(env, indent=1))
print("\n".join(md)); print(f"DONE {time.time()-t0:.0f}s")
