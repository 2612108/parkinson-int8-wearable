"""R1/M10b: re-estimate the feature random forest when features are computed with the device's causal filter.
Run from ~/Desktop/PARKINSON: .venv/bin/python firmware/tools/rf_causal_eval.py
Outputs reports/board/B4b_rf_causal.json. Uses the zero-phase feature cache built by gen_b4_data.py."""
import json, pathlib, sys, time
import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "firmware" / "tools"))
import scripts.p3_baseline as p3
import gen_b4_data as g

def causal_bandpass(x, fs, lo=0.5, hi=20.0, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfilt(sos, x, axis=-1)
p3.bandpass = causal_bandpass   # p3.prep() now filters causally (threads share the patched module)

def main():
    zp = np.load(ROOT / "firmware/b4_rf/main/generated/p3_pd_hc_features.npz", allow_pickle=True)
    Xz, y, sids, keys = zp["Xp"].astype(np.float32), zp["y"].astype(int), zp["sids"].tolist(), zp["keys"].tolist()
    base = p3.find_root(ROOT / "data/raw/pads"); t0 = time.time()
    res = Parallel(n_jobs=-1, backend="threading")(delayed(g.extract_subject_rec_only)(base, s) for s in sids)
    nf = Xz.shape[1] // len(keys); kk = [tuple(k.split("|")) for k in keys]
    Xc = np.full_like(Xz, np.nan)
    for i, (sid, rf) in enumerate(res):
        assert sid == sids[i]
        for j, k in enumerate(kk):
            k2 = next((kx for kx in rf if (str(kx[0]), str(kx[1])) == k), None)
            if k2 is not None: Xc[i, j * nf:(j + 1) * nf] = rf[k2]
    out = dict(t_causal_features_s=round(time.time() - t0, 1), nan_zp=int(np.isnan(Xz).sum()), nan_causal=int(np.isnan(Xc).sum()), folds=[])
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xz, y)):
        r = {}
        for train_name, Xtr_all in (("zp", Xz), ("causal", Xc)):
            imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr_all[tr])
            clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=0, n_jobs=-1)
            clf.fit(imp.transform(Xtr_all[tr]), y[tr])
            for test_name, Xte_all in (("zp", Xz), ("causal", Xc)):
                r[f"train_{train_name}_test_{test_name}"] = round(float(roc_auc_score(y[te], clf.predict_proba(imp.transform(Xte_all[te]))[:, 1])), 4)
        r["fold"] = k; out["folds"].append(r); print(r, flush=True)
    for key in [k for k in out["folds"][0] if k != "fold"]:
        v = [f[key] for f in out["folds"]]; out[key + "_mean"] = round(float(np.mean(v)), 4); out[key + "_sd"] = round(float(np.std(v, ddof=1)), 4)
    (ROOT / "reports/board/B4b_rf_causal.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
