"""P4 — Mô hình học sâu trên PADS, đánh giá mức bệnh nhân không rò rỉ (TensorFlow/Keras, chạy GPU Kaggle).

python scripts/p4_dl.py --pads <thư mục PADS> --out reports/p4 [--models ds_tcn cnn1d ...] [--inputs raw inv]
                        [--seeds 0] [--epochs 60] [--quick]

Giao thức (ghi rõ trong bài):
- Ngoài: StratifiedKFold 5 theo bệnh nhân (cùng seed với P3 → cùng các fold).
- Trong: tách 20% bệnh nhân của tập train làm validation (phân tầng) để early stopping, hiệu chỉnh nhiệt độ
  và chọn ngưỡng (Youden trên validation). Không có thông tin nào từ fold test.
- Chuẩn hóa z-score theo kênh: thống kê tính trên cửa sổ của tập train (trong) mỗi fold.
- Đầu vào: cửa sổ 4 s @ 50 Hz (200 mẫu), chồng 50 %, từ mọi bài × hai cổ tay.
  raw = acc xyz, gyro xyz, |acc|, |gyro| (8 kênh);
  inv = bất biến với phép quay của đồng hồ: |a|, |ω|, a·ω, |a×ω|, |Δa|, |Δω| (6 kênh; PADS không có trọng lực).
- Gộp lên bệnh nhân: (a) meanprob = trung bình xác suất cửa sổ; (b) mil = attention-MIL trên túi cửa sổ.
- QUY TẮC BO: không dùng dilation (TFLM sai bit với SPACE_TO_BATCH_ND, xem reports/board/B1_DIAG.txt).
Kết quả: results.json, oof_<task>_<model>_<input>_<agg>_seed<k>.csv, models/ (encoder .keras của ds_tcn từng fold).
"""
import argparse, json, pathlib, sys, time, platform
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import bandpass, resample  # noqa: E402
from src.evalkit import metrics, bootstrap_ci, youden_threshold  # noqa: E402
from src.pads_io import load_txt, invariant_channels  # noqa: E402

FS_IN, FS, WIN, HOP = 100, 50, 200, 100


# ------------------------------------------------------------------ dữ liệu
def find_root(p):
    p = pathlib.Path(p)
    for obs in p.rglob("observation_001.json"):
        base = obs.parent.parent
        if (base / "patients").is_dir():
            return base
    raise SystemExit(f"không tìm thấy PADS dưới {p}")


def label_of(c):
    c = str(c).strip().lower()
    return "HC" if c.startswith("healthy") else "PD" if c.startswith("parkinson") else "DD"


def windows_of(x6):
    """x6: (6, T) @100 Hz (đã nội suy theo Time) → (raw8, inv6) cửa sổ (n, 200, C).
    PADS không có trọng lực trong acc → inv dùng kênh bất biến quay (src/pads_io.invariant_channels)."""
    xf = bandpass(resample(x6, FS_IN, FS), FS)
    a, w = xf[0:3], xf[3:6]
    raw_all = np.vstack([a, w, np.linalg.norm(a, axis=0), np.linalg.norm(w, axis=0)])
    inv_all = invariant_channels(a, w)
    raw, inv = [], []
    for s in range(0, xf.shape[1] - WIN + 1, HOP):
        raw.append(raw_all[:, s:s + WIN].T); inv.append(inv_all[:, s:s + WIN].T)
    return np.asarray(raw, np.float32), np.asarray(inv, np.float32)


def build(base, quick=False):
    from joblib import Parallel, delayed
    pats = []
    for pj in sorted((base / "patients").glob("patient_*.json")):
        d = json.loads(pj.read_text()); sid = str(d["id"]).zfill(3)
        if (base / "movement" / f"observation_{sid}.json").exists():
            pats.append(dict(sid=sid, label=label_of(d.get("condition"))))
    P = pd.DataFrame(pats)
    if quick:
        P = P.groupby("label").head(20).reset_index(drop=True)

    def one(sid):
        obs = json.loads((base / "movement" / f"observation_{sid}.json").read_text())
        R, I = [], []
        for sess in obs["session"]:
            for rec in sess["records"]:
                f = base / "movement" / rec["file_name"]
                if f.exists():
                    r, i = windows_of(load_txt(f)); R.append(r); I.append(i)
        return np.concatenate(R), np.concatenate(I)
    res = Parallel(n_jobs=-1)(delayed(one)(s) for s in P.sid)
    owner = np.concatenate([np.full(len(r), k) for k, (r, _) in enumerate(res)])
    return P, np.concatenate([r for r, _ in res]), np.concatenate([i for _, i in res]), owner


# ------------------------------------------------------------------ mô hình (không dilation)
def encoder(name, C, tf):
    L = tf.keras.layers
    i = L.Input((WIN, C))
    if name == "cnn1d":
        x = i
        for f in (32, 64, 64):
            x = L.Conv1D(f, 5, padding="same", activation="relu")(x); x = L.MaxPooling1D(2)(x)
        x = L.GlobalAveragePooling1D()(x)
    elif name == "ds_cnn":
        x = L.Conv1D(32, 5, strides=2, padding="same", activation="relu")(i)
        for f in (48, 64, 64):
            x = L.DepthwiseConv1D(5, padding="same")(x); x = L.Conv1D(f, 1, activation="relu")(x)
            x = L.MaxPooling1D(2)(x)
        x = L.GlobalAveragePooling1D()(x)
    elif name == "ds_tcn":  # đề xuất: khối DS dư, tăng trường tiếp nhận bằng stride (không dilation), ≤30k tham số
        x = L.Conv1D(32, 7, strides=2, padding="same")(i); x = L.BatchNormalization()(x); x = L.ReLU()(x)
        for f, s in ((32, 1), (48, 2), (48, 1), (64, 2), (64, 1)):
            y = L.DepthwiseConv1D(7, strides=s, padding="same")(x); y = L.BatchNormalization()(y)
            y = L.Conv1D(f, 1)(y); y = L.BatchNormalization()(y)
            sc = x if (s == 1 and x.shape[-1] == f) else L.Conv1D(f, 1, strides=s)(x)
            x = L.ReLU()(L.Add()([sc, y]))
        x = L.GlobalAveragePooling1D()(x)
    elif name == "cnn_lstm":
        x = L.Conv1D(32, 5, padding="same", activation="relu")(i); x = L.MaxPooling1D(2)(x)
        x = L.Conv1D(64, 5, padding="same", activation="relu")(x); x = L.MaxPooling1D(2)(x)
        x = L.LSTM(64)(x)
    elif name == "inception":  # trần trên (lớn), không dành cho MCU
        x = i
        for _ in range(3):
            b = [L.Conv1D(32, k, padding="same", activation="relu")(x) for k in (9, 19, 39)]
            b.append(L.Conv1D(32, 1, activation="relu")(L.MaxPooling1D(3, 1, padding="same")(x)))
            x = L.BatchNormalization()(L.Concatenate()(b))
        x = L.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(name)
    x = L.Dropout(0.3)(x)
    return tf.keras.Model(i, x, name=f"enc_{name}")


def window_model(enc, tf):
    L = tf.keras.layers
    o = L.Dense(1)(enc.output)
    return tf.keras.Model(enc.input, o)


def mil_model(enc, tf, K):
    L = tf.keras.layers
    bag = L.Input((K, WIN, enc.input.shape[-1]))
    h = L.TimeDistributed(enc)(bag)                                   # (B, K, D)
    a = L.Dense(1)(L.Dense(32, activation="tanh")(h))                 # attention (Ilse 2018, bản không gated)
    a = L.Softmax(axis=1)(a)
    import keras
    z = keras.ops.sum(keras.ops.multiply(a, h), axis=1)
    return tf.keras.Model(bag, L.Dense(1)(z))


# ------------------------------------------------------------------ huấn luyện
def zfit(X):
    m = X.reshape(-1, X.shape[-1]).mean(0); s = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    return m.astype(np.float32), s.astype(np.float32)


def temp_scale(logit, y):
    from scipy.optimize import minimize_scalar
    def nll(T):
        p = 1 / (1 + np.exp(-logit / T)); p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def patient_logits_window(model, X, owner, idx, bs=1024):
    lg = model.predict(X, batch_size=bs, verbose=0).ravel()
    p = 1 / (1 + np.exp(-lg))
    # trung bình xác suất cửa sổ → logit bệnh nhân
    out = np.array([p[owner == k].mean() for k in idx])
    return np.log(np.clip(out, 1e-6, 1 - 1e-6) / np.clip(1 - out, 1e-6, 1))


def run_fold(tf, enc_name, X, owner, y_pat, tr_p, va_p, te_p, agg, seed, epochs, K=48, patience=6, bs=512):
    import gc
    tf.keras.backend.clear_session(); gc.collect()   # tránh rò RAM qua các fold (P4b v2 bị kill im lặng)
    tf.keras.utils.set_random_seed(seed)
    C = X.shape[-1]
    w_tr = np.isin(owner, tr_p)
    mu, sd = zfit(X[w_tr]); Xn = (X - mu) / sd
    enc = encoder(enc_name, C, tf)
    pos = y_pat[tr_p].mean(); cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True)
    if agg == "meanprob":
        m = window_model(enc, tf)
        m.compile(tf.keras.optimizers.Adam(1e-3), tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.AUC(from_logits=True, name="auc")])
        yw = y_pat[owner]
        w_va = np.isin(owner, va_p)
        h = m.fit(Xn[w_tr], yw[w_tr], validation_data=(Xn[w_va], yw[w_va]), epochs=epochs, batch_size=bs,
                  class_weight=cw, callbacks=[es], verbose=0)
        lv = patient_logits_window(m, Xn, owner, va_p); lt = patient_logits_window(m, Xn, owner, te_p)
    else:
        rng = np.random.default_rng(seed)
        groups = {k: np.where(owner == k)[0] for k in np.unique(owner)}
        def bag(k):
            g = groups[k]; return Xn[rng.choice(g, K, replace=len(g) < K)]
        def gen(ids):
            while True:
                b = rng.choice(ids, 16)
                yield np.stack([bag(k) for k in b]), y_pat[b].astype(np.float32)
        m = mil_model(enc, tf, K)
        m.compile(tf.keras.optimizers.Adam(1e-3), tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.AUC(from_logits=True, name="auc")])
        Xva = np.stack([bag(k) for k in va_p for _ in range(4)]); yva = np.repeat(y_pat[va_p], 4)
        h = m.fit(gen(tr_p), steps_per_epoch=max(20, len(tr_p) // 16), validation_data=(Xva, yva),
                  epochs=epochs, callbacks=[es], verbose=0)
        def plog(ids, reps=8):  # trung bình logit trên nhiều túi ngẫu nhiên (đánh giá ổn định), dự đoán theo lô
            B = np.stack([bag(k) for k in ids for _ in range(reps)])
            return m.predict(B, batch_size=32, verbose=0).reshape(len(ids), reps).mean(1)
        lv, lt = plog(va_p), plog(te_p)
    T = temp_scale(lv, y_pat[va_p])
    pv, pt = 1 / (1 + np.exp(-lv / T)), 1 / (1 + np.exp(-lt / T))
    thr = youden_threshold(y_pat[va_p], pv)
    return pt, thr, dict(T=T, params=int(enc.count_params()), epochs_run=len(h.history["loss"])), enc, (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads", required=True); ap.add_argument("--out", default=str(ROOT / "reports/p4"))
    ap.add_argument("--models", nargs="+", default=["ds_tcn", "ds_cnn", "cnn1d", "cnn_lstm", "inception"])
    ap.add_argument("--inputs", nargs="+", default=["raw", "inv"])
    ap.add_argument("--aggs", nargs="+", default=["meanprob", "mil"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=["PD_vs_HC", "PD_vs_DD"])
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    import tensorflow as tf, traceback, resource
    out = pathlib.Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P, XR, XI, owner = build(find_root(a.pads), a.quick)
    info = dict(n_patients=len(P), labels=P.label.value_counts().to_dict(), n_windows=int(len(owner)),
                t_build_s=round(time.time() - t0, 1), tf=tf.__version__, gpus=[d.name for d in tf.config.list_physical_devices("GPU")],
                python=platform.python_version(), args=vars(a))
    print(json.dumps(info, indent=1)); (out / "run_info.json").write_text(json.dumps(info, indent=1))
    results = {}
    rpath = out / "results.json"
    for pair, tname in [(("PD", "HC"), "PD_vs_HC"), (("PD", "DD"), "PD_vs_DD")]:
        if tname not in a.tasks:
            continue
        sel = np.where(P.label.isin(pair))[0]
        y_all = (P.label == "PD").astype(int).to_numpy()
        wmask = np.isin(owner, sel)
        for inp in a.inputs:
            X = (XR if inp == "raw" else XI)[wmask]; ow = owner[wmask]
            for mname in a.models:
                for agg in a.aggs:
                    key = f"{tname}/{mname}/{inp}/{agg}"
                    per_seed = []
                    try:
                      for sd in a.seeds:
                          oof = np.zeros(len(sel)); thr_v = np.zeros(len(sel)); meta = []
                          ysel = y_all[sel]
                          for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=sd).split(sel, ysel)):
                              tr_p, va_p = train_test_split(sel[tr], test_size=0.2, stratify=ysel[tr], random_state=sd)
                              t1 = time.time()
                              pt, thr, md, enc, norm = run_fold(tf, mname, X, ow, y_all, tr_p, va_p, sel[te], agg, sd, a.epochs, patience=a.patience)
                              oof[te], thr_v[te] = pt, thr
                              md.update(fold=k, sec=round(time.time() - t1, 1), maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)); meta.append(md)
                              print(key, "fold", k, md, flush=True)
                              if mname == "ds_tcn" and agg == "meanprob":
                                  enc.save(out / "models" / f"enc_{tname}_{inp}_seed{sd}_fold{k}.keras")
                                  np.savez(out / "models" / f"norm_{tname}_{inp}_seed{sd}_fold{k}.npz", mu=norm[0], sd=norm[1])
                          yh = (oof >= thr_v).astype(int)
                          m = metrics(ysel, oof); m["bal_acc"] = float(balanced_accuracy_score(ysel, yh))
                          m["sens"] = float(yh[ysel == 1].mean()); m["spec"] = float(1 - yh[ysel == 0].mean())
                          m["ci95"] = bootstrap_ci(ysel, oof, thr=thr_v)
                          m["folds"] = meta; m["seed"] = sd
                          pd.DataFrame(dict(sid=P.sid.to_numpy()[sel], y=ysel, p=oof, thr=thr_v)).to_csv(
                              out / f"oof_{tname}_{mname}_{inp}_{agg}_seed{sd}.csv", index=False)
                          per_seed.append(m)
                          print(key, sd, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
                    except Exception:
                        err = traceback.format_exc(); print("LỖI", key, err, flush=True)
                        results[key + "/ERROR"] = err
                    results[key] = per_seed
                    rpath.write_text(json.dumps(results, indent=1, default=float))  # ghi dần để không mất khi hết giờ
    info["t_total_s"] = round(time.time() - t0, 1); (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print("xong", info["t_total_s"])


if __name__ == "__main__":
    main()
