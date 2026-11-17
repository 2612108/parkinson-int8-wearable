"""B2b (thăm dò): ảnh hưởng của lọc nhân quả trên bo so với lọc zero-phase lúc huấn luyện.
Mô hình: reports/p56a/models/PD_vs_HC_raw_{qat,ptq}_fold0.tflite; tập test fold 0 (PD vs HC, seed 0) tái tạo như p56a.
Ba đường ống tiền xử lý trên tín hiệu thô 100 Hz:
  zp     : resample_poly + sosfiltfilt trên cả bản ghi (như huấn luyện)
  causal : resample_poly + sosfilt (nhân quả, chạy liên tục trên cả bản ghi, như thiết bị chạy streaming)
  causal_win : lọc nhân quả reset trạng thái ở mỗi cửa sổ 4 s (như firmware B2 hiện tại)
In AUROC mức bệnh nhân (trung bình xác suất cửa sổ) và tỉ lệ quyết định trùng với zp ở ngưỡng Youden của zp.
Dùng: .venv/bin/python tools/causal_eval.py --data data/raw
"""
import argparse, json, pathlib, sys
import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import p56_quant_shift as P56  # noqa: E402  (chỉ dùng hàm đọc dữ liệu, hằng số, to_input)
from src.features import resample  # noqa: E402

FS_IN, FS, WIN, HOP = 100, 50, 200, 100
SOS = signal.butter(4, [0.5, 20.0], btype="band", fs=FS, output="sos")


def wins(xf):
    n = (xf.shape[1] - WIN) // HOP + 1
    return np.stack([xf[:, s * HOP:s * HOP + WIN].T for s in range(n)]).astype(np.float32) if n > 0 else np.zeros((0, WIN, 6), np.float32)


def prep(x6, mode):
    r = resample(x6, FS_IN, FS)
    if mode == "zp":
        return wins(signal.sosfiltfilt(SOS, r, axis=-1))
    if mode == "causal":
        return wins(signal.sosfilt(SOS, r, axis=-1))
    n = (r.shape[1] - WIN) // HOP + 1   # causal_win: lọc riêng từng cửa sổ, trạng thái 0
    return np.stack([signal.sosfilt(SOS, r[:, s * HOP:s * HOP + WIN], axis=-1).T for s in range(n)]).astype(np.float32) if n > 0 else np.zeros((0, WIN, 6), np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / "data/raw")); a = ap.parse_args()
    try:
        import tensorflow as tf
    except ImportError:   # máy anh Tuyến không có TF: dùng LiteRT (cùng nhân TFLite)
        import types
        from ai_edge_litert import interpreter as _L
        tf = types.SimpleNamespace(lite=types.SimpleNamespace(Interpreter=_L.Interpreter))
    Pd, recs = P56.load_pads(P56.find_pads(a.data))
    sel = np.where(Pd.label.isin(["PD", "HC"]))[0]; y_all = (Pd.label == "PD").astype(int).to_numpy()
    tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(sel, y_all[sel])))
    te_p = sel[te]; y = y_all[te_p]
    nz = np.load(ROOT / "reports/p56a/models/norm_PD_vs_HC_raw_fold0.npz"); mu, sd = nz["mu"], nz["sd"]
    out = {"n_test": int(len(te_p)), "n_pd": int(y.sum())}
    for name in ("qat", "ptq"):
        run = P56.Int8Runner(tf, (ROOT / f"reports/p56a/models/PD_vs_HC_raw_{name}_fold0.tflite").read_bytes())
        pp = {}
        for mode in ("zp", "causal", "causal_win"):
            probs = []
            for i in te_p:
                W = [prep(x, mode) for _, _, x in recs[i]]; W = np.concatenate([w for w in W if len(w)])
                lg = run((P56.to_input(W, "raw") - mu) / sd)
                probs.append(float(np.mean(1 / (1 + np.exp(-lg)))))
            pp[mode] = np.array(probs)
        fpr_thr = None
        from sklearn.metrics import roc_curve
        f, t, th = roc_curve(y, pp["zp"]); thr = th[np.argmax(t - f)]
        out[name] = {m: dict(auroc=round(float(roc_auc_score(y, p)), 4),
                             agree_with_zp=round(float(np.mean((p >= thr) == (pp["zp"] >= thr))), 4),
                             mean_abs_dp=round(float(np.mean(np.abs(p - pp["zp"]))), 4)) for m, p in pp.items()}
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "reports/board/B2b_causal_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
