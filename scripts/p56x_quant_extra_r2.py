"""P5+P6 — Lượng tử hóa INT8 dưới dịch chuyển thiết bị + kiểm định ngoài Monipar (TF + tf_keras + TFMOT).

python scripts/p56_quant_shift.py --data /kaggle/input --out reports/p56 --task PD_vs_HC --inputs raw inv [--monipar]
                                   [--folds 5] [--seed 0] [--epochs 40] [--qat_epochs 6] [--quick]

Mô hình: DS-TCN bằng Conv2D/DepthwiseConv2D kernel (k,1) (TFMOT không hỗ trợ Conv1D; op TFLite chỉ còn
ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MEAN — đều có ESP-NN). KHÔNG dilation.
Biến thể (mỗi fold ngoài, chia theo bệnh nhân, 20 % train làm validation):
  fp32, fp32_sr (huấn luyện có ngẫu nhiên hóa cảm biến), ptq (INT8 PTQ từ fp32), ptq_sr (từ fp32_sr),
  qat (QAT tinh chỉnh từ fp32), qat_sr (QAT + ngẫu nhiên hóa, từ fp32_sr).
Nhiệt độ + ngưỡng: khớp trên validation (sạch) cho từng biến thể, giữ nguyên khi đánh giá dưới dịch chuyển.
Dịch chuyển mô phỏng (áp lên tín hiệu thô 100 Hz của bệnh nhân test, rồi chạy lại toàn bộ tiền xử lý):
  fs (50/25 Hz), jitter thời gian, dải đo + độ phân giải ADC có trọng lực (±4 g 12-bit, ±2 g 10-bit, ±2 g 8-bit),
  nhiễu trắng, quay trục, đổi tay (phản xạ), mất gyro.
Monipar (--monipar): huấn luyện trên toàn bộ PADS (PD vs HC, các bài khớp), đầu vào chỉ gia tốc, đánh giá zero-shot.
Xuất cho bo: reports/.../models/<input>_{ptq,qat_sr}_fold0.tflite + bench_inputs_<input>.npy (50 cửa sổ test đã chuẩn hóa).
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import argparse, json, pathlib, sys, time, gc
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import bandpass, resample  # noqa: E402
from src.evalkit import metrics, bootstrap_ci, youden_threshold, ece  # noqa: E402
from src.pads_io import load_txt, invariant_channels  # noqa: E402

FS_IN, FS, WIN, HOP = 100, 50, 200, 100
MATCHED = {"Relaxed": "rest", "RelaxedTask": "rest", "StretchHold": "postural", "HoldWeight": "postural", "TouchNose": "hand_to_chest"}


# ---------------------------------------------------------------- dữ liệu PADS
def find_pads(p):
    for obs in pathlib.Path(p).rglob("observation_001.json"):
        if (obs.parent.parent / "patients").is_dir():
            return obs.parent.parent
    raise SystemExit("không thấy PADS")


def label_of(c):
    c = str(c).strip().lower()
    return "HC" if c.startswith("healthy") else "PD" if c.startswith("parkinson") else "DD"


def load_pads(base, quick=False):
    from joblib import Parallel, delayed
    rows = []
    for pj in sorted((base / "patients").glob("patient_*.json")):
        d = json.loads(pj.read_text()); rows.append(dict(sid=str(d["id"]).zfill(3), label=label_of(d.get("condition"))))
    P = pd.DataFrame(rows)
    if quick:
        P = P.groupby("label").head(15).reset_index(drop=True)

    def one(sid):
        obs = json.loads((base / "movement" / f"observation_{sid}.json").read_text()); out = []
        for sess in obs["session"]:
            for rec in sess["records"]:
                f = base / "movement" / rec["file_name"]
                if f.exists():
                    out.append((sess["record_name"], rec["device_location"], load_txt(f)))
        return out
    recs = Parallel(n_jobs=-1)(delayed(one)(s) for s in P.sid)
    return P, recs   # recs[i] = list (task, wrist, x6 @100 Hz)


CAUSAL = False   # D7: --causal → lọc Butterworth nhân quả (sosfilt), khớp với firmware trên bo


def preprocess(x6, fs_in=FS_IN, causal=None):
    if (CAUSAL if causal is None else causal):
        from scipy import signal as _sg
        _sos = _sg.butter(4, [0.5, 20.0], btype="band", fs=FS, output="sos")
        xf = _sg.sosfilt(_sos, resample(x6, fs_in, FS), axis=-1)
    else:
        xf = bandpass(resample(x6, fs_in, FS), FS)
    n = (xf.shape[1] - WIN) // HOP + 1
    if n <= 0:
        return np.zeros((0, WIN, 6), np.float32)
    return np.stack([xf[:, s * HOP:s * HOP + WIN].T for s in range(n)]).astype(np.float32)   # (n, 200, 6) a,w


def windows_for(recs_i, task_filter=None, shift=None, rng=None):
    W = []
    for task, wrist, x in recs_i:
        if task_filter and task not in task_filter:
            continue
        fs_in = FS_IN
        if shift is not None and shift[0] == "causalin":   # R2: test-time causal filtering of a zero-phase-trained model
            w = preprocess(x, fs_in, causal=True)
            if len(w):
                W.append(w)
            continue
        if shift is not None:
            x, fs_in = apply_shift(x, shift, rng)
        w = preprocess(x, fs_in)
        if len(w):
            W.append(w)
    return np.concatenate(W) if W else np.zeros((0, WIN, 6), np.float32)


# ---------------------------------------------------------------- đầu vào
def to_input(W, mode):
    a, w = W[..., 0:3], W[..., 3:6]
    an, wn = np.linalg.norm(a, axis=-1, keepdims=True), np.linalg.norm(w, axis=-1, keepdims=True)
    if mode == "raw":
        X = np.concatenate([a, w, an, wn], -1)
    elif mode == "inv":
        X = np.stack([invariant_channels(a[i].T, w[i].T).T for i in range(len(W))]) if len(W) else np.zeros((0, WIN, 6))
    elif mode == "acc":
        X = np.concatenate([a, an], -1)
    elif mode == "invacc":
        da = np.linalg.norm(np.diff(a, axis=1, prepend=a[:, :1]), axis=-1, keepdims=True)
        X = np.concatenate([an, da], -1)
    else:
        raise ValueError(mode)
    return X.astype(np.float32)[:, :, None, :]   # (n, 200, 1, C)


# ---------------------------------------------------------------- dịch chuyển mô phỏng (tín hiệu thô 100 Hz, g & rad/s, không trọng lực)
SHIFTS = [("clean", None),
          ("fs50", ("fs", 50)), ("fs25", ("fs", 25)),
          ("jitter10", ("jitter", 0.10)), ("jitter30", ("jitter", 0.30)),
          ("adc4g12b", ("adc", (4, 12))), ("adc2g10b", ("adc", (2, 10))), ("adc2g8b", ("adc", (2, 8))),
          ("noise_lo", ("noise", (0.005, 0.005))), ("noise_hi", ("noise", (0.02, 0.02))),
          ("rot15", ("rot", 15)), ("rot45", ("rot", 45)), ("rot180", ("rot", 180)),
          ("mirror", ("mirror", 1)), ("nogyro", ("nogyro", 1))]


def apply_shift(x, shift, rng):
    kind, lv = shift; x = x.copy(); fs = FS_IN
    if kind == "fs":
        return resample(x, FS_IN, lv).astype(np.float32), lv
    if kind == "jitter":   # đồng hồ lấy mẫu dao động nhưng bị coi là đều
        t = np.arange(x.shape[1]) / FS_IN
        tj = np.sort(t + rng.normal(0, lv / FS_IN, size=t.shape))
        x = np.stack([np.interp(tj, t, c) for c in x])
    elif kind == "adc":    # thêm trọng lực theo hướng ngẫu nhiên, cắt theo dải đo, lượng tử hóa, bỏ trọng lực
        R, bits = lv
        g = Rotation.random(random_state=rng.integers(1 << 31)).apply([0, 0, 1.0])[:, None]
        a = np.clip(x[0:3] + g, -R, R); lsb = 2 * R / 2 ** bits
        x[0:3] = np.round(a / lsb) * lsb - g
    elif kind == "noise":
        sa, sw = lv; x[0:3] += rng.normal(0, sa, x[0:3].shape); x[3:6] += rng.normal(0, sw, x[3:6].shape)
    elif kind == "rot":
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        M = Rotation.from_rotvec(np.deg2rad(lv) * ax).as_matrix()
        x[0:3] = M @ x[0:3]; x[3:6] = M @ x[3:6]
    elif kind == "mirror":  # phản xạ qua mặt phẳng dọc: a_x đổi dấu; vectơ trục ω: ω_y, ω_z đổi dấu
        x[0] *= -1; x[4] *= -1; x[5] *= -1
    elif kind == "nogyro":
        x[3:6] = 0
    return x.astype(np.float32), fs


def sensor_randomize(X6, rng):
    """Ngẫu nhiên hóa cảm biến khi huấn luyện, trên cửa sổ đã lọc (n,200,6): quay ngẫu nhiên, hệ số khuếch đại,
    nhiễu, bão hòa nhẹ, thỉnh thoảng phản xạ."""
    n = len(X6); out = X6.copy()
    R = Rotation.random(n, random_state=rng.integers(1 << 31)).as_matrix().astype(np.float32)
    out[..., 0:3] = np.einsum("nij,ntj->nti", R, out[..., 0:3]); out[..., 3:6] = np.einsum("nij,ntj->nti", R, out[..., 3:6])
    out *= rng.uniform(0.9, 1.1, (n, 1, 1)).astype(np.float32)
    out[..., 0:3] += rng.normal(0, 1, (n, 1, 1)) * rng.uniform(0, 0.02) * rng.normal(size=out[..., 0:3].shape)
    out[..., 3:6] += rng.uniform(0, 0.02) * rng.normal(size=out[..., 3:6].shape)
    m = rng.random(n) < 0.25
    out[m, :, 0] *= -1; out[m, :, 4] *= -1; out[m, :, 5] *= -1
    return out.astype(np.float32)


# ---------------------------------------------------------------- mô hình
def ds_tcn2d(C, tf):
    L = tf.keras.layers
    i = L.Input((WIN, 1, C))
    x = L.Conv2D(32, (7, 1), strides=(2, 1), padding="same")(i); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    for f, s in ((32, 1), (48, 2), (48, 1), (64, 2), (64, 1)):
        y = L.DepthwiseConv2D((7, 1), strides=(s, s), padding="same")(x); y = L.BatchNormalization()(y)  # (s,s) ≡ (s,1) vì W=1 (D3c); D10: khôi phục BN sau depthwise
        y = L.Conv2D(f, 1)(y); y = L.BatchNormalization()(y)
        sc = x if (s == 1 and x.shape[-1] == f) else L.Conv2D(f, 1, strides=(s, 1))(x)
        x = L.ReLU()(L.Add()([sc, y]))
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(i, L.Dense(1)(L.Dropout(0.3)(x)))


def cnn2d(C, tf):
    """D6/E3: CNN thường (không tách depthwise), chỉ Conv2D (k,1) → op INT8 có sẵn trong TFLM."""
    L = tf.keras.layers
    i = L.Input((WIN, 1, C)); x = i
    for f, k, s in ((32, 7, 2), (48, 7, 2), (64, 5, 2), (64, 3, 1)):
        x = L.Conv2D(f, (k, 1), strides=(s, s), padding="same")(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(i, L.Dense(1)(L.Dropout(0.3)(x)))


ARCH = {"ds_tcn": ds_tcn2d, "cnn": cnn2d}


def fit(tf, model, Xtr6, ytr, Xva, yva, mode, mu, sd, sr, epochs, lr, rng, patience=6, bs=512):
    """Huấn luyện mức cửa sổ bằng vòng lặp train_on_batch (KHÔNG dùng generator: P4 đo được rò ~4 GB RAM/fold
    với model.fit(generator) → bị kill ở 31 GB). Ngẫu nhiên hóa trên tín hiệu a,w rồi mới tạo đầu vào.
    Early stopping theo AUROC cửa sổ trên validation, khôi phục trọng số tốt nhất."""
    pos = ytr.mean(); w = np.where(ytr == 1, 0.5 / pos, 0.5 / (1 - pos)).astype(np.float32)
    model.compile(tf.keras.optimizers.Adam(lr), tf.keras.losses.BinaryCrossentropy(from_logits=True))
    Xvn = (Xva - mu) / sd
    best, best_w, wait, ep_run = -1.0, model.get_weights(), 0, 0
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr6))
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]; W6 = Xtr6[b]
            if sr:
                W6 = sensor_randomize(W6, rng)
            model.train_on_batch((to_input(W6, mode) - mu) / sd, ytr[b].astype(np.float32), sample_weight=w[b])
        ep_run = ep + 1
        auc = roc_auc_score(yva, model.predict(Xvn, batch_size=2048, verbose=0).ravel())
        if auc > best + 1e-4:
            best, best_w, wait = auc, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.set_weights(best_w)
    return ep_run


def to_tflite(tf, model, rep):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([rep[k:k + 1]] for k in range(len(rep)))
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
    return c.convert()


def to_tflite16(tf, model, rep):
    """D9/E5: PTQ chế độ 16x8 (trọng số int8, kích hoạt int16), vào/ra float — chẩn đoán lỗi do lượng tử kích hoạt."""
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([rep[k:k + 1]] for k in range(len(rep)))
    c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    return b"F16x8" + c.convert()


def fold_bn_cle(tf, model, cle=True):
    """R2: build an equivalent DS-TCN without BatchNorm (BN folded into the preceding convolution) and, if cle,
    apply cross-layer equalization between each depthwise conv and the following pointwise conv (no nonlinearity
    in between, so the rescaling is exact; Nagel et al. 2019)."""
    L = tf.keras.layers
    W = {}
    lay = model.layers
    def fold(conv, bn, depthwise=False):
        w = conv.get_weights(); k = w[0]; b = w[1] if len(w) > 1 else np.zeros(k.shape[-2] if depthwise else k.shape[-1], np.float32)
        g, be, mm, mv = bn.get_weights(); sc = g / np.sqrt(mv + bn.epsilon)
        if depthwise:
            k = k * sc[None, None, :, None]
        else:
            k = k * sc[None, None, None, :]
        return k.astype(np.float32), ((b - mm) * sc + be).astype(np.float32)
    # walk graph: stem conv, bn, relu; blocks: dw, bn, pw, bn, [shortcut conv], add, relu
    convs = [l for l in lay if isinstance(l, (L.Conv2D, L.DepthwiseConv2D))]
    bns = [l for l in lay if isinstance(l, L.BatchNormalization)]
    def bn_after(l):
        for n in l.outbound_nodes:
            if isinstance(n.outbound_layer, L.BatchNormalization):
                return n.outbound_layer
        return None
    C = model.input.shape[-1]; i = L.Input(model.input.shape[1:])
    stem = convs[0]; k, b = fold(stem, bn_after(stem))
    x = L.Conv2D(k.shape[-1], k.shape[:2], strides=stem.strides, padding="same", name="stem")(i)
    W["stem"] = [k, b]; x = L.ReLU()(x)
    dws = [l for l in convs if isinstance(l, L.DepthwiseConv2D)]
    blk = 0
    for dw in dws:
        bn1 = bn_after(dw); pw = bn1.outbound_nodes[0].outbound_layer; bn2 = bn_after(pw)
        kd, bd = fold(dw, bn1, depthwise=True); kp, bp = fold(pw, bn2)
        if cle:
            r1 = np.abs(kd).max(axis=(0, 1, 3)) + 1e-12            # per depthwise output channel
            r2 = np.abs(kp).max(axis=(0, 1, 3)) + 1e-12            # per pointwise input channel
            sfac = np.sqrt(r1 * r2) / r2                            # s_i ; dw/=s, pw*=s
            kd = kd / sfac[None, None, :, None]; bd = bd / sfac; kp = kp * sfac[None, None, :, None]
        # shortcut: identity or 1x1 conv (no BN)
        add = bn2.outbound_nodes[0].outbound_layer
        sc_src = [n for n in add.inbound_nodes[0].inbound_layers if n is not bn2][0]
        y = L.DepthwiseConv2D(dw.kernel_size, strides=dw.strides, padding="same", name=f"dw{blk}")(x)
        y = L.Conv2D(kp.shape[-1], 1, name=f"pw{blk}")(y)
        W[f"dw{blk}"] = [kd, bd]; W[f"pw{blk}"] = [kp, bp]
        if isinstance(sc_src, L.Conv2D):
            scl = L.Conv2D(sc_src.filters, 1, strides=sc_src.strides, name=f"sc{blk}"); sc = scl(x); W[f"sc{blk}"] = sc_src.get_weights()
        else:
            sc = x
        x = L.ReLU()(L.Add()([sc, y])); blk += 1
    x = L.GlobalAveragePooling2D()(x)
    dense = [l for l in lay if isinstance(l, L.Dense)][0]
    o = L.Dense(1, name="head")(x); W["head"] = dense.get_weights()
    m2 = tf.keras.Model(i, o)
    for n, w in W.items():
        m2.get_layer(n).set_weights(w)
    return m2


def quant_diag(tf, model, rep, out_json):
    """D10/R1: chẩn đoán lượng tử (fold 0): (a) sai số từng lớp của PTQ int8 (QuantizationDebugger);
    (b) phân tán biên độ theo kênh của đầu ra các lớp trước khi lượng tử; (c) PTQ int8 giữ stem + đầu vào ở float."""
    import io, pandas as _pd
    res = {}
    def conv(float_io):
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([rep[k:k + 1]] for k in range(len(rep)))
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if not float_io:
            c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
        return c
    try:
        dbg = tf.lite.experimental.QuantizationDebugger(converter=conv(True),
              debug_dataset=lambda: ([rep[k:k + 1]] for k in range(min(100, len(rep)))))
        dbg.run(); buf = io.StringIO(); dbg.layer_statistics_dump(buf); buf.seek(0)
        df = _pd.read_csv(buf)
        df["rmse_over_scale"] = np.sqrt(df["mean_squared_error"]) / df["scale"]
        res["layers"] = df[["op_name", "tensor_name", "scale", "mean_squared_error", "max_abs_error", "rmse_over_scale"]].to_dict("records")
        first_conv = df[df.op_name == "CONV_2D"].tensor_name.iloc[0]
        opts = tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=[first_conv])
        dbg2 = tf.lite.experimental.QuantizationDebugger(converter=conv(True),
               debug_dataset=lambda: ([rep[k:k + 1]] for k in range(min(100, len(rep)))), debug_options=opts)
        res["stem_float_model"] = True
        sel_model = dbg2.get_nondebug_quantized_model()
    except Exception as e:  # noqa: BLE001
        res["debugger_error"] = repr(e)[:300]; sel_model = None
    try:
        L = tf.keras.layers
        tgt = [l for l in model.layers if isinstance(l, (L.BatchNormalization, L.Conv2D, L.DepthwiseConv2D, L.ReLU, L.Add))]
        sub = tf.keras.Model(model.input, [l.output for l in tgt])
        outs = sub.predict(rep, batch_size=256, verbose=0)
        rng_stats = []
        for l, o in zip(tgt, outs):
            m = np.abs(o).reshape(-1, o.shape[-1]).max(0)
            rng_stats.append(dict(layer=l.name, type=type(l).__name__, ch=int(o.shape[-1]),
                                  max_over_median=float(m.max() / (np.median(m) + 1e-9)), tensor_max=float(m.max())))
        res["channel_ranges"] = rng_stats
    except Exception as e:  # noqa: BLE001
        res["range_error"] = repr(e)[:300]
    pathlib.Path(out_json).write_text(json.dumps(res, indent=1, default=float))
    return (b"F16x8" + sel_model) if sel_model is not None else None


class FloatIORunner:
    def __init__(self, tf, b, bs=256):
        self.it = tf.lite.Interpreter(model_content=b); self.bs = bs
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.it.resize_tensor_input(self.i["index"], [bs] + list(self.i["shape"][1:])); self.it.allocate_tensors()
    def __call__(self, Xn):
        out = np.zeros(len(Xn), np.float32)
        for k in range(0, len(Xn), self.bs):
            xb = Xn[k:k + self.bs].astype(np.float32); m = len(xb)
            if m < self.bs:
                xb = np.concatenate([xb, np.zeros((self.bs - m,) + xb.shape[1:], np.float32)])
            self.it.set_tensor(self.i["index"], xb); self.it.invoke()
            out[k:k + m] = self.it.get_tensor(self.o["index"]).ravel()[:m]
        return out


class Int8Runner:
    def __init__(self, tf, b, bs=256):
        self.it = tf.lite.Interpreter(model_content=b); self.bs = bs
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.it.resize_tensor_input(self.i["index"], [bs] + list(self.i["shape"][1:])); self.it.allocate_tensors()
        self.si, self.zi = self.i["quantization"]; self.so, self.zo = self.o["quantization"]
    def __call__(self, Xn):
        out = np.zeros(len(Xn), np.float32)
        for k in range(0, len(Xn), self.bs):
            xb = Xn[k:k + self.bs]; m = len(xb)
            q = np.clip(np.round(xb / self.si + self.zi), -128, 127).astype(np.int8)
            if m < self.bs:
                q = np.concatenate([q, np.zeros((self.bs - m,) + q.shape[1:], np.int8)])
            self.it.set_tensor(self.i["index"], q); self.it.invoke()
            out[k:k + m] = (self.it.get_tensor(self.o["index"]).ravel()[:m].astype(np.float32) - self.zo) * self.so
        return out


def patient_logit(lg_w, owner, ids):
    p = 1 / (1 + np.exp(-lg_w))
    pp = np.array([p[owner == k].mean() if (owner == k).any() else 0.5 for k in ids])
    return np.log(np.clip(pp, 1e-6, 1 - 1e-6) / np.clip(1 - pp, 1e-6, 1))


def temp_scale(lv, y):
    from scipy.optimize import minimize_scalar
    def nll(T):
        p = np.clip(1 / (1 + np.exp(-lv / T)), 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


# ---------------------------------------------------------------- một bộ (train, val, test) → mọi biến thể
def train_variants(tf, tfmot, Xtr6, ytr, Xva6, yva, mode, a, rng):
    Xva = to_input(Xva6, mode)
    sub = rng.choice(len(Xtr6), min(len(Xtr6), 20000), replace=False)
    Xs = to_input(Xtr6[sub], mode); mu = Xs.reshape(-1, Xs.shape[-1]).mean(0); sd = Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-6
    rep = (to_input(Xtr6[rng.choice(len(Xtr6), 300, replace=False)], mode) - mu) / sd
    models, meta = {}, {}
    for base, sr in ((("fp32", False),) if getattr(a, "no_sr", False) else (("fp32", False), ("fp32_sr", True))):
        tf.keras.backend.clear_session(); gc.collect()
        m = ARCH[a.arch](Xva.shape[-1], tf)
        meta[base + "_epochs"] = fit(tf, m, Xtr6, ytr, Xva, yva, mode, mu, sd, sr, a.epochs, 1e-3, rng)
        models[base] = m
        models["ptq" + base[4:]] = to_tflite(tf, m, rep)
        if a.extra and base == "fp32":
            # E1: PTQ hiệu chuẩn 2000 cửa sổ ngẫu nhiên từ tập train (thay vì 300)
            rep2k = (to_input(Xtr6[rng.choice(len(Xtr6), min(2000, len(Xtr6)), replace=False)], mode) - mu) / sd
            models["ptq_c2k"] = to_tflite(tf, m, rep2k)
            try:
                models["ptq_a16"] = to_tflite16(tf, m, rep)
            except Exception as e:   # noqa: BLE001
                print("16x8 conversion failed:", repr(e)[:200], flush=True)
            # E2: đối chứng fine-tune FP32 cùng lịch với QAT (lr 1e-4, qat_epochs, patience 3), không fake-quant
            ft = tf.keras.models.clone_model(m); ft.set_weights(m.get_weights())
            meta["fp32_ft_epochs"] = fit(tf, ft, Xtr6, ytr, Xva, yva, mode, mu, sd, sr, a.qat_epochs, 1e-4, rng, patience=3)
            models["fp32_ft"] = ft; models["ptq_ft"] = to_tflite(tf, ft, rep)
            if getattr(a, "r2", False):
                try:
                    mf = fold_bn_cle(tf, m, cle=False); mc = fold_bn_cle(tf, m, cle=True)
                    tst = rep[:64]; e1 = float(np.abs(mf.predict(tst, verbose=0) - m.predict(tst, verbose=0)).max())
                    e2 = float(np.abs(mc.predict(tst, verbose=0) - m.predict(tst, verbose=0)).max())
                    meta["cle_check_maxdiff"] = [e1, e2]
                    models["ptq_fold"] = to_tflite(tf, mf, rep); models["ptq_cle"] = to_tflite(tf, mc, rep)
                except Exception as e:  # noqa: BLE001
                    meta["cle_error"] = repr(e)[:300]
                for d in range(getattr(a, "calib_draws", 0)):
                    rd = np.random.default_rng(50000 + 100 * getattr(a, "_fold", 0) + d)
                    models[f"ptqd300_{d}"] = to_tflite(tf, m, (to_input(Xtr6[rd.choice(len(Xtr6), 300, replace=False)], mode) - mu) / sd)
                    models[f"ptqd2k_{d}"] = to_tflite(tf, m, (to_input(Xtr6[rd.choice(len(Xtr6), 2000, replace=False)], mode) - mu) / sd)
            if getattr(a, "diag", False) and getattr(a, "_fold", -1) == 0:
                sm = quant_diag(tf, m, rep, str(pathlib.Path(a.out) / "quant_diag_fold0.json"))
                if sm is not None:
                    models["ptq_stemf"] = sm
        q = tfmot.quantization.keras.quantize_model(m)
        meta["qat" + base[4:] + "_epochs"] = fit(tf, q, Xtr6, ytr, Xva, yva, mode, mu, sd, sr, a.qat_epochs, 1e-4, rng, patience=3)
        models["qat" + base[4:]] = to_tflite(tf, q, rep)
    meta["params"] = int(models["fp32"].count_params())
    meta["tflite_bytes"] = {k: len(v) for k, v in models.items() if isinstance(v, bytes)}
    return models, (mu, sd), meta


def predict_fn(tf, models):
    fns = {}
    for k, v in models.items():
        if isinstance(v, bytes) and v[:5] == b"F16x8":
            fns[k] = FloatIORunner(tf, v[5:])
        elif isinstance(v, bytes):
            r = Int8Runner(tf, v); fns[k] = r
        else:
            fns[k] = (lambda mm: (lambda Xn: mm.predict(Xn, batch_size=1024, verbose=0).ravel()))(v)
    return fns


def evaluate(fns, calib, W6_by_pat, ids, y, mode, mu, sd):
    owner = np.concatenate([np.full(len(W6_by_pat[k]), k) for k in ids])
    Xn = (to_input(np.concatenate([W6_by_pat[k] for k in ids]), mode) - mu) / sd
    out = {}
    for name, f in fns.items():
        lg = patient_logit(f(Xn), owner, ids)
        T, thr = calib[name]; p = 1 / (1 + np.exp(-lg / T))
        out[name] = p
    res = {}
    for name, p in out.items():
        T, thr = calib[name]; yh = (p >= thr).astype(int); ref = (out["fp32"] >= calib["fp32"][1]).astype(int)
        refb = "fp32_sr" if name.endswith("_sr") else "fp32"
        res[name] = dict(auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                         bal_acc=float(balanced_accuracy_score(y, yh)), ece=ece(y, p),
                         sens=float(np.mean(yh[y == 1] == 1)) if (y == 1).any() else np.nan,
                         spec=float(np.mean(yh[y == 0] == 0)) if (y == 0).any() else np.nan,
                         brier=float(np.mean((p - y) ** 2)),
                         agree_fp32=float(np.mean(yh == (out[refb] >= calib[refb][1]).astype(int))))
    return res, out


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default=str(ROOT / "reports/p56"))
    ap.add_argument("--task", default="PD_vs_HC", choices=["PD_vs_HC", "PD_vs_DD"])
    ap.add_argument("--inputs", nargs="+", default=["raw", "inv"]); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat_epochs", type=int, default=6); ap.add_argument("--monipar", action="store_true")
    ap.add_argument("--monipar_inputs", nargs="+", default=["acc", "invacc"])
    ap.add_argument("--skip_cv", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arch", default="ds_tcn", choices=["ds_tcn", "cnn"]); ap.add_argument("--extra", action="store_true"); ap.add_argument("--diag", action="store_true"); ap.add_argument("--r2", action="store_true"); ap.add_argument("--calib_draws", type=int, default=0); ap.add_argument("--rep_inits", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    a = ap.parse_args()
    global CAUSAL
    CAUSAL = a.causal
    import tensorflow as tf, tensorflow_model_optimization as tfmot
    out = pathlib.Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); tf.keras.utils.set_random_seed(a.seed)
    t0 = time.time()
    P, recs = load_pads(find_pads(a.data), a.quick)
    neg = "HC" if a.task == "PD_vs_HC" else "DD"
    sel = np.where(P.label.isin(["PD", neg]))[0]; y_all = (P.label == "PD").astype(int).to_numpy()
    clean = {k: windows_for(recs[k]) for k in sel}
    info = dict(task=a.task, n=len(sel), n_pos=int(y_all[sel].sum()), n_windows=int(sum(len(v) for v in clean.values())),
                t_load_s=round(time.time() - t0, 1), tf=tf.__version__, tfmot=tfmot.__version__,
                gpus=[d.name for d in tf.config.list_physical_devices("GPU")], args=vars(a), shifts=[s for s, _ in SHIFTS])
    print(json.dumps(info, indent=1), flush=True); (out / "run_info.json").write_text(json.dumps(info, indent=1))
    rows = []
    if a.rep_inits > 0:   # R2: repeated retraining of fold 0 (fixed folds), different initializations
        mode = a.inputs[0]; a.r2 = True; a.no_sr = True; a._fold = 0
        tr, te = next(iter(StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(sel, y_all[sel])))
        tr_p, va_p = train_test_split(sel[tr], test_size=0.2, stratify=y_all[sel[tr]], random_state=a.seed); te_p = sel[te]
        Xtr6 = np.concatenate([clean[i] for i in tr_p]); ytr = np.concatenate([np.full(len(clean[i]), y_all[i]) for i in tr_p])
        Xva6 = np.concatenate([clean[i] for i in va_p]); yva = np.concatenate([np.full(len(clean[i]), y_all[i]) for i in va_p])
        for r in range(a.rep_inits):
            t1 = time.time(); tf.keras.utils.set_random_seed(7000 + r); rr = np.random.default_rng(a.seed)   # same data order/rep, new init
            models, (mu, sd), meta = train_variants(tf, tfmot, Xtr6, ytr, Xva6, yva, mode, a, rr)
            fns = predict_fn(tf, models); calib = {n: (1.0, 0.5) for n in fns}
            _, pv = evaluate(fns, calib, clean, va_p, y_all[va_p], mode, mu, sd)
            for n in fns:
                lg = np.log(np.clip(pv[n], 1e-6, 1 - 1e-6) / np.clip(1 - pv[n], 1e-6, 1))
                T = temp_scale(lg, y_all[va_p]); calib[n] = (T, youden_threshold(y_all[va_p], 1 / (1 + np.exp(-lg / T))))
            res, _ = evaluate(fns, calib, clean, te_p, y_all[te_p], mode, mu, sd)
            for n, v in res.items():
                rows.append(dict(rep=r, variant=n, **v))
            pd.DataFrame(rows).to_csv(out / "rep_results.csv", index=False)
            print("rep", r, round(time.time() - t1), json.dumps(meta)[:300], flush=True)
        a.skip_cv = True
    if not a.skip_cv:
        for mode in a.inputs:
            preds = {}
            for k, (tr, te) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(sel, y_all[sel])):
                t1 = time.time()
                tr_p, va_p = train_test_split(sel[tr], test_size=0.2, stratify=y_all[sel[tr]], random_state=a.seed)
                te_p = sel[te]
                Xtr6 = np.concatenate([clean[i] for i in tr_p]); ytr = np.concatenate([np.full(len(clean[i]), y_all[i]) for i in tr_p])
                Xva6 = np.concatenate([clean[i] for i in va_p]); yva = np.concatenate([np.full(len(clean[i]), y_all[i]) for i in va_p])
                a._fold = k
                models, (mu, sd), meta = train_variants(tf, tfmot, Xtr6, ytr, Xva6, yva, mode, a, rng)
                fns = predict_fn(tf, models)
                # hiệu chỉnh trên validation sạch
                calib = {n: (1.0, 0.5) for n in fns}
                _, pv = evaluate(fns, calib, clean, va_p, y_all[va_p], mode, mu, sd)
                for n in fns:
                    lg = np.log(np.clip(pv[n], 1e-6, 1 - 1e-6) / np.clip(1 - pv[n], 1e-6, 1))
                    T = temp_scale(lg, y_all[va_p]); calib[n] = (T, youden_threshold(y_all[va_p], 1 / (1 + np.exp(-lg / T))))
                with open(out / "calib.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(fold=k, input=mode, calib={n: list(v) for n, v in calib.items()},
                                             val_sid=[str(x) for x in P.sid.to_numpy()[va_p]], val_y=[int(v) for v in y_all[va_p]],
                                             val_p={n: [float(v) for v in pv[n]] for n in ("fp32", "ptq", "qat", "qat_sr") if n in pv})) + "\n")
                shifts_run = SHIFTS + ([("causalin", ("causalin", 1))] if getattr(a, "r2", False) and not CAUSAL else [])
                for sname, shift in shifts_run:
                    fsub = fns if sname == "clean" else {n: f for n, f in fns.items() if not n.startswith("ptqd")}
                    W = clean if shift is None else {i: windows_for(recs[i], shift=shift, rng=np.random.default_rng(1000 + int(i))) for i in te_p}
                    res, pt = evaluate(fsub, calib, W, te_p, y_all[te_p], mode, mu, sd)
                    for n, r in res.items():
                        rows.append(dict(task=a.task, input=mode, fold=k, shift=sname, variant=n, **r))
                        preds.setdefault((sname, n), {}).update(dict(zip(P.sid.to_numpy()[te_p], pt[n])))
                preds.setdefault(("meta", "fold"), {}).update({sid: k for sid in P.sid.to_numpy()[te_p]})
                if k == 0:   # xuất cho bo
                    for n in [v for v in ("ptq", "qat_sr", "qat", "ptq_sr", "ptq_c2k", "ptq_ft", "ptq_a16") if v in models]:
                        b = models[n]; b = b[5:] if b[:5] == b"F16x8" else b
                        (out / "models" / f"{a.task}_{mode}_{n}_fold0.tflite").write_bytes(b)
                    Xb = (to_input(np.concatenate([clean[i] for i in te_p])[:: max(1, sum(len(clean[i]) for i in te_p) // 50)][:50], mode) - mu) / sd
                    np.save(out / "models" / f"bench_inputs_{a.task}_{mode}.npy", Xb.astype(np.float32))
                    np.savez(out / "models" / f"norm_{a.task}_{mode}_fold0.npz", mu=mu, sd=sd)
                import resource
                meta.update(fold=k, sec=round(time.time() - t1, 1), maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
                print(mode, json.dumps(meta), flush=True)
                with open(out / "fold_meta.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(input=mode, **meta)) + "\n")
                pd.DataFrame(rows).to_csv(out / "shift_results.csv", index=False)
            # dự đoán mức bệnh nhân gộp các fold (để kiểm định cặp DeLong/McNemar ở P7)
            pr = pd.DataFrame({f"{s}|{n}": pd.Series(v) for (s, n), v in preds.items()})
            pr["y"] = [y_all[np.where(P.sid.to_numpy() == s)[0][0]] for s in pr.index]
            pr.to_csv(out / f"patient_preds_{a.task}_{mode}.csv")
    if a.monipar:
        rows_m = run_monipar(tf, tfmot, a, P, recs, y_all, out, rng)
        pd.DataFrame(rows_m).to_csv(out / "monipar_results.csv", index=False)
    info["t_total_s"] = round(time.time() - t0, 1); (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print("xong", info["t_total_s"], flush=True)


def run_monipar(tf, tfmot, a, P, recs, y_all, out, rng):
    from src.monipar_io import load as load_monipar
    M = load_monipar(a.data)
    sub = sorted({r["subject"] for r in M}); y_m = np.array([1 if s.split("_")[0] in ("supervised", "remote") else 0 for s in sub])
    Wm = {}
    for r in M:
        xf = bandpass(r["x"], FS)                           # loại trọng lực + nhiễu thấp tần (giống PADS)
        aw = np.concatenate([xf, np.zeros_like(xf)])        # không có gyro
        n = (aw.shape[1] - WIN) // HOP + 1
        if n > 0:
            Wm.setdefault(r["subject"], []).append(np.stack([aw[:, s * HOP:s * HOP + WIN].T for s in range(n)]))
    Wm = {k: np.concatenate(v).astype(np.float32) for k, v in Wm.items()}
    sel = np.where(P.label.isin(["PD", "HC"]))[0]
    pads_w = {k: windows_for(recs[k], task_filter=set(MATCHED)) for k in sel}
    tr_p, va_p = train_test_split(sel, test_size=0.2, stratify=y_all[sel], random_state=a.seed)
    Xtr6 = np.concatenate([pads_w[i] for i in tr_p]); ytr = np.concatenate([np.full(len(pads_w[i]), y_all[i]) for i in tr_p])
    Xva6 = np.concatenate([pads_w[i] for i in va_p]); yva = np.concatenate([np.full(len(pads_w[i]), y_all[i]) for i in va_p])
    rows = []
    for mode in a.monipar_inputs:
        models, (mu, sd), meta = train_variants(tf, tfmot, Xtr6, ytr, Xva6, yva, mode, a, rng)
        fns = predict_fn(tf, models)
        calib = {n: (1.0, 0.5) for n in fns}
        _, pv = evaluate(fns, calib, pads_w, va_p, y_all[va_p], mode, mu, sd)
        for n in fns:
            lg = np.log(np.clip(pv[n], 1e-6, 1 - 1e-6) / np.clip(1 - pv[n], 1e-6, 1))
            T = temp_scale(lg, y_all[va_p]); calib[n] = (T, youden_threshold(y_all[va_p], 1 / (1 + np.exp(-lg / T))))
        idx = list(range(len(sub))); Wi = {i: Wm[s] for i, s in enumerate(sub)}
        res, pt = evaluate(fns, calib, Wi, idx, y_m, mode, mu, sd)
        for n, r in res.items():
            ci = bootstrap_ci(y_m, pt[n], thr=calib[n][1], n_boot=2000)
            rows.append(dict(input=mode, variant=n, n=len(sub), n_pd=int(y_m.sum()), **r,
                             auroc_lo=ci["auroc"][0], auroc_hi=ci["auroc"][1], bal_acc_lo=ci["bal_acc"][0], bal_acc_hi=ci["bal_acc"][1]))
        pd.DataFrame({n: pt[n] for n in pt} | {"subject": sub, "y": y_m}).to_csv(out / f"monipar_preds_{mode}.csv", index=False)
        for n in ("ptq", "qat_sr"):
            (out / "models" / f"monipar_{mode}_{n}.tflite").write_bytes(models[n])
        print("monipar", mode, json.dumps(meta), flush=True)
    return rows


if __name__ == "__main__":
    main()
