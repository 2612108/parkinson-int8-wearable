"""Đặc trưng thủ công cho baseline RF/XGB. Input một đoạn tín hiệu x: (C, T), fs Hz."""
import numpy as np
from scipy import signal, stats

BANDS = [(0.5, 3), (3, 7), (7, 12), (12, 20)]  # 3–7 Hz: run nghỉ/tư thế PD


def bandpass(x, fs, lo=0.5, hi=20.0, order=4):
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return signal.resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=-1)


def channel_features(x, fs):
    """x: (T,) → vector đặc trưng."""
    f, P = signal.welch(x, fs=fs, nperseg=min(len(x), 256))
    tot = np.trapezoid(P, f) + 1e-12
    bp = [np.trapezoid(P[(f >= a) & (f < b)], f[(f >= a) & (f < b)]) / tot for a, b in BANDS]
    m = (f >= 0.5) & (f <= 20)
    pk = f[m][np.argmax(P[m])] if m.any() else 0.0
    pn = P[m] / (P[m].sum() + 1e-12)
    sent = -np.sum(pn * np.log(pn + 1e-12))
    d = np.diff(x)
    return np.r_[np.std(x), stats.iqr(x), stats.skew(x), stats.kurtosis(x), np.sqrt(np.mean(d ** 2)),
                 np.log(tot), pk, sent, bp]


FEAT_NAMES = ["std", "iqr", "skew", "kurt", "rms_diff", "log_pow", "peak_f", "spec_ent"] + [f"bp_{a}_{b}" for a, b in BANDS]


def segment_features(seg, fs, acc_idx=(0, 1, 2), gyro_idx=(3, 4, 5)):
    """seg: (6, T) acc+gyro (hoặc (3,T) chỉ acc). Thêm kênh độ lớn vector (bất biến hướng)."""
    chans = [seg[i] for i in range(seg.shape[0])]
    chans.append(np.linalg.norm(seg[list(acc_idx)], axis=0))
    if seg.shape[0] >= 6:
        chans.append(np.linalg.norm(seg[list(gyro_idx)], axis=0))
    return np.concatenate([channel_features(c, fs) for c in chans])
