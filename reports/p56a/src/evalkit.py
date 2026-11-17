"""Đánh giá không rò rỉ ở mức bệnh nhân: nested CV, chỉ số, bootstrap. Dùng chung P3–P7."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def ece(y, p, n_bins=10):
    """Expected calibration error cho bài toán nhị phân (p = P(y=1))."""
    y, p = np.asarray(y), np.asarray(p)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(bal_acc=balanced_accuracy_score(y, yh),
                auroc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                ece=ece(y, p), brier=brier_score_loss(y, p), n=len(y))


def bootstrap_ci(y, p, thr=0.5, n_boot=2000, seed=0, keys=("bal_acc", "auroc", "sens", "spec", "ece")):
    """CI 95% bằng bootstrap theo bệnh nhân (mỗi hàng = một bệnh nhân), phân tầng theo lớp."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        m = metrics(y[idx], p[idx], thr)
        for k in keys:
            out[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def youden_threshold(y, p):
    """Ngưỡng tối đa balanced accuracy — CHỈ gọi trên dự đoán OOF của tập train."""
    cands = np.unique(np.r_[0.5, p])
    best = max(cands, key=lambda t: balanced_accuracy_score(y, (p >= t).astype(int)))
    return float(best)


def outer_folds(y, n_splits=5, seed=0):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
