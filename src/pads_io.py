"""Đọc tín hiệu PADS. Audit 15/11: file .txt có 7 cột (Time [s], acc xyz [g], gyro xyz [rad/s]);
lấy mẫu danh định 100 Hz nhưng khoảng cách mẫu dao động → nội suy tuyến tính lên lưới 100 Hz đều theo cột Time.
Gia tốc PADS KHÔNG chứa trọng lực (|acc| lúc nghỉ ≈ 0.05–0.2 g) → không ước lượng hướng trọng lực từ acc được."""
import numpy as np

FS_NOMINAL = 100.0


def load_txt(path, fs=FS_NOMINAL):
    try:
        a = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        a = np.loadtxt(path, dtype=np.float64)
    if a.shape[1] < 7:
        return a[:, :6].T.astype(np.float32)
    t = a[:, 0] - a[0, 0]
    ok = np.r_[True, np.diff(t) > 0]          # bỏ mốc thời gian trùng/lùi
    t, x = t[ok], a[ok, 1:7]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / fs)
    return np.stack([np.interp(grid, t, x[:, c]) for c in range(6)]).astype(np.float32)


def invariant_channels(a, w):
    """Kênh bất biến với phép quay cố định của đồng hồ (đeo lệch/đổi hướng), không cần trọng lực.
    a, w: (3, T) gia tốc và vận tốc góc đã lọc. Trả (6, T):
    |a|, |w|, a·w, |a×w|, |Δa|, |Δw|  (tích vô hướng/có hướng của hai vectơ cùng quay → bất biến)."""
    an, wn = np.linalg.norm(a, axis=0), np.linalg.norm(w, axis=0)
    dot = np.sum(a * w, axis=0)
    crs = np.linalg.norm(np.cross(a.T, w.T).T, axis=0)
    da = np.linalg.norm(np.diff(a, axis=1, prepend=a[:, :1]), axis=0)
    dw = np.linalg.norm(np.diff(w, axis=1, prepend=w[:, :1]), axis=0)
    return np.vstack([an, wn, dot, crs, da, dw]).astype(np.float32)
