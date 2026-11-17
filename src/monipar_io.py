"""Đọc Monipar (Zenodo 8104853) cho kiểm định ngoài.
Mỗi file .mat có một mảng cell (người × tuần); mỗi lượt là ma trận (N, 5):
  cột 0 thời gian [ms], cột 1–3 gia tốc xyz [m/s², CÓ trọng lực], cột 4 nhãn bài tập theo mẫu
  (0 chuyển tư thế, 1 nghỉ, 2 run tư thế, 3 tay lên ngực, 4 gõ ngón, 5 mở-nắm tay, 6 sấp-ngửa, 7 đứng dậy, 8 đi).
Trả về các đoạn liên tục cùng nhãn, đã: đổi sang g (/9.81), nội suy theo mốc ms lên fs đều."""
import pathlib
import numpy as np

G = 9.81
FILES = {"supervised": ("MONIPAR_PD_SUPERVISED.mat", "SUPERVISED_RAWDATA", "PD"),
         "remote": ("MONIPAR_PD_REMOTE.mat", "REMOTE_RAWDATA", "PD"),
         "healthy": ("MONIPAR_HEALTHYCONTROL.mat", "HEALTHYCONTROL_RAWDATA", "HC")}
TASKS = {1: "rest", 2: "postural", 3: "hand_to_chest", 4: "finger_tap", 5: "open_close", 6: "pronation", 7: "sit_to_stand", 8: "gait"}
# ánh xạ gần đúng sang bài PADS (ghi rõ trong bài báo)
TO_PADS = {"rest": "Relaxed", "postural": "StretchHold", "hand_to_chest": "TouchNose"}


def find_dir(root):
    root = pathlib.Path(root)
    for f in root.rglob(FILES["supervised"][0]):
        return f.parent
    raise SystemExit(f"không thấy Monipar dưới {root}")


def segments(arr, fs=50.0, min_sec=4.0):
    t = arr[:, 0] / 1000.0; lab = arr[:, 4].astype(int)
    ok = np.r_[True, np.diff(t) > 0]
    t, a, lab = t[ok], arr[ok, 1:4] / G, lab[ok]
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(lab)]):
        if lab[s] not in TASKS or t[e - 1] - t[s] < min_sec:
            continue
        grid = np.arange(t[s], t[e - 1], 1.0 / fs)
        x = np.stack([np.interp(grid, t[s:e], a[s:e, c]) for c in range(3)]).astype(np.float32)
        yield TASKS[lab[s]], x


def load(root, fs=50.0, tasks=("rest", "postural", "hand_to_chest")):
    """→ list dict(subject, group, label, week, task, x(3,T) [g, có trọng lực])"""
    import scipy.io as sio
    d = find_dir(root); out = []
    for group, (fn, key, label) in FILES.items():
        D = sio.loadmat(d / fn)[key]
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i, j].size == 0:
                    continue
                for task, x in segments(np.asarray(D[i, j], dtype=np.float64), fs):
                    if task in tasks:
                        out.append(dict(subject=f"{group}_{i + 1}", group=group, label=label, week=j + 1, task=task, x=x))
    return out
