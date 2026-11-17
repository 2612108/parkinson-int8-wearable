"""Sinh dữ liệu C và huấn luyện mô hình Random Forest cho B4 (firmware/b4_rf):
1) Import hàm từ scripts/p3_baseline.py (không sửa scripts/ hay src/), dựng ma trận đặc trưng PD vs HC (355 bệnh nhân x 2112 đặc trưng).
2) Chia fold 0 giống P3: StratifiedKFold(5, shuffle=True, random_state=0).
3) Huấn luyện RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=0) trên train fold 0.
4) Ghi kết quả đánh giá test fold 0 (AUROC, balanced accuracy, ...) vào reports/board/B4_rf_python.json.
5) Chuyển RF sang C (float) bằng emlearn (cả chế độ leaf_bits=8 xác suất mềm sát sklearn và leaf_bits=0 biểu quyết đa số).
6) Xuất 20 vector đặc trưng test fold 0 + nhãn/xác suất sklearn tham chiếu + 1 bản ghi PADS thật 10 s x 6 kênh @ 100 Hz
   kèm 96 đặc trưng tham chiếu (12 đặc trưng/kênh x 8 kênh sau hạ mẫu 100->50 Hz và lọc nhân quả Butterworth bậc 4 0.5-20 Hz).
"""
import json, pathlib, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_pydeps = ROOT / "firmware" / ".pydeps"
if _pydeps.is_dir() and str(_pydeps) not in sys.path:
    sys.path.insert(0, str(_pydeps))

import scripts.p3_baseline as p3
from src.features import FEAT_NAMES, BANDS, resample, segment_features
from src.pads_io import load_txt
import emlearn


def extract_subject_rec_only(base, sid):
    rec_feats = {}
    for task, wrist, f in p3.subject_records(base, sid):
        if not f.exists():
            continue
        x = p3.prep(p3.load_txt(f))
        rec_feats[(task, wrist)] = p3.segment_features(x, p3.FS)
    return sid, rec_feats


def load_or_build_pd_hc_matrix(cache_path: pathlib.Path):
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["Xp"].astype(np.float32), d["y"].astype(np.int32), d["sids"].tolist(), d["keys"].tolist(), float(d["t_build_s"])

    t0 = time.time()
    base = p3.find_root(ROOT / "data/raw/pads")
    pats = []
    for pj in sorted((base / "patients").glob("patient_*.json")):
        d = json.loads(pj.read_text())
        pats.append(dict(
            sid=str(d["id"]).zfill(3),
            condition=d.get("condition"),
            label=p3.label_of(d.get("condition")),
            age=d.get("age"),
            gender=d.get("gender"),
        ))
    P = pd.DataFrame(pats)
    P = P[[(base / "movement" / f"observation_{s}.json").exists() for s in P.sid]].reset_index(drop=True)
    idx = P.label.isin(("PD", "HC")).to_numpy()
    P_sub = P[idx].reset_index(drop=True)
    y = (P_sub.label == "PD").astype(np.int32).to_numpy()
    sids = P_sub.sid.tolist()

    res = Parallel(n_jobs=-1, verbose=0)(delayed(extract_subject_rec_only)(base, s) for s in sids)
    keys = sorted({k for _, rf in res for k in rf})
    nf = len(FEAT_NAMES) * 8
    Xp = np.full((len(res), len(keys) * nf), np.nan, np.float32)
    for i, (sid, rf) in enumerate(res):
        for j, k in enumerate(keys):
            if k in rf:
                Xp[i, j * nf:(j + 1) * nf] = rf[k]
    t_build = time.time() - t0
    key_strs = [f"{t}|{w}" for t, w in keys]
    np.savez(cache_path, Xp=Xp, y=y, sids=np.array(sids), keys=np.array(key_strs), t_build_s=t_build)
    return Xp, y, sids, key_strs, t_build


def main():
    out_dir = ROOT / "firmware/b4_rf/main/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    board_dir = ROOT / "reports/board"
    board_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "p3_pd_hc_features.npz"
    print("[B4] Building/loading PADS PD vs HC feature matrix via scripts/p3_baseline.py...", flush=True)
    Xp, y, sids, keys, t_build = load_or_build_pd_hc_matrix(cache_path)
    print(f"[B4] Matrix shape={Xp.shape}, n_pos={int(y.sum())}, n_neg={int((y==0).sum())}, build_time={t_build:.1f}s", flush=True)

    # Chia fold 0 giống P3: StratifiedKFold(5, shuffle=True, random_state=0)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    tr, te = next(iter(skf.split(Xp, y)))

    if np.isnan(Xp).any():
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Xtr = imp.fit_transform(Xp[tr]).astype(np.float32)
        Xte = imp.transform(Xp[te]).astype(np.float32)
    else:
        Xtr = Xp[tr].astype(np.float32)
        Xte = Xp[te].astype(np.float32)
    ytr, yte = y[tr], y[te]
    sids_te = [sids[i] for i in te]

    # Huấn luyện RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=0)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(Xtr, ytr)

    prob_te = clf.predict_proba(Xte)[:, 1].astype(np.float64)
    pred_te = clf.predict(Xte).astype(np.int32)
    auroc_fold0 = float(roc_auc_score(yte, prob_te))
    bal_acc_fold0 = float(balanced_accuracy_score(yte, pred_te))
    acc_fold0 = float(accuracy_score(yte, pred_te))
    sens_fold0 = float(((pred_te == 1) & (yte == 1)).sum() / max((yte == 1).sum(), 1))
    spec_fold0 = float(((pred_te == 0) & (yte == 0)).sum() / max((yte == 0).sum(), 1))

    total_nodes = int(sum(est.tree_.node_count for est in clf.estimators_))
    total_leaves = int(sum(est.tree_.n_leaves for est in clf.estimators_))
    max_tree_depth = int(max(est.tree_.max_depth for est in clf.estimators_))

    # Chuyển sang C bằng emlearn (float)
    # 1. leaf_bits=8 (soft probability voting, khớp sát xác suất sklearn.predict_proba)
    cmodel_soft = emlearn.convert(clf, method="inline", dtype="float", leaf_bits=8)
    cmodel_soft.save(name="pd_rf", file=str(out_dir / "pd_rf.h"), include_proba=True)

    # 2. leaf_bits=0 (majority hard voting mặc định của emlearn)
    cmodel_hard = emlearn.convert(clf, method="inline", dtype="float", leaf_bits=0)
    cmodel_hard.save(name="pd_rf_hard", file=str(out_dir / "pd_rf_hard.h"), include_proba=True)

    # Tính đầu ra dự kiến của emlearn (trên toàn tập test fold 0) để lưu vào B4_rf_python.json
    # Trong sklearn >= 1.4, tree_.value[node, 0] đã chuẩn hóa tổng = 1
    eml_soft_probs_te = np.zeros((len(Xte), 2), dtype=np.float32)
    eml_hard_votes_te = np.zeros((len(Xte), 2), dtype=np.float32)
    for est in clf.estimators_:
        leaf_ids = est.apply(Xte)
        vals = est.tree_.value[leaf_ids, 0, :]  # (n_te, 2) in [0, 1]
        # emlearn leaf_bits=8 quantizes each leaf prob to uint8(round(p*254)/254*255) = round(p*254)*255/254
        q_uint8 = (np.round(vals * 254.0) / 254.0 * 255.0).astype(np.uint8)
        eml_soft_probs_te += q_uint8.astype(np.float32) / np.float32(255.0)
        hard_cls = np.argmax(vals, axis=1)
        for i, c in enumerate(hard_cls):
            eml_hard_votes_te[i, c] += 1.0
    eml_soft_probs_te /= np.float32(len(clf.estimators_))
    eml_hard_probs_te = eml_hard_votes_te / np.float32(len(clf.estimators_))
    eml_soft_pred_te = np.argmax(eml_soft_probs_te, axis=1).astype(np.int32)
    eml_hard_pred_te = np.argmax(eml_hard_votes_te, axis=1).astype(np.int32)

    auroc_eml_soft = float(roc_auc_score(yte, eml_soft_probs_te[:, 1]))
    auroc_eml_hard = float(roc_auc_score(yte, eml_hard_probs_te[:, 1]))

    rf_json = dict(
        task="PD_vs_HC",
        cv_split="StratifiedKFold(n_splits=5, shuffle=True, random_state=0), fold=0",
        model="RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=0)",
        n_patients_total=int(len(Xp)),
        n_train_fold0=int(len(Xtr)),
        n_test_fold0=int(len(Xte)),
        n_test_pd=int((yte == 1).sum()),
        n_test_hc=int((yte == 0).sum()),
        n_records_per_patient=int(len(keys)),
        n_features_per_record=int(len(FEAT_NAMES) * 8),
        n_features_total=int(Xp.shape[1]),
        total_tree_nodes=total_nodes,
        total_tree_leaves=total_leaves,
        max_tree_depth_actual=max_tree_depth,
        auroc_test_fold0_sklearn=round(auroc_fold0, 6),
        auroc_test_fold0_emlearn_soft=round(auroc_eml_soft, 6),
        auroc_test_fold0_emlearn_hard=round(auroc_eml_hard, 6),
        bal_acc_test_fold0_sklearn=round(bal_acc_fold0, 6),
        acc_test_fold0_sklearn=round(acc_fold0, 6),
        sens_test_fold0_sklearn=round(sens_fold0, 6),
        spec_test_fold0_sklearn=round(spec_fold0, 6),
        label_match_20_soft=f"{int(np.sum(eml_soft_pred_te[:20] == pred_te[:20]))}/20",
        label_match_20_hard=f"{int(np.sum(eml_hard_pred_te[:20] == pred_te[:20]))}/20",
        max_abs_prob_diff_20_soft=float(np.max(np.abs(eml_soft_probs_te[:20, 1] - prob_te[:20]))),
        max_abs_prob_diff_20_hard=float(np.max(np.abs(eml_hard_probs_te[:20, 1] - prob_te[:20]))),
        feature_build_time_s=round(t_build, 2),
    )
    (board_dir / "B4_rf_python.json").write_text(json.dumps(rf_json, indent=2, ensure_ascii=False) + "\n")
    print("[B4] Saved reports/board/B4_rf_python.json:", json.dumps(rf_json, indent=2), flush=True)

    # Nhúng 20 vector đặc trưng test đầu tiên của fold 0
    n_embed = 20
    X_embed = Xte[:n_embed].astype(np.float32)
    y_true_embed = yte[:n_embed].astype(np.int32)
    y_sk_embed = pred_te[:n_embed].astype(np.int32)
    p_sk_embed = prob_te[:n_embed].astype(np.float32)
    p_eml_soft_embed = eml_soft_probs_te[:n_embed, 1].astype(np.float32)
    p_eml_hard_embed = eml_hard_probs_te[:n_embed, 1].astype(np.float32)
    sids_embed = sids_te[:n_embed]

    # Phần (3): Bản ghi PADS thật 10 s x 6 kênh @ 100 Hz (1000 mẫu x 6 kênh)
    rec_rel = "data/raw/pads/movement/timeseries/005_HoldWeight_LeftWrist.txt"
    x_full = load_txt(ROOT / rec_rel)
    raw_10s = x_full[:, :1000].astype(np.float32)  # (6, 1000) = 10.0 s @ 100 Hz
    assert raw_10s.shape == (6, 1000)

    h_fir = signal.firwin(41, 0.5, window=("kaiser", 5.0)).astype(np.float32)
    sos_f64 = signal.butter(4, [0.5, 20.0], btype="band", fs=50.0, output="sos")
    sos_f32 = sos_f64.astype(np.float32)

    # Đường ống Python: hạ mẫu 100->50 Hz (resample_poly) + lọc nhân quả Butterworth bậc 4 0.5-20 Hz (sosfilt) + segment_features
    r_10s = resample(raw_10s, 100, 50).astype(np.float32)
    xf_10s_causal = signal.sosfilt(sos_f32, r_10s, axis=-1).astype(np.float32)
    ref_feats_96_causal = segment_features(xf_10s_causal, 50.0).astype(np.float32)
    assert ref_feats_96_causal.shape == (96,)

    # Kèm tham chiếu zero-phase trên cùng bản ghi 10 s để thông tin đầy đủ
    xf_10s_zp = signal.sosfiltfilt(sos_f64, r_10s, axis=-1).astype(np.float32)
    ref_feats_96_zp = segment_features(xf_10s_zp, 50.0).astype(np.float32)

    ch_names = ["ax", "ay", "az", "wx", "wy", "wz", "acc_mag", "gyro_mag"]
    feat_96_names = [f"{ch}_{fn}" for ch in ch_names for fn in FEAT_NAMES]

    h_txt = f"""#pragma once
#include <stdint.h>

#define B4_N_TEST 20
#define B4_N_FEAT {Xp.shape[1]}
#define B4_N_TREES 100
#define B4_MAX_DEPTH 8

#define B4_REC_RAW_C 6
#define B4_REC_RAW_T 1000
#define B4_REC_DECIM_T 500
#define B4_REC_OUT_C 8
#define B4_FEATS_PER_CH 12
#define B4_REC_N_FEATS 96
#define B4_FIR_LEN 41
#define B4_SOS_SECTIONS 4

extern const char* const g_b4_test_sid[B4_N_TEST];
extern const int32_t g_b4_test_y_true[B4_N_TEST];
extern const int32_t g_b4_test_y_sklearn[B4_N_TEST];
extern const float g_b4_test_prob_sklearn[B4_N_TEST];
extern const float g_b4_test_prob_eml_soft[B4_N_TEST];
extern const float g_b4_test_prob_eml_hard[B4_N_TEST];
extern const float g_b4_test_X[B4_N_TEST][B4_N_FEAT];

extern const char* const g_b4_rec_file;
extern const float g_b4_rec_raw_10s[B4_REC_RAW_C][B4_REC_RAW_T];
extern const float g_b4_fir_kaiser41[B4_FIR_LEN];
extern const float g_b4_butter4_bp_sos[B4_SOS_SECTIONS][6];
extern const float g_b4_ref_feats96_causal[B4_REC_N_FEATS];
extern const float g_b4_ref_feats96_zp[B4_REC_N_FEATS];
extern const char* const g_b4_feat96_names[B4_REC_N_FEATS];
"""
    (out_dir / "b4_data.h").write_text(h_txt)

    def fmt_f_arr(arr):
        return ", ".join(f"{float(v):.9e}f" for v in np.asarray(arr).ravel())

    cc = ['#include "b4_data.h"\n']
    cc.append("const char* const g_b4_test_sid[B4_N_TEST] = { " +
              ", ".join(f'"{s}"' for s in sids_embed) + " };\n")
    cc.append("const int32_t g_b4_test_y_true[B4_N_TEST] = { " +
              ", ".join(str(int(v)) for v in y_true_embed) + " };\n")
    cc.append("const int32_t g_b4_test_y_sklearn[B4_N_TEST] = { " +
              ", ".join(str(int(v)) for v in y_sk_embed) + " };\n")
    cc.append("const float g_b4_test_prob_sklearn[B4_N_TEST] = { " +
              ", ".join(f"{float(v):.8f}f" for v in p_sk_embed) + " };\n")
    cc.append("const float g_b4_test_prob_eml_soft[B4_N_TEST] = { " +
              ", ".join(f"{float(v):.8f}f" for v in p_eml_soft_embed) + " };\n")
    cc.append("const float g_b4_test_prob_eml_hard[B4_N_TEST] = { " +
              ", ".join(f"{float(v):.8f}f" for v in p_eml_hard_embed) + " };\n")

    cc.append("alignas(16) const float g_b4_test_X[B4_N_TEST][B4_N_FEAT] = {\n")
    for i in range(n_embed):
        cc.append(f"  {{ {fmt_f_arr(X_embed[i])} }},\n")
    cc.append("};\n")

    cc.append(f'const char* const g_b4_rec_file = "{rec_rel} [0..10s, 1000 samples x 6ch @ 100Hz]";\n')
    cc.append(f"const float g_b4_fir_kaiser41[B4_FIR_LEN] = {{ {fmt_f_arr(h_fir)} }};\n")
    sos_rows = [f"  {{ {fmt_f_arr(sos_f32[s])} }}" for s in range(4)]
    cc.append("const float g_b4_butter4_bp_sos[B4_SOS_SECTIONS][6] = {\n" + ",\n".join(sos_rows) + "\n};\n")
    cc.append(f"const float g_b4_ref_feats96_causal[B4_REC_N_FEATS] = {{ {fmt_f_arr(ref_feats_96_causal)} }};\n")
    cc.append(f"const float g_b4_ref_feats96_zp[B4_REC_N_FEATS] = {{ {fmt_f_arr(ref_feats_96_zp)} }};\n")
    cc.append("const char* const g_b4_feat96_names[B4_REC_N_FEATS] = {\n  " +
              ",\n  ".join(f'"{nm}"' for nm in feat_96_names) + "\n};\n")

    cc.append("alignas(16) const float g_b4_rec_raw_10s[B4_REC_RAW_C][B4_REC_RAW_T] = {\n")
    for c in range(6):
        cc.append(f"  {{ {fmt_f_arr(raw_10s[c])} }},\n")
    cc.append("};\n")

    (out_dir / "b4_data.cc").write_text("".join(cc))
    print("[B4] Generated pd_rf.h, pd_rf_hard.h, and b4_data.{h,cc} successfully.", flush=True)


if __name__ == "__main__":
    main()
