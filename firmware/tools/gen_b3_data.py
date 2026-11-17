"""Sinh dữ liệu C cho firmware/b3_causal:
- Mô hình nhân quả: reports/p56x4/models/PD_vs_HC_raw_qat_fold0.tflite (DS-TCN INT8 huấn luyện với --causal)
- Tham số chuẩn hóa: reports/p56x4/models/norm_PD_vs_HC_raw_fold0.npz
- Hệ số FIR polyphase 100->50 Hz (41 taps Kaiser beta=5.0 khớp scipy.signal.resample_poly(x, 1, 2))
- Hệ số IIR Butterworth bậc 4 thông dải 0.5-20 Hz nhân quả (4 tầng biquad SOS ở fs=50 Hz)
- Nhúng TOÀN BỘ bản ghi PADS thật 100 Hz (005_Relaxed_LeftWrist.txt, 2046 mẫu @ 100 Hz = 20.46 s, 10 khối 2 s, 9 cửa sổ 4 s hop 2 s;
  kèm thêm 2 bản ghi đầy đủ 001_Relaxed_LeftWrist.txt [PD] và 081_Relaxed_LeftWrist.txt [HC] để đối chiếu mở rộng)
- Tính sẵn đầu ra tham chiếu của đường ống Python chuẩn:
    scipy.signal.resample_poly cả bản ghi -> scipy.signal.sosfilt cả bản ghi -> cắt cửa sổ 200 mẫu hop 100
"""
import json, pathlib, subprocess, sys
import numpy as np
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_pydeps = ROOT / "firmware" / ".pydeps"
if _pydeps.is_dir() and str(_pydeps) not in sys.path:
    sys.path.insert(0, str(_pydeps))

from src.pads_io import load_txt
from src.features import resample, bandpass
from ai_edge_litert import interpreter as _tfl


def main():
    model_path = ROOT / "reports/p56x4/models/PD_vs_HC_raw_qat_fold0.tflite"
    norm_path = ROOT / "reports/p56x4/models/norm_PD_vs_HC_raw_fold0.npz"
    bench_in_path = ROOT / "reports/p56x4/models/bench_inputs_PD_vs_HC_raw.npy"
    out_dir = ROOT / "firmware/b3_causal/main/generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gọi tflite_to_c.py để sinh model_data.{h,cc}, op_resolver.h, test_data.{h,cc}
    subprocess.run([
        sys.executable,
        str(ROOT / "firmware/tools/tflite_to_c.py"),
        "--model", str(model_path),
        "--inputs", str(bench_in_path),
        "--n", "10",
        "--out", str(out_dir),
    ], check=True)

    # 2. Đọc norm mu, sd và thông tin lượng tử hóa của mô hình p56x4
    norm = np.load(norm_path)
    mu = norm["mu"].astype(np.float32)
    sd = norm["sd"].astype(np.float32)

    mbytes = model_path.read_bytes()
    it = _tfl.Interpreter(model_content=mbytes, experimental_op_resolver_type=_tfl.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = float(ind["quantization"][0]), int(ind["quantization"][1])
    out_scale, out_zp = float(outd["quantization"][0]), int(outd["quantization"][1])

    # 3. Hệ số FIR polyphase decimation 100 -> 50 Hz (41 taps, Kaiser beta=5.0)
    h_fir = signal.firwin(41, 0.5, window=("kaiser", 5.0)).astype(np.float32)

    # 4. Hệ số IIR Butterworth bậc 4 thông dải 0.5-20 Hz @ fs=50 Hz (4 biquad sections)
    sos_f64 = signal.butter(4, [0.5, 20.0], btype="band", fs=50.0, output="sos")
    sos = sos_f64.astype(np.float32)
    assert sos.shape == (4, 6)

    def run_tflite_from_x8(x8):
        xn = (x8.astype(np.float32) - mu) / sd
        q_in = np.clip(np.round(xn / in_scale + in_zp), -128, 127).astype(np.int8)
        it.set_tensor(ind["index"], q_in[None, :, None, :])
        it.invoke()
        q_out = int(it.get_tensor(outd["index"]).ravel()[0])
        logit = float((q_out - out_zp) * out_scale)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        return q_in, q_out, logit, prob

    # 5. Đọc TOÀN BỘ bản ghi PADS thật 100 Hz
    # Bản ghi chính (Idx 0): 005_Relaxed_LeftWrist.txt (bệnh nhân PD tập test fold 0, 2046 mẫu @ 100 Hz = 20.46 s)
    # Kèm 2 bản ghi đầy đủ bổ sung (001_Relaxed_LeftWrist.txt [PD], 081_Relaxed_LeftWrist.txt [HC])
    rec_specs = [
        ("data/raw/pads/movement/timeseries/005_Relaxed_LeftWrist.txt", "PD", "005_Relaxed_LeftWrist (fold0_test PD, 20.46s)"),
        ("data/raw/pads/movement/timeseries/001_Relaxed_LeftWrist.txt", "PD", "001_Relaxed_LeftWrist (PD, 20.46s)"),
        ("data/raw/pads/movement/timeseries/081_Relaxed_LeftWrist.txt", "HC", "081_Relaxed_LeftWrist (fold0_test HC, 20.46s)"),
    ]

    raw_len = 2046
    n_blocks = 10   # 10 khối 2s (200 mẫu @ 100Hz -> 100 mẫu @ 50Hz mỗi khối) = 20.0s đầu phủ trọn 9 cửa sổ
    n_win = 9       # 9 cửa sổ 4s (200 mẫu @ 50Hz) với bước nhảy hop 2s (100 mẫu @ 50Hz)

    records_meta = []
    raw_records = []

    for rel_path, label, desc in rec_specs:
        fpath = ROOT / rel_path
        x_full = load_txt(fpath).astype(np.float32)
        assert x_full.shape == (6, raw_len), f"Unexpected shape {x_full.shape} for {rel_path}"
        raw_records.append(x_full)

        # Đường ống Python chuẩn: scipy resample_poly cả bản ghi -> scipy sosfilt cả bản ghi -> cắt cửa sổ 200 mẫu hop 100
        r_py = resample(x_full, 100, 50)
        xf_py_cont = signal.sosfilt(sos_f64, r_py, axis=-1).astype(np.float32)
        # Đối chiếu thêm: zero-phase cả bản ghi (sosfiltfilt) và nhân quả reset mỗi cửa sổ (causal_win)
        xf_py_zp = bandpass(r_py, 50).astype(np.float32)

        win_list = []
        for w in range(n_win):
            s50 = w * 100
            seg_cont = xf_py_cont[:, s50 : s50 + 200]
            x8_cont = np.vstack([
                seg_cont[0:3], seg_cont[3:6],
                np.linalg.norm(seg_cont[0:3], axis=0),
                np.linalg.norm(seg_cont[3:6], axis=0),
            ]).T.astype(np.float32)
            qin_cont, q_cont, l_cont, p_cont = run_tflite_from_x8(x8_cont)

            seg_zp = xf_py_zp[:, s50 : s50 + 200]
            x8_zp = np.vstack([
                seg_zp[0:3], seg_zp[3:6],
                np.linalg.norm(seg_zp[0:3], axis=0),
                np.linalg.norm(seg_zp[3:6], axis=0),
            ]).T.astype(np.float32)
            _, q_zp, l_zp, p_zp = run_tflite_from_x8(x8_zp)

            seg_rst = signal.sosfilt(sos_f64, r_py[:, s50 : s50 + 200], axis=-1).astype(np.float32)
            x8_rst = np.vstack([
                seg_rst[0:3], seg_rst[3:6],
                np.linalg.norm(seg_rst[0:3], axis=0),
                np.linalg.norm(seg_rst[3:6], axis=0),
            ]).T.astype(np.float32)
            _, q_rst, l_rst, p_rst = run_tflite_from_x8(x8_rst)

            win_list.append(dict(
                win_idx=w,
                time_range_s=f"{w*2.0:.1f}..{w*2.0+4.0:.1f}s",
                start_50hz=s50,
                q_py_cont=q_cont,
                logit_py_cont=l_cont,
                prob_py_cont=p_cont,
                qin_cksum=int(np.sum(qin_cont)),
                q_py_zp=q_zp,
                prob_py_zp=p_zp,
                q_py_reset=q_rst,
                prob_py_reset=p_rst,
            ))

        records_meta.append(dict(
            file=rel_path,
            label=label,
            desc=desc,
            raw_samples_100hz=raw_len,
            n_blocks_2s=n_blocks,
            n_windows_4s=n_win,
            windows=win_list,
        ))

    n_rec = len(records_meta)

    h_txt = f"""#pragma once
#include <stdint.h>

#define B3_N_REC {n_rec}
#define B3_RAW_C 6
#define B3_RAW_TOTAL_T {raw_len}
#define B3_BLOCK_RAW_T 200
#define B3_BLOCK_DECIM_T 100
#define B3_N_BLOCKS {n_blocks}
#define B3_WIN_DECIM_T 200
#define B3_HOP_DECIM_T 100
#define B3_N_WIN {n_win}
#define B3_OUT_C 8
#define B3_FIR_LEN 41
#define B3_SOS_SECTIONS 4

extern const char* const g_b3_rec_file[B3_N_REC];
extern const char* const g_b3_rec_label[B3_N_REC];
extern const char* const g_b3_rec_desc[B3_N_REC];

// Toàn bộ bản ghi PADS thật 100 Hz: [B3_N_REC][6][2046]
extern const float g_b3_raw_records[B3_N_REC][B3_RAW_C][B3_RAW_TOTAL_T];

// Hệ số FIR polyphase hạ mẫu 100 -> 50 Hz (41 taps Kaiser beta=5.0)
extern const float g_fir_kaiser41[B3_FIR_LEN];

// Hệ số IIR Butterworth thông dải 0.5-20 Hz bậc 4 nhân quả @ 50 Hz: [4][6]
extern const float g_butter4_bp_sos[B3_SOS_SECTIONS][6];

// Tham số chuẩn hóa z-score (reports/p56x4/models/norm_PD_vs_HC_raw_fold0.npz)
extern const float g_norm_mu[B3_OUT_C];
extern const float g_norm_sd[B3_OUT_C];

// Đầu ra tham chiếu của đường ống Python chuẩn (resample_poly cả bản ghi + sosfilt cả bản ghi + cắt cửa sổ 200 hop 100)
extern const int8_t g_ref_q_py_cont[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_cont[B3_N_REC][B3_N_WIN];
extern const int32_t g_ref_qin_cksum[B3_N_REC][B3_N_WIN];

// Đầu ra tham chiếu bổ sung (zero-phase cả bản ghi và nhân quả reset mỗi cửa sổ)
extern const int8_t g_ref_q_py_zp[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_zp[B3_N_REC][B3_N_WIN];
extern const int8_t g_ref_q_py_reset[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_reset[B3_N_REC][B3_N_WIN];
"""
    (out_dir / "b3_data.h").write_text(h_txt)

    def fmt_f_arr(arr):
        return ", ".join(f"{float(v):.9e}f" for v in np.asarray(arr).ravel())

    cc = ['#include "b3_data.h"\n']
    cc.append("const char* const g_b3_rec_file[B3_N_REC] = {\n  " +
              ",\n  ".join(f'"{r["file"]}"' for r in records_meta) + "\n};\n")
    cc.append("const char* const g_b3_rec_label[B3_N_REC] = {\n  " +
              ",\n  ".join(f'"{r["label"]}"' for r in records_meta) + "\n};\n")
    cc.append("const char* const g_b3_rec_desc[B3_N_REC] = {\n  " +
              ",\n  ".join(f'"{r["desc"]}"' for r in records_meta) + "\n};\n")

    cc.append(f"const float g_fir_kaiser41[B3_FIR_LEN] = {{ {fmt_f_arr(h_fir)} }};\n")
    sos_rows = [f"  {{ {fmt_f_arr(sos[s])} }}" for s in range(4)]
    cc.append("const float g_butter4_bp_sos[B3_SOS_SECTIONS][6] = {\n" + ",\n".join(sos_rows) + "\n};\n")
    cc.append(f"const float g_norm_mu[B3_OUT_C] = {{ {fmt_f_arr(mu)} }};\n")
    cc.append(f"const float g_norm_sd[B3_OUT_C] = {{ {fmt_f_arr(sd)} }};\n")

    def fmt_2d_int(key):
        rows = ["  { " + ", ".join(str(w[key]) for w in r["windows"]) + " }" for r in records_meta]
        return "{\n" + ",\n".join(rows) + "\n}"

    def fmt_2d_float(key):
        rows = ["  { " + ", ".join(f'{w[key]:.8f}f' for w in r["windows"]) + " }" for r in records_meta]
        return "{\n" + ",\n".join(rows) + "\n}"

    cc.append(f"const int8_t g_ref_q_py_cont[B3_N_REC][B3_N_WIN] = {fmt_2d_int('q_py_cont')};\n")
    cc.append(f"const float g_ref_prob_py_cont[B3_N_REC][B3_N_WIN] = {fmt_2d_float('prob_py_cont')};\n")
    cc.append(f"const int32_t g_ref_qin_cksum[B3_N_REC][B3_N_WIN] = {fmt_2d_int('qin_cksum')};\n")
    cc.append(f"const int8_t g_ref_q_py_zp[B3_N_REC][B3_N_WIN] = {fmt_2d_int('q_py_zp')};\n")
    cc.append(f"const float g_ref_prob_py_zp[B3_N_REC][B3_N_WIN] = {fmt_2d_float('prob_py_zp')};\n")
    cc.append(f"const int8_t g_ref_q_py_reset[B3_N_REC][B3_N_WIN] = {fmt_2d_int('q_py_reset')};\n")
    cc.append(f"const float g_ref_prob_py_reset[B3_N_REC][B3_N_WIN] = {fmt_2d_float('prob_py_reset')};\n")

    cc.append("alignas(16) const float g_b3_raw_records[B3_N_REC][B3_RAW_C][B3_RAW_TOTAL_T] = {\n")
    for rx in raw_records:
        cc.append("  {\n")
        for c in range(6):
            cc.append(f"    {{ {fmt_f_arr(rx[c])} }},\n")
        cc.append("  },\n")
    cc.append("};\n")

    (out_dir / "b3_data.cc").write_text("".join(cc))
    (out_dir / "b3_meta.json").write_text(json.dumps(dict(
        model=str(model_path.relative_to(ROOT)),
        norm=str(norm_path.relative_to(ROOT)),
        in_scale=in_scale, in_zp=in_zp,
        out_scale=out_scale, out_zp=out_zp,
        mu=mu.tolist(), sd=sd.tolist(),
        records=records_meta,
    ), indent=2))
    print("Generated b3_data.{h,cc} and b3_meta.json successfully.")
    for r in records_meta:
        probs = [round(w["prob_py_cont"], 4) for w in r["windows"]]
        print(f"  {r['desc']}: {len(probs)} windows -> probs={probs}")


if __name__ == "__main__":
    main()
