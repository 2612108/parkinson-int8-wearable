"""Sinh dữ liệu C cho firmware/b2_e2e:
- Đóng gói mô hình reports/p56a/models/PD_vs_HC_raw_qat_fold0.tflite (qua tflite_to_c.py)
- Sinh hệ số lọc FIR polyphase 100->50 Hz (41 taps Kaiser beta=5.0 khớp scipy.signal.resample_poly(x, 1, 2))
- Sinh hệ số lọc IIR Butterworth thông dải 0.5-20 Hz bậc 4 nhân quả (4 tầng biquad SOS ở fs=50 Hz)
- Sinh tham số chuẩn hóa mu, sd từ reports/p56a/models/norm_PD_vs_HC_raw_fold0.npz
- Trích xuất các cửa sổ 4 s (400 mẫu x 6 kênh @ 100 Hz) từ các bản ghi PADS thật trong data/raw/pads/movement/timeseries/
  kèm đầu ra tham chiếu của đường ống Python:
    (a) Python Zero-Phase trên cửa sổ 4 s (resample_poly + sosfiltfilt)
    (b) Python Zero-Phase trên toàn bản ghi rồi cắt cửa sổ (như lúc huấn luyện)
    (c) Python Nhân quả (resample_poly + sosfilt biquad nhân quả) để đối chiếu bit-exact với bo
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
from src.features import bandpass, resample
from ai_edge_litert import interpreter as _tfl


def main():
    model_path = ROOT / "reports/p56a/models/PD_vs_HC_raw_qat_fold0.tflite"
    norm_path = ROOT / "reports/p56a/models/norm_PD_vs_HC_raw_fold0.npz"
    bench_in_path = ROOT / "reports/p56a/models/bench_inputs_PD_vs_HC_raw.npy"
    out_dir = ROOT / "firmware/b2_e2e/main/generated"
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

    # 2. Đọc norm mu, sd và thông tin lượng tử hóa của mô hình
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
    sos = signal.butter(4, [0.5, 20.0], btype="band", fs=50.0, output="sos").astype(np.float32)
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

    def c_exact_preprocess(x_raw):
        # x_raw: (6, 400) float32
        x50 = np.zeros((6, 200), dtype=np.float32)
        for c in range(6):
            for n in range(200):
                acc = np.float32(0.0)
                center = 2 * n + 20
                for k in range(41):
                    idx = center - k
                    if 0 <= idx < 400:
                        acc = np.float32(acc + h_fir[k] * x_raw[c, idx])
                x50[c, n] = acc
        xf = np.zeros((6, 200), dtype=np.float32)
        for c in range(6):
            buf = x50[c].copy()
            for s in range(4):
                b0, b1, b2, a0, a1, a2 = sos[s]
                z1, z2 = np.float32(0.0), np.float32(0.0)
                for n in range(200):
                    x = buf[n]
                    y = np.float32(b0 * x + z1)
                    z1 = np.float32(b1 * x - a1 * y + z2)
                    z2 = np.float32(b2 * x - a2 * y)
                    buf[n] = y
            xf[c] = buf
        x8 = np.zeros((200, 8), dtype=np.float32)
        for n in range(200):
            ax, ay, az = xf[0, n], xf[1, n], xf[2, n]
            wx, wy, wz = xf[3, n], xf[4, n], xf[5, n]
            x8[n, 0] = ax; x8[n, 1] = ay; x8[n, 2] = az
            x8[n, 3] = wx; x8[n, 4] = wy; x8[n, 5] = wz
            x8[n, 6] = np.sqrt(np.float32(ax * ax + ay * ay + az * az))
            x8[n, 7] = np.sqrt(np.float32(wx * wx + wy * wy + wz * wz))
        return x8

    # 5. Chọn các cửa sổ 4s từ bản ghi PADS thật
    # Cửa sổ 0 là cửa sổ chuẩn dùng để đo latency 1000 lần và đo năng lượng.
    specs = [
        ("data/raw/pads/movement/timeseries/001_Relaxed_LeftWrist.txt", 0, "interp100", "PD", "001_Relaxed_LeftWrist [0..4s, load_txt]"),
        ("data/raw/pads/movement/timeseries/001_Relaxed_LeftWrist.txt", 0, "raw_txt",   "PD", "001_Relaxed_LeftWrist [0..4s, raw cols]"),
        ("data/raw/pads/movement/timeseries/001_Relaxed_LeftWrist.txt", 200, "interp100", "PD", "001_Relaxed_LeftWrist [2..6s, load_txt]"),
        ("data/raw/pads/movement/timeseries/005_Relaxed_LeftWrist.txt", 0, "interp100", "PD", "005_Relaxed_LeftWrist [0..4s, fold0_test PD]"),
        ("data/raw/pads/movement/timeseries/005_Relaxed_LeftWrist.txt", 200, "interp100", "PD", "005_Relaxed_LeftWrist [2..6s, fold0_test PD]"),
        ("data/raw/pads/movement/timeseries/029_Relaxed_LeftWrist.txt", 0, "interp100", "PD", "029_Relaxed_LeftWrist [0..4s, fold0_test PD]"),
        ("data/raw/pads/movement/timeseries/081_Relaxed_LeftWrist.txt", 0, "interp100", "HC", "081_Relaxed_LeftWrist [0..4s, fold0_test HC]"),
        ("data/raw/pads/movement/timeseries/108_Relaxed_LeftWrist.txt", 0, "interp100", "HC", "108_Relaxed_LeftWrist [0..4s, fold0_test HC]"),
    ]

    windows_meta = []
    raw_windows = []
    for rel_path, start, mode, label, desc in specs:
        fpath = ROOT / rel_path
        if mode == "raw_txt":
            arr = np.loadtxt(fpath, delimiter=",", dtype=np.float64)
            x_full = arr[:, 1:7].T.astype(np.float32)
        else:
            x_full = load_txt(fpath)
        x_win = x_full[:, start:start + 400].astype(np.float32)
        assert x_win.shape == (6, 400)

        # (a) Python zero-phase trên cùng cửa sổ 4s (resample_poly + sosfiltfilt)
        xr_win = resample(x_win, 100, 50)
        xf_zp_win = signal.sosfiltfilt(sos, xr_win, axis=-1)
        x8_zp_win = np.vstack([
            xf_zp_win[0:3], xf_zp_win[3:6],
            np.linalg.norm(xf_zp_win[0:3], axis=0),
            np.linalg.norm(xf_zp_win[3:6], axis=0)
        ]).T.astype(np.float32)
        _, q_zp_win, l_zp_win, p_zp_win = run_tflite_from_x8(x8_zp_win)

        # (b) Python zero-phase trên toàn bản ghi rồi cắt cửa sổ (như lúc huấn luyện)
        xf_zp_full_all = bandpass(resample(x_full, 100, 50), 50)
        xf_zp_full = xf_zp_full_all[:, start // 2 : start // 2 + 200]
        x8_zp_full = np.vstack([
            xf_zp_full[0:3], xf_zp_full[3:6],
            np.linalg.norm(xf_zp_full[0:3], axis=0),
            np.linalg.norm(xf_zp_full[3:6], axis=0)
        ]).T.astype(np.float32)
        _, q_zp_full, l_zp_full, p_zp_full = run_tflite_from_x8(x8_zp_full)

        # (c) Đường ống nhân quả C-exact (FIR polyphase + IIR biquad nhân quả)
        x8_causal = c_exact_preprocess(x_win)
        q_in_causal, q_causal, l_causal, p_causal = run_tflite_from_x8(x8_causal)

        raw_windows.append(x_win)
        windows_meta.append(dict(
            file=rel_path,
            start_sample_100hz=start,
            mode=mode,
            true_label=label,
            desc=desc,
            q_py_causal=q_causal,
            logit_py_causal=l_causal,
            prob_py_causal=p_causal,
            q_py_zerophase_win=q_zp_win,
            logit_py_zerophase_win=l_zp_win,
            prob_py_zerophase_win=p_zp_win,
            q_py_zerophase_full=q_zp_full,
            logit_py_zerophase_full=l_zp_full,
            prob_py_zerophase_full=p_zp_full,
            q_in_checksum=int(np.sum(q_in_causal)),
        ))

    n_win = len(raw_windows)

    # Sinh e2e_data.h
    h_txt = f"""#pragma once
#include <stdint.h>

#define E2E_N_WIN {n_win}
#define E2E_RAW_C 6
#define E2E_RAW_T 400
#define E2E_DECIM_T 200
#define E2E_OUT_C 8
#define E2E_FIR_LEN 41
#define E2E_SOS_SECTIONS 4

extern const char* const g_e2e_win_desc[E2E_N_WIN];
extern const char* const g_e2e_win_file[E2E_N_WIN];
extern const int g_e2e_win_start[E2E_N_WIN];
extern const char* const g_e2e_win_label[E2E_N_WIN];

// Tín hiệu thô 100 Hz: [E2E_N_WIN][6][400]
extern const float g_e2e_raw_in[E2E_N_WIN][E2E_RAW_C][E2E_RAW_T];

// Hệ số FIR polyphase hạ mẫu 100 -> 50 Hz (41 taps)
extern const float g_fir_kaiser41[E2E_FIR_LEN];

// Hệ số IIR Butterworth thông dải 0.5-20 Hz bậc 4 nhân quả @ 50 Hz: [4][6] = {{b0, b1, b2, a0, a1, a2}}
extern const float g_butter4_bp_sos[E2E_SOS_SECTIONS][6];

// Tham số chuẩn hóa z-score (8 kênh: ax, ay, az, wx, wy, wz, |a|, |w|)
extern const float g_norm_mu[E2E_OUT_C];
extern const float g_norm_sd[E2E_OUT_C];

// Đầu ra tham chiếu trên PC cho từng cửa sổ
extern const int8_t g_ref_q_causal[E2E_N_WIN];
extern const float g_ref_prob_causal[E2E_N_WIN];
extern const int8_t g_ref_q_zerophase_win[E2E_N_WIN];
extern const float g_ref_prob_zerophase_win[E2E_N_WIN];
extern const int8_t g_ref_q_zerophase_full[E2E_N_WIN];
extern const float g_ref_prob_zerophase_full[E2E_N_WIN];
extern const int32_t g_ref_qin_checksum[E2E_N_WIN];
"""
    (out_dir / "e2e_data.h").write_text(h_txt)

    def fmt_f_arr(arr):
        return ", ".join(f"{float(v):.9e}f" for v in np.asarray(arr).ravel())

    cc_lines = ['#include "e2e_data.h"\n']
    cc_lines.append("const char* const g_e2e_win_desc[E2E_N_WIN] = {\n  " +
                    ",\n  ".join(f'"{m["desc"]}"' for m in windows_meta) + "\n};\n")
    cc_lines.append("const char* const g_e2e_win_file[E2E_N_WIN] = {\n  " +
                    ",\n  ".join(f'"{m["file"]}"' for m in windows_meta) + "\n};\n")
    cc_lines.append("const int g_e2e_win_start[E2E_N_WIN] = { " +
                    ", ".join(str(m["start_sample_100hz"]) for m in windows_meta) + " };\n")
    cc_lines.append("const char* const g_e2e_win_label[E2E_N_WIN] = { " +
                    ", ".join(f'"{m["true_label"]}"' for m in windows_meta) + " };\n")

    cc_lines.append(f"const float g_fir_kaiser41[E2E_FIR_LEN] = {{ {fmt_f_arr(h_fir)} }};\n")
    sos_rows = [f"  {{ {fmt_f_arr(sos[s])} }}" for s in range(4)]
    cc_lines.append("const float g_butter4_bp_sos[E2E_SOS_SECTIONS][6] = {\n" + ",\n".join(sos_rows) + "\n};\n")
    cc_lines.append(f"const float g_norm_mu[E2E_OUT_C] = {{ {fmt_f_arr(mu)} }};\n")
    cc_lines.append(f"const float g_norm_sd[E2E_OUT_C] = {{ {fmt_f_arr(sd)} }};\n")

    cc_lines.append("const int8_t g_ref_q_causal[E2E_N_WIN] = { " +
                    ", ".join(str(m["q_py_causal"]) for m in windows_meta) + " };\n")
    cc_lines.append("const float g_ref_prob_causal[E2E_N_WIN] = { " +
                    ", ".join(f'{m["prob_py_causal"]:.8f}f' for m in windows_meta) + " };\n")
    cc_lines.append("const int8_t g_ref_q_zerophase_win[E2E_N_WIN] = { " +
                    ", ".join(str(m["q_py_zerophase_win"]) for m in windows_meta) + " };\n")
    cc_lines.append("const float g_ref_prob_zerophase_win[E2E_N_WIN] = { " +
                    ", ".join(f'{m["prob_py_zerophase_win"]:.8f}f' for m in windows_meta) + " };\n")
    cc_lines.append("const int8_t g_ref_q_zerophase_full[E2E_N_WIN] = { " +
                    ", ".join(str(m["q_py_zerophase_full"]) for m in windows_meta) + " };\n")
    cc_lines.append("const float g_ref_prob_zerophase_full[E2E_N_WIN] = { " +
                    ", ".join(f'{m["prob_py_zerophase_full"]:.8f}f' for m in windows_meta) + " };\n")
    cc_lines.append("const int32_t g_ref_qin_checksum[E2E_N_WIN] = { " +
                    ", ".join(str(m["q_in_checksum"]) for m in windows_meta) + " };\n")

    cc_lines.append("alignas(16) const float g_e2e_raw_in[E2E_N_WIN][E2E_RAW_C][E2E_RAW_T] = {\n")
    for w_idx, xw in enumerate(raw_windows):
        cc_lines.append("  {\n")
        for c in range(6):
            cc_lines.append(f"    {{ {fmt_f_arr(xw[c])} }},\n")
        cc_lines.append("  },\n")
    cc_lines.append("};\n")

    (out_dir / "e2e_data.cc").write_text("".join(cc_lines))
    (out_dir / "e2e_meta.json").write_text(json.dumps(dict(
        model=str(model_path.relative_to(ROOT)),
        norm=str(norm_path.relative_to(ROOT)),
        in_scale=in_scale, in_zp=in_zp,
        out_scale=out_scale, out_zp=out_zp,
        mu=mu.tolist(), sd=sd.tolist(),
        windows=windows_meta,
    ), indent=2))
    print("Generated e2e_data.{h,cc} and e2e_meta.json successfully.")
    for idx, m in enumerate(windows_meta):
        print(f"  Win {idx}: {m['desc']} -> Causal(q={m['q_py_causal']}, p={m['prob_py_causal']:.4f}) | "
              f"ZP_win(q={m['q_py_zerophase_win']}, p={m['prob_py_zerophase_win']:.4f}) | "
              f"ZP_full(q={m['q_py_zerophase_full']}, p={m['prob_py_zerophase_full']:.4f})")


if __name__ == "__main__":
    main()
