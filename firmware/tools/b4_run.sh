#!/usr/bin/env bash
# B4: Random Forest đặc trưng (emlearn float, 100 cây, max_depth=8) + Trích xuất 96 đặc trưng/bản ghi 10s trên ESP32-S3
set -euo pipefail

ROOT="$HOME/Desktop/PARKINSON"
PROJ="$ROOT/firmware/b4_rf"
BOARD_DIR="$ROOT/reports/board"
mkdir -p "$BOARD_DIR"

PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)
PORT="${PORT:-/dev/ttyACM0}"

echo "[B4] Starting at $(date -Iseconds) on PORT=$PORT"

export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"
if ! "$ROOT/.venv/bin/python" -c "import emlearn" >/dev/null 2>&1; then
  echo "[B4] Installing emlearn + setuptools into firmware/.pydeps..."
  "$HOME/.local/bin/uv" pip install --target "$ROOT/firmware/.pydeps" --python "$ROOT/.venv/bin/python" emlearn setuptools
  mkdir -p "$ROOT/firmware/.pydeps/distutils"
  echo "from setuptools._distutils import *" > "$ROOT/firmware/.pydeps/distutils/__init__.py"
  echo "from setuptools._distutils.ccompiler import *" > "$ROOT/firmware/.pydeps/distutils/ccompiler.py"
fi

echo "[B4] Step 1: Training RF in Python & generating C files via firmware/tools/gen_b4_data.py..."
"$ROOT/.venv/bin/python" "$ROOT/firmware/tools/gen_b4_data.py"

echo "[B4] Step 2: Sourcing ESP-IDF environment & building firmware/b4_rf..."
# shellcheck disable=SC1091
. "$HOME/esp/esp-idf/export.sh" >/dev/null 2>&1

cd "$PROJ"
if ! idf.py build > "$BOARD_DIR/B4_build.log" 2>&1; then
  echo "[B4] ERROR: Build failed. Last 40 lines of B4_build.log:"
  tail -40 "$BOARD_DIR/B4_build.log"
  exit 1
fi
echo "[B4] Build succeeded."

idf.py size > "$BOARD_DIR/B4_size.txt" 2>&1 || true
idf.py size-components > "$BOARD_DIR/B4_size_components.txt" 2>&1 || true

echo "[B4] Step 3: Flashing firmware/b4_rf to $PORT..."
if ! idf.py -p "$PORT" flash > "$BOARD_DIR/B4_flash.log" 2>&1; then
  echo "[B4] ERROR: Flash failed. Last 30 lines of B4_flash.log:"
  tail -30 "$BOARD_DIR/B4_flash.log"
  exit 1
fi
echo "[B4] Flash succeeded."

cd "$ROOT"
echo "[B4] Step 4: Capturing serial output from $PORT..."
"$ROOT/.venv/bin/python" - "$PORT" "$BOARD_DIR/B4_serial.log" "$BOARD_DIR/B4_SUMMARY.txt" "$BOARD_DIR/B4_size.txt" "$BOARD_DIR/B4_rf_python.json" <<'PY'
import datetime, glob, json, pathlib, re, sys, time
import serial

port_arg, log_path_str, summary_path_str, size_path_str, rf_json_str = sys.argv[1:6]
log_path = pathlib.Path(log_path_str)
summary_path = pathlib.Path(summary_path_str)
size_path = pathlib.Path(size_path_str)
rf_json_path = pathlib.Path(rf_json_str)

def find_port(preferred):
    if pathlib.Path(preferred).exists():
        return preferred
    cands = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return cands[0] if cands else preferred

lines = []
t0 = time.time()
timeout_s = 60.0
did_reset = False

while time.time() - t0 < timeout_s:
    cur_port = find_port(port_arg)
    try:
        s = serial.Serial(cur_port, 115200, timeout=1.0)
    except Exception:
        time.sleep(0.25)
        continue
    try:
        if not did_reset:
            s.dtr = False
            s.rts = True
            time.sleep(0.12)
            s.rts = False
            did_reset = True
        while time.time() - t0 < timeout_s:
            raw = s.readline()
            if not raw:
                continue
            ln = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
            if not ln:
                continue
            lines.append(ln)
            print(f"[{time.time() - t0:6.2f}s] {ln}", flush=True)
            if ln.startswith("#B4 END") or ln.startswith("#ERR"):
                break
        if any(x.startswith("#B4 END") or x.startswith("#ERR") for x in lines):
            break
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        time.sleep(0.2)
        continue
    finally:
        try:
            s.close()
        except Exception:
            pass

log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[B4] Saved serial log ({len(lines)} lines) to {log_path}", flush=True)

rf_py = json.loads(rf_json_path.read_text(encoding="utf-8")) if rf_json_path.exists() else {}
chip_info = {}
mem_info = {}
rf_cmps = []
rf_sum = {}
feat_cmps = []
feat_sum = {}
step_stats = {}
errors = []

for ln in lines:
    if ln.startswith("#ERR"):
        errors.append(ln)
    elif ln.startswith("#chip "):
        chip_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#memory "):
        mem_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#rf_cmp "):
        rf_cmps.append(dict(re.findall(r"(\w+)=([\w./+-]+)", ln)))
    elif ln.startswith("#rf_summary "):
        rf_sum = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#feat_cmp "):
        feat_cmps.append(dict(re.findall(r"(\w+)=([\w./+-]+)", ln)))
    elif ln.startswith("#feat_summary "):
        feat_sum = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#step_latency "):
        d = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
        if "step" in d:
            step_stats[d["step"]] = d

flash_img_bytes = "N/A"
flash_bin_bytes = "N/A"
dram_static_bytes = "N/A"
flash_code_bytes = "N/A"
flash_rodata_bytes = "N/A"
if size_path.exists():
    stxt = size_path.read_text(errors="ignore")
    m_img = re.search(r"Total image size:\s*(\d+)\s*bytes", stxt)
    m_bin = re.search(r"b4_rf\.bin binary size (0x[0-9a-fA-F]+) bytes", stxt)
    m_dram = re.search(r"(?:Used static DRAM:|DIRAM\s*│)\s*(\d+)", stxt)
    m_code = re.search(r"(?:Flash code:|Flash Code\s*│)\s*(\d+)", stxt)
    m_ro = re.search(r"(?:Flash rodata:|Flash Data\s*│)\s*(\d+)", stxt)
    if m_img:
        flash_img_bytes = f"{int(m_img.group(1)):,} B ({int(m_img.group(1))/1024:.1f} KB)"
    if m_bin:
        bval = int(m_bin.group(1), 16)
        flash_bin_bytes = f"{bval:,} B ({m_bin.group(1)})"
    if m_dram:
        dram_static_bytes = f"{int(m_dram.group(1)):,} B ({int(m_dram.group(1))/1024:.1f} KB)"
    if m_code:
        flash_code_bytes = f"{int(m_code.group(1)):,} B ({int(m_code.group(1))/1024:.1f} KB)"
    if m_ro:
        flash_rodata_bytes = f"{int(m_ro.group(1)):,} B ({int(m_ro.group(1))/1024:.1f} KB)"

rf_soft_st = step_stats.get("rf_infer_proba_soft_100trees", {})
rf_hard_st = step_stats.get("rf_infer_label_hard_100trees", {})
f_down_st = step_stats.get("rec10s_1_downsample_100to50", {})
f_bp_st = step_stats.get("rec10s_2_bandpass_biquad4", {})
f_ch8_st = step_stats.get("rec10s_3_make_8channels", {})
f_ext_st = step_stats.get("rec10s_4_extract_96feats_welch256", {})
f_tot_st = step_stats.get("rec10s_total_feature_pipeline", {})

rec_tot_mean_ms = float(f_tot_st.get("mean_ms", 0) or 0)
rec_tot_p95_ms = float(f_tot_st.get("p95_ms", 0) or 0)
rf_soft_mean_us = float(rf_soft_st.get("mean_us", 0) or 0)
rf_soft_p95_us = float(rf_soft_st.get("p95_us", 0) or 0)
rf_hard_mean_us = float(rf_hard_st.get("mean_us", 0) or 0)
rf_hard_p95_us = float(rf_hard_st.get("p95_us", 0) or 0)

out_lines = [
    "# B4 RANDOM FOREST & ON-BOARD FEATURE EXTRACTION SUMMARY (ESP32-S3)",
    f"Generated: {datetime.datetime.now().astimezone().isoformat()}",
    "Board: ESP32-S3 (Sparkle IoT XH-S3E N16R8, 240 MHz, 16 MB Flash, 8 MB Octal PSRAM)",
    f"Port: {port_arg} (USB-Serial-JTAG)",
    "",
    "====================================================================================================",
    "1. MÔ HÌNH RANDOM FOREST (PYTHON SCIKIT-LEARN -> EMLEARN C FLOAT) & AUROC TEST FOLD 0",
    "====================================================================================================",
    f"- Bài toán & Chia dữ liệu     : PD vs HC ({rf_py.get('n_patients_total', 355)} bệnh nhân), {rf_py.get('cv_split', '')}",
    f"- Tập Train / Test Fold 0     : n_train = {rf_py.get('n_train_fold0', 284)}, n_test = {rf_py.get('n_test_fold0', 71)} (PD={rf_py.get('n_test_pd', 55)}, HC={rf_py.get('n_test_hc', 16)})",
    f"- Số đặc trưng (Features)     : {rf_py.get('n_features_total', 2112)} đặc trưng/bệnh nhân ({rf_py.get('n_records_per_patient', 22)} bản ghi [11 bài x 2 cổ tay] x 96 đặc trưng/bản ghi [12 đặc trưng/kênh x 8 kênh])",
    f"- Cấu hình Random Forest      : n_estimators=100, max_depth=8, class_weight='balanced', random_state=0",
    f"- Quy mô rừng cây             : {rf_py.get('total_tree_nodes', 'N/A')} nút (nodes), {rf_py.get('total_tree_leaves', 'N/A')} lá (leaves), độ sâu thực tế tối đa = {rf_py.get('max_tree_depth_actual', 8)}",
    f"- AUROC Test Fold 0 (sklearn) : {rf_py.get('auroc_test_fold0_sklearn', 'N/A')} (Balanced Acc = {rf_py.get('bal_acc_test_fold0_sklearn', 'N/A')}, Acc = {rf_py.get('acc_test_fold0_sklearn', 'N/A')}, Sens = {rf_py.get('sens_test_fold0_sklearn', 'N/A')}, Spec = {rf_py.get('spec_test_fold0_sklearn', 'N/A')})",
    f"- AUROC Test Fold 0 (emlearn) : {rf_py.get('auroc_test_fold0_emlearn_soft', 'N/A')} (soft leaf_bits=8) / {rf_py.get('auroc_test_fold0_emlearn_hard', 'N/A')} (hard majority vote)",
    "  (Chi tiết lưu tại: reports/board/B4_rf_python.json)",
    "",
    "====================================================================================================",
    "2. TÀI NGUYÊN BỘ NHỚ TRÊN BO (FLASH & RAM)",
    "====================================================================================================",
    f"- Tổng Flash Image (idf.py size) : {flash_img_bytes} (binary: {flash_bin_bytes})",
    f"- Flash Code / Flash Rodata      : Code = {flash_code_bytes} | Rodata = {flash_rodata_bytes} (gồm cả 20 vector test nhúng sẵn = 168.96 KB)",
    f"- Bảng lá RF (pd_rf_leaves)      : {mem_info.get('rf_leaves_rodata_bytes', 'N/A')} B trong Flash Rodata; RAM động khi suy luận RF = 0 B (stack ~64 B)",
    f"- DSP & Welch FFT Static Buffers : {mem_info.get('feat_dsp_static_ram_bytes', 'N/A')} B ({int(mem_info.get('feat_dsp_static_ram_bytes', 0))/1024:.2f} KB SRAM nội cho bản ghi 10s)",
    f"- Tổng Static DRAM (idf.py size) : {dram_static_bytes}",
    f"- Internal SRAM Free             : {chip_info.get('internal_free', 'N/A')} B / {chip_info.get('internal_total', 'N/A')} B",
    "",
    "====================================================================================================",
    "3. ĐỘ TRỄ SUY LUẬN RANDOM FOREST & ĐỐI CHIẾU 20 VECTOR TEST VỚI SCIKIT-LEARN",
    "====================================================================================================",
    f"- Khớp nhãn dự đoán với sklearn trên 20 vector test : {rf_sum.get('label_match_soft', 'N/A')} (soft voting leaf_bits=8) | {rf_sum.get('label_match_hard', 'N/A')} (hard voting leaf_bits=0)",
    f"- Khớp bit-exact với mô hình emlearn tham chiếu PC  : {rf_sum.get('eml_ref_exact', 'N/A')}",
    f"- Sai số xác suất |P_bo - P_sklearn| (soft voting)  : Max = {float(rf_sum.get('max_abs_dp_soft', 0)):.6f}, Mean = {float(rf_sum.get('mean_abs_dp_soft', 0)):.6f} (do lượng tử hóa lá 8-bit 1/255)",
    f"- Sai số xác suất |P_bo - P_sklearn| (hard voting)  : Max = {float(rf_sum.get('max_abs_dp_hard', 0)):.6f}",
    f"- Độ trễ suy luận RF (n={rf_soft_st.get('n', 1000)} lần, 100 cây, 2112 đặc trưng):",
    f"  + Suy luận xác suất mềm (pd_rf_predict_proba)     : Mean = {rf_soft_mean_us:.2f} us ({rf_soft_mean_us/1000.0:.4f} ms) | p50 = {rf_soft_st.get('p50_us', 'N/A')} us | p95 = {rf_soft_p95_us:.2f} us ({rf_soft_p95_us/1000.0:.4f} ms)",
    f"  + Suy luận nhãn đa số (pd_rf_hard_predict)        : Mean = {rf_hard_mean_us:.2f} us ({rf_hard_mean_us/1000.0:.4f} ms) | p50 = {rf_hard_st.get('p50_us', 'N/A')} us | p95 = {rf_hard_p95_us:.2f} us ({rf_hard_p95_us/1000.0:.4f} ms)",
    "",
    f"{'Idx':<3} | {'SID':<4} | {'y_true':>6} | {'y_sk':>4} | {'P_sklearn':>9} | {'y_bo_soft':>9} | {'P_bo_soft':>9} | {'|dP_soft|':>9} | {'y_bo_hard':>9} | {'P_bo_hard':>9}",
    "-" * 96,
]

for r in rf_cmps:
    out_lines.append(
        f"{r.get('idx',''):<3} | {r.get('sid',''):<4} | {r.get('y_true',''):>6} | {r.get('y_sklearn',''):>4} | "
        f"{float(r.get('prob_sklearn',0)):9.6f} | {r.get('y_board_soft',''):>9} | {float(r.get('prob_board_soft',0)):9.6f} | "
        f"{float(r.get('dp_soft',0)):9.6f} | {r.get('y_board_hard',''):>9} | {float(r.get('prob_board_hard',0)):9.6f}"
    )

step_rows = [
    ("rec10s_1_downsample_100to50",      "1. Hạ mẫu 100->50 Hz (FIR polyphase 41 taps, 1000->500 mẫu x 6 kênh)"),
    ("rec10s_2_bandpass_biquad4",        "2. Lọc thông dải 0.5-20 Hz (Butterworth bậc 4 nhân quả, 500 mẫu x 6 kênh)"),
    ("rec10s_3_make_8channels",          "3. Tạo 8 kênh (ax, ay, az, wx, wy, wz, |a|, |w|, 500 mẫu x 8 kênh)"),
    ("rec10s_4_extract_96feats_welch256","4. Trích xuất 96 đặc trưng (12 đặc trưng x 8 kênh, gồm Welch PSD FFT-256)"),
    ("rec10s_total_feature_pipeline",    "TỔNG TRÍCH XUẤT ĐẶC TRƯNG 1 BẢN GHI 10s (Bước 1 -> 4)"),
]

out_lines.extend([
    "",
    "====================================================================================================",
    "4. ĐỘ TRỄ & ĐỘ CHÍNH XÁC TRÍCH XUẤT 96 ĐẶC TRƯNG TRÊN BO CHO 1 BẢN GHI 10 GIÂY (1000 MẪU x 6 KÊNH @ 100 Hz)",
    "   Bản ghi: data/raw/pads/movement/timeseries/005_HoldWeight_LeftWrist.txt [0..10s]",
    "====================================================================================================",
    f"- Số đặc trưng mỗi bản ghi     : {feat_sum.get('n_feats', 96)} (12 đặc trưng/kênh x 8 kênh: std, iqr, skew, kurt, rms_diff, log_pow, peak_f, spec_ent, 4 band powers)",
    f"- Sai số tương đối tối đa so với Python (scipy/numpy): {float(feat_sum.get('max_rel_err', 0)):.6e} ({float(feat_sum.get('max_rel_err', 0))*100:.6f}%) tại đặc trưng #{feat_sum.get('worst_idx', '?')} ({feat_sum.get('worst_name', '?')})",
    f"- Sai số tương đối trung bình (trên 96 đặc trưng)    : {float(feat_sum.get('mean_rel_err', 0)):.6e}",
    f"- Sai số tuyệt đối tối đa (trên 96 đặc trưng)        : {float(feat_sum.get('max_abs_err', 0)):.6e}",
    "",
    f"{'Bước xử lý (1 bản ghi 10s x 6 kênh @ 100 Hz)':<72} | {'Mean (us)':>10} | {'p50 (us)':>9} | {'p95 (us)':>9} | {'Mean (ms)':>10} | {'p95 (ms)':>10} | {'Tỷ trọng':>8}",
    "-" * 142,
])

for key, label in step_rows:
    st = step_stats.get(key, {})
    mean_us = float(st.get("mean_us", 0) or 0)
    p50_us = int(st.get("p50_us", 0) or 0)
    p95_us = int(st.get("p95_us", 0) or 0)
    mean_ms = float(st.get("mean_ms", 0) or 0)
    p95_ms = float(st.get("p95_ms", 0) or 0)
    pct = (100.0 * mean_ms / rec_tot_mean_ms) if rec_tot_mean_ms > 0 else 0.0
    out_lines.append(
        f"{label:<72} | {mean_us:10.2f} | {p50_us:9d} | {p95_us:9d} | {mean_ms:10.4f} | {p95_ms:10.4f} | {pct:7.2f}%"
    )

out_lines.extend([
    "-" * 142,
    f"-> Thời gian tính đặc trưng cho 1 bản ghi 10s x 6 kênh @ 100 Hz: Mean = {rec_tot_mean_ms:.3f} ms (p95 = {rec_tot_p95_ms:.3f} ms).",
    f"-> Nếu tính đủ 22 bản ghi (11 bài x 2 cổ tay = 2112 đặc trưng) + suy luận RF 100 cây: "
    f"22 x {rec_tot_mean_ms:.3f} ms + {rf_soft_mean_us/1000.0:.3f} ms = {22.0*rec_tot_mean_ms + rf_soft_mean_us/1000.0:.2f} ms.",
    "",
    f"Errors: {', '.join(errors) if errors else 'none'}",
])

summary_txt = "\n".join(out_lines) + "\n"
summary_path.write_text(summary_txt, encoding="utf-8")
print("\n" + summary_txt, flush=True)
if errors or not step_stats:
    sys.exit(1)
PY

echo "[B4] Completed successfully at $(date -Iseconds)"
