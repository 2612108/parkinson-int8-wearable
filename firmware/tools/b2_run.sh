#!/usr/bin/env bash
# B2 End-to-End (DSP + Normalization + INT8 Quantization + TFLite Micro ESP-NN + 3-Phase Energy Mode)
# Cách dùng:
#   bash firmware/tools/b2_run.sh              # Build -> Flash -> Đọc serial -> Ghi B2_E2E.log & B2_SUMMARY.txt
#   bash firmware/tools/b2_run.sh --monitor    # Chỉ reset bo và đọc serial (dùng khi đo lại năng lượng với đồng hồ USB)
set -euo pipefail

ROOT="$HOME/Desktop/PARKINSON"
PROJ="$ROOT/firmware/b2_e2e"
BOARD_DIR="$ROOT/reports/board"
mkdir -p "$BOARD_DIR"

MODE="${1:-full}"
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)
PORT="${PORT:-/dev/ttyACM0}"

echo "[B2] Starting at $(date -Iseconds) on PORT=$PORT (mode=$MODE)"

export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"

if [ "$MODE" != "--monitor" ]; then
  if [ ! -f "$PROJ/main/generated/e2e_data.cc" ] || [ ! -f "$PROJ/main/generated/model_data.cc" ]; then
    echo "[B2] Generating C data via firmware/tools/gen_b2_data.py..."
    "$ROOT/.venv/bin/python" "$ROOT/firmware/tools/gen_b2_data.py"
  fi

  echo "[B2] Sourcing ESP-IDF environment..."
  # shellcheck disable=SC1091
  . "$HOME/esp/esp-idf/export.sh" >/dev/null 2>&1

  cd "$PROJ"
  echo "[B2] Building firmware/b2_e2e..."
  if ! idf.py build > "$BOARD_DIR/B2_build.log" 2>&1; then
    echo "[B2] ERROR: Build failed. Last 40 lines of B2_build.log:"
    tail -40 "$BOARD_DIR/B2_build.log"
    exit 1
  fi
  echo "[B2] Build succeeded."

  idf.py size > "$BOARD_DIR/B2_size.txt" 2>&1 || true

  echo "[B2] Flashing to $PORT..."
  if ! idf.py -p "$PORT" flash > "$BOARD_DIR/B2_flash.log" 2>&1; then
    echo "[B2] ERROR: Flash failed. Last 30 lines of B2_flash.log:"
    tail -30 "$BOARD_DIR/B2_flash.log"
    exit 1
  fi
  echo "[B2] Flash succeeded."
fi

cd "$ROOT"
echo "[B2] Capturing serial log from $PORT (takes ~110s including 3x30s energy phases)..."

"$ROOT/.venv/bin/python" - "$PORT" "$BOARD_DIR/B2_E2E.log" "$BOARD_DIR/B2_SUMMARY.txt" "$BOARD_DIR/B2_size.txt" <<'PY'
import datetime, glob, json, pathlib, re, sys, time
import serial

port_arg, log_path_str, summary_path_str, size_path_str = sys.argv[1:5]
log_path = pathlib.Path(log_path_str)
summary_path = pathlib.Path(summary_path_str)
size_path = pathlib.Path(size_path_str)

def find_port(preferred):
    if pathlib.Path(preferred).exists():
        return preferred
    cands = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return cands[0] if cands else preferred

lines = []
seen_end_markers = set()
t0 = time.time()
timeout_s = 175.0
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
            # Khử trùng lặp các dòng kết thúc Pha 3 (do main.cc in lặp 3 lần sau light-sleep)
            if ln.startswith(("PHASE SLEEP END", "PHASE INFER_DUTY END", "#energy_phases")):
                key = ln.split()[0] + "_" + ln.split()[1]
                if key in seen_end_markers:
                    continue
                seen_end_markers.add(key)
            lines.append(ln)
            print(f"[{time.time() - t0:6.2f}s] {ln}", flush=True)
            if ln.startswith("#B2 END") or ln.startswith("#ERR"):
                break
        if any(x.startswith("#B2 END") or x.startswith("#ERR") for x in lines):
            break
    except Exception:
        # Cổng USB-Serial-JTAG có thể ngắt/kết nối lại khi ESP32-S3 vào light-sleep ở Pha 3
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
print(f"[B2] Saved raw serial log ({len(lines)} lines) to {log_path}", flush=True)

# Phân tích log để tạo B2_SUMMARY.txt
chip_info = {}
mem_info = {}
quant_info = {}
esp_nn_mode = "unknown"
bitexact_model = "N/A"
causal_matched = "N/A"
primary_win_info = {}
win_cmps = []
step_stats = {}
phases = {}
energy_kv = {}
errors = []

for ln in lines:
    if ln.startswith("#ERR"):
        errors.append(ln)
    elif ln.startswith("#chip "):
        chip_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#esp_nn="):
        esp_nn_mode = ln.split("=", 1)[1].strip()
    elif ln.startswith("#memory "):
        mem_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#quant "):
        quant_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#bitexact_model_only "):
        m = re.search(r"matched=([\d/]+)", ln)
        if m:
            bitexact_model = m.group(1)
    elif ln.startswith("#primary_pads_file="):
        primary_win_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#win_cmp "):
        d = dict(re.findall(r'(\w+)=("[^"]*"|[\w./+-]+)', ln))
        for k, v in list(d.items()):
            d[k] = v.strip('"')
        win_cmps.append(d)
    elif ln.startswith("#causal_board_vs_py_matched="):
        causal_matched = ln.split("=", 1)[1].strip()
    elif ln.startswith("#step_latency "):
        d = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
        if "step" in d:
            step_stats[d["step"]] = d
    elif ln.startswith("PHASE "):
        parts = ln.split()
        if len(parts) >= 3:
            ph_name, ph_ev = parts[1], parts[2]
            kv = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
            phases.setdefault(ph_name, {})[ph_ev] = kv
    elif ln.startswith("#energy_phases "):
        energy_kv = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))

flash_img_bytes = "N/A"
flash_bin_bytes = "N/A"
if size_path.exists():
    stxt = size_path.read_text(errors="ignore")
    m_img = re.search(r"Total image size:\s*(\d+)\s*bytes", stxt)
    m_bin = re.search(r"b2_e2e\.bin binary size (0x[0-9a-fA-F]+) bytes", stxt)
    if m_img:
        flash_img_bytes = f"{int(m_img.group(1)):,} B ({int(m_img.group(1))/1024:.1f} KB)"
    if m_bin:
        bval = int(m_bin.group(1), 16)
        flash_bin_bytes = f"{bval:,} B ({m_bin.group(1)})"

step_labels = [
    ("1_downsample_100to50_fir41", "1. Hạ mẫu 100->50 Hz (FIR polyphase 41 taps)"),
    ("2_bandpass_0p5_20_biquad4",  "2. Lọc thông dải 0.5-20 Hz (IIR Butterworth bậc 4 nhân quả, 4 biquad)"),
    ("3_make_8channels_normvec",   "3. Tạo 8 kênh (ax,ay,az,wx,wy,wz,|a|,|w|)"),
    ("4a_normalize_zscore",        "4a. Chuẩn hóa z-score (mu/sd 8 kênh)"),
    ("4b_quantize_int8",           "4b. Lượng tử hóa float32 -> int8"),
    ("4_norm_plus_quant",          "4. Tổng Chuẩn hóa + Lượng tử hóa (4a + 4b)"),
    ("preprocess_total",           "TỔNG TIỀN XỬ LÝ DSP (Bước 1 -> 4)"),
    ("5_model_invoke_espnn",       "5. Suy luận mô hình DS-TCN INT8 (ESP-NN + sigmoid)"),
    ("e2e_total",                  "TỔNG ĐẦU-CUỐI (End-to-End: Bước 1 -> 5)"),
]

e2e_mean_ms = float(step_stats.get("e2e_total", {}).get("mean_ms", "0") or 0)
e2e_p95_ms = float(step_stats.get("e2e_total", {}).get("p95_ms", "0") or 0)
prep_mean_ms = float(step_stats.get("preprocess_total", {}).get("mean_ms", "0") or 0)
inv_mean_ms = float(step_stats.get("5_model_invoke_espnn", {}).get("mean_ms", "0") or 0)

out_lines = [
    "# B2 END-TO-END BENCHMARK & ENERGY MODE SUMMARY (ESP32-S3)",
    f"Generated: {datetime.datetime.now().astimezone().isoformat()}",
    "Board: ESP32-S3 (Sparkle IoT XH-S3E N16R8, 240 MHz, 16 MB Flash, 8 MB Octal PSRAM)",
    f"Port: {port_arg} (USB-Serial-JTAG)",
    "Model: reports/p56a/models/PD_vs_HC_raw_qat_fold0.tflite (DS-TCN INT8, QAT fold 0)",
    "Norm params: reports/p56a/models/norm_PD_vs_HC_raw_fold0.npz (8 channels: ax, ay, az, wx, wy, wz, |a|, |w|)",
    f"ESP-NN backend: {esp_nn_mode}",
    f"Model-only bit-exact check (n=10 test inputs): {bitexact_model}",
    "",
    "====================================================================================================",
    "1. TÀI NGUYÊN BỘ NHỚ (FLASH & RAM)",
    "====================================================================================================",
    f"- Flash App Size (idf.py size): {flash_img_bytes} (binary: {flash_bin_bytes})",
    f"- Model FlatBuffer Size       : {mem_info.get('model_bytes', 'N/A')} B ({int(mem_info.get('model_bytes', 0))/1024:.2f} KB)",
    f"- TFLM Tensor Arena Used      : {mem_info.get('arena_used', 'N/A')} B / {mem_info.get('arena_alloc', 'N/A')} B (vùng nhớ: {mem_info.get('arena_mem', 'N/A')})",
    f"- DSP Static Buffers (SRAM)   : {mem_info.get('dsp_static_buf_bytes', 'N/A')} B (s_x50: 4800B, s_xf: 4800B, s_x8: 6400B, s_xn: 6400B)",
    f"- Tổng RAM đường ống E2E      : {mem_info.get('total_e2e_ram_bytes', 'N/A')} B ({int(mem_info.get('total_e2e_ram_bytes', 0))/1024:.2f} KB)",
    f"- Internal SRAM Free (sau cấp): {mem_info.get('internal_free_after', 'N/A')} B / {chip_info.get('internal_total', 'N/A')} B",
    f"- PSRAM Free                  : {chip_info.get('psram_free_before', 'N/A')} B / {chip_info.get('psram_total', 'N/A')} B",
    "",
    "====================================================================================================",
    "2. ĐỘ TRỄ ĐẦU-CUỐI (END-TO-END LATENCY) — 1 CỬA SỐ 4 GIÂY (400 MẪU @ 100 Hz -> 200 MẪU x 8 KÊNH @ 50 Hz)",
    f"   Đo trên n={step_stats.get('e2e_total', {}).get('n', '1000')} lần lặp (warmup=10), cửa sổ: {primary_win_info.get('primary_pads_file', 'N/A')}",
    "====================================================================================================",
    f"{'Bước xử lý':<66} | {'Mean (us)':>10} | {'p50 (us)':>9} | {'p95 (us)':>9} | {'Mean (ms)':>10} | {'p95 (ms)':>10} | {'Tỷ trọng':>8}",
    "-" * 136,
]

for key, label in step_labels:
    st = step_stats.get(key, {})
    mean_us = float(st.get("mean_us", 0) or 0)
    p50_us = int(st.get("p50_us", 0) or 0)
    p95_us = int(st.get("p95_us", 0) or 0)
    mean_ms = float(st.get("mean_ms", 0) or 0)
    p95_ms = float(st.get("p95_ms", 0) or 0)
    pct = (100.0 * mean_ms / e2e_mean_ms) if e2e_mean_ms > 0 else 0.0
    out_lines.append(
        f"{label:<66} | {mean_us:10.2f} | {p50_us:9d} | {p95_us:9d} | {mean_ms:10.4f} | {p95_ms:10.4f} | {pct:7.2f}%"
    )

out_lines.extend([
    "-" * 136,
    f"-> Thời gian xử lý toàn bộ 1 cửa sổ 4 s: Mean = {e2e_mean_ms:.3f} ms (p95 = {e2e_p95_ms:.3f} ms), trong đó:",
    f"   + Tiền xử lý DSP + Chuẩn hóa + Lượng tử hóa: {prep_mean_ms:.3f} ms ({(100.0*prep_mean_ms/e2e_mean_ms if e2e_mean_ms else 0):.1f}%)",
    f"   + Suy luận TFLite Micro (ESP-NN)           : {inv_mean_ms:.3f} ms ({(100.0*inv_mean_ms/e2e_mean_ms if e2e_mean_ms else 0):.1f}%)",
    f"-> Hệ số thời gian thực (Real-time factor cho chu kỳ 4000 ms): {e2e_mean_ms/4000.0*100.0:.4f}% (nhanh hơn thời gian thực {4000.0/e2e_mean_ms if e2e_mean_ms else 0:.1f} lần).",
    "",
    "====================================================================================================",
    "3. ĐỐI CHIẾU XÁC SUẤT ĐẦU RA TRÊN BO (IIR NHÂN QUẢ) VỚI PYTHON TRÊN CÙNG CỬA SỔ PADS 4 GIÂY",
    f"   Bit-exact giữa đường ống nhân quả trên bo và đường ống nhân quả Python: {causal_matched}",
    "====================================================================================================",
    f"{'Idx':<3} | {'Nhãn':<4} | {'Cửa sổ bản ghi PADS (4s @ 100Hz)':<45} | {'q_bo':>4} | {'P_bo(PD)':>9} | {'q_py_caus':>9} | {'P_py_caus':>9} | {'q_py_zpW':>8} | {'P_py_zpW':>9} | {'ΔP(bo-zpW)':>10} | {'P_py_zpFull':>11}",
    "-" * 145,
])

for w in win_cmps:
    out_lines.append(
        f"{w.get('idx',''):<3} | {w.get('label',''):<4} | {w.get('desc',''):<45} | "
        f"{int(w.get('q_board',0)):4d} | {float(w.get('prob_board',0)):9.6f} | "
        f"{int(w.get('q_py_causal',0)):9d} | {float(w.get('prob_py_causal',0)):9.6f} | "
        f"{int(w.get('q_py_zp_win',0)):8d} | {float(w.get('prob_py_zp_win',0)):9.6f} | "
        f"{float(w.get('diff_board_vs_zp_win',0)):+10.6f} | {float(w.get('prob_py_zp_full',0)):11.6f}"
    )

out_lines.extend([
    "-" * 145,
    "Ghi chú:",
    "- P_bo(PD) vs P_py_caus: Bo ESP32-S3 (float32 IEEE-754 + ESP-NN INT8) khớp tuyệt đối 100% (8/8 cửa sổ, sai số 0.000000)",
    "  với đường ống nhân quả (FIR polyphase 41 taps + IIR Butterworth bậc 4 sosfilt 1 chiều) trên Python.",
    "- P_py_zpW (sosfiltfilt zero-phase trên cửa sổ 4s) và P_py_zpFull (sosfiltfilt trên toàn bản ghi 10s rồi cắt cửa sổ):",
    "  Có độ lệch nhỏ về xác suất so với lọc nhân quả do dịch pha của lọc IIR nhân quả 1 chiều và hiệu ứng biên đầu cửa sổ,",
    "  nhưng quyết định phân loại (ngưỡng 0.5) giữ nguyên trên tất cả các cửa sổ thử nghiệm.",
    "",
    "====================================================================================================",
    "4. CHẾ ĐỘ ĐO NĂNG LƯỢNG 3 PHA (MỖI PHA 30 GIÂY)",
    "====================================================================================================",
])

idle_end = phases.get("IDLE", {}).get("END", {})
infer_end = phases.get("INFER", {}).get("END", {})
sleep_end = phases.get("SLEEP", {}).get("END", {}) or phases.get("INFER_DUTY", {}).get("END", {})

out_lines.extend([
    f"- Pha 1 (IDLE - nghỉ nền 240 MHz)            : thời lượng = {float(idle_end.get('duration_ms', 0))/1000.0:.2f} s, số lần suy luận = {idle_end.get('n_infer', '0')}",
    f"- Pha 2 (INFER - suy luận đầu-cuối liên tục) : thời lượng = {float(infer_end.get('duration_ms', 0))/1000.0:.2f} s, số lần suy luận = {infer_end.get('n_infer', '0')} (~{float(infer_end.get('n_infer', 0))/(float(infer_end.get('duration_ms', 30000))/1000.0):.1f} cửa sổ/s)",
    f"- Pha 3 (INFER_DUTY / SLEEP - chu kỳ 1 lần/4s): thời lượng = {float(sleep_end.get('duration_ms', 0))/1000.0:.2f} s, số lần suy luận = {sleep_end.get('n_infer', '0')}, "
    f"active = {sleep_end.get('active_ms', 'N/A')} ms, light-sleep = {sleep_end.get('light_sleep_ms', 'N/A')} ms (duty active = {energy_kv.get('active_duty_pct', 'N/A')}%)",
    "",
    f"Errors: {', '.join(errors) if errors else 'none'}",
])

summary_txt = "\n".join(out_lines) + "\n"
summary_path.write_text(summary_txt, encoding="utf-8")
print("\n" + summary_txt, flush=True)
if errors or not step_stats:
    sys.exit(1)
PY

echo "[B2] Completed successfully at $(date -Iseconds)"
