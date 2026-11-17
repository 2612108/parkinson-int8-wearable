#!/usr/bin/env bash
# B3: Mô hình nhân quả (reports/p56k0/models/PD_vs_HC_raw_qat_fold0.tflite)
# Phát lại TOÀN BỘ bản ghi PADS thật 100 Hz theo khối 2 s, cửa sổ 4 s hop 2 s,
# lọc Butterworth bậc 4 0.5-20 Hz nhân quả LIÊN TỤC (giữ trạng thái biquad qua các khối).
# Bỏ pha năng lượng. Sau khi B3 hoàn tất, tự động kích hoạt B4 chạy nền tuần tự.
set -euo pipefail

ROOT="$HOME/Desktop/PARKINSON"
PROJ="$ROOT/firmware/b3_causal"
BOARD_DIR="$ROOT/reports/board"
mkdir -p "$BOARD_DIR"

CHAIN_B4="${1:-chain}"
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)
PORT="${PORT:-/dev/ttyACM0}"

echo "[B3] Starting at $(date -Iseconds) on PORT=$PORT"

export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"

echo "[B3] Step 1: Generating C data via firmware/tools/gen_b3k_data.py..."
"$ROOT/.venv/bin/python" "$ROOT/firmware/tools/gen_b3k_data.py"

echo "[B3] Step 2: Sourcing ESP-IDF environment & building firmware/b3_causal..."
# shellcheck disable=SC1091
. "$HOME/esp/esp-idf/export.sh" >/dev/null 2>&1

cd "$PROJ"
if ! idf.py build > "$BOARD_DIR/B3k_build.log" 2>&1; then
  echo "[B3] ERROR: Build failed. Last 40 lines of B3k_build.log:"
  tail -40 "$BOARD_DIR/B3k_build.log"
  exit 1
fi
echo "[B3] Build succeeded."

idf.py size > "$BOARD_DIR/B3k_size.txt" 2>&1 || true

echo "[B3] Step 3: Flashing firmware/b3_causal to $PORT..."
if ! idf.py -p "$PORT" flash > "$BOARD_DIR/B3k_flash.log" 2>&1; then
  echo "[B3] ERROR: Flash failed. Last 30 lines of B3k_flash.log:"
  tail -30 "$BOARD_DIR/B3k_flash.log"
  exit 1
fi
echo "[B3] Flash succeeded."

cd "$ROOT"
echo "[B3] Step 4: Capturing serial log from $PORT..."

"$ROOT/.venv/bin/python" - "$PORT" "$BOARD_DIR/B3k_serial.log" "$BOARD_DIR/B3k_SUMMARY.txt" "$BOARD_DIR/B3k_size.txt" <<'PY'
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
            if ln.startswith("#B3 END") or ln.startswith("#ERR"):
                break
        if any(x.startswith("#B3 END") or x.startswith("#ERR") for x in lines):
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
print(f"[B3] Saved raw serial log ({len(lines)} lines) to {log_path}", flush=True)

chip_info = {}
mem_info = {}
quant_info = {}
esp_nn_mode = "unknown"
bitexact_model = "N/A"
rec_begins = {}
rec_summaries = {}
win_streams = []
overall_info = {}
step_stats = {}
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
    elif ln.startswith("#record_begin "):
        d = dict(re.findall(r'(\w+)=("[^"]*"|[\w./+-]+)', ln))
        for k, v in list(d.items()):
            d[k] = v.strip('"')
        rec_begins[int(d.get("rec_idx", 0))] = d
    elif ln.startswith("#win_stream "):
        d = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
        win_streams.append(d)
    elif ln.startswith("#record_summary "):
        d = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
        rec_summaries[int(d.get("rec_idx", 0))] = d
    elif ln.startswith("#b3_stream_overall "):
        overall_info = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
    elif ln.startswith("#step_latency "):
        d = dict(re.findall(r"(\w+)=([\w./+-]+)", ln))
        if "step" in d:
            step_stats[d["step"]] = d

flash_img_bytes = "N/A"
flash_bin_bytes = "N/A"
dram_static_bytes = "N/A"
if size_path.exists():
    stxt = size_path.read_text(errors="ignore")
    m_img = re.search(r"Total image size:\s*(\d+)\s*bytes", stxt)
    m_bin = re.search(r"b3_causal\.bin binary size (0x[0-9a-fA-F]+) bytes", stxt)
    m_dram = re.search(r"Used static DRAM:\s*(\d+)\s*bytes", stxt)
    if m_img:
        flash_img_bytes = f"{int(m_img.group(1)):,} B ({int(m_img.group(1))/1024:.1f} KB)"
    if m_bin:
        bval = int(m_bin.group(1), 16)
        flash_bin_bytes = f"{bval:,} B ({m_bin.group(1)})"
    if m_dram:
        dram_static_bytes = f"{int(m_dram.group(1)):,} B ({int(m_dram.group(1))/1024:.1f} KB)"

step_labels = [
    ("blk_1_downsample_2s_100to50", "1. [Mỗi khối 2s] Hạ mẫu 100->50 Hz (FIR polyphase 41 taps, 200->100 mẫu x 6 kênh)"),
    ("blk_2_biquad_continuous_2s",  "2. [Mỗi khối 2s] Lọc Butterworth bậc 4 0.5-20 Hz nhân quả LIÊN TỤC (giữ trạng thái z1,z2)"),
    ("blk_3_make8ch_shift_win_2s",  "3. [Mỗi khối 2s] Dịch cửa sổ trượt 4s (hop 100 mẫu) + tính 8 kênh (100 mẫu mới)"),
    ("blk_preprocess_2s_total",     "-> TỔNG TIỀN XỬ LÝ MỖI KHỐI 2 GIÂY (Bước 1 + 2 + 3)"),
    ("win_4_norm_quant_4s",         "4. [Mỗi lần suy luận 4s] Chuẩn hóa z-score (mu/sd p56x4) + lượng tử hóa INT8 (200x8)"),
    ("win_5_model_invoke_espnn",    "5. [Mỗi lần suy luận 4s] Suy luận mô hình DS-TCN INT8 (ESP-NN + sigmoid)"),
    ("win_infer_4s_total",          "-> TỔNG MỖI LẦN SUY LUẬN CỬA SỔ 4 GIÂY (Bước 4 + 5)"),
    ("hop_2s_active_total",         "-> TỔNG THỜI GIAN HOẠT ĐỘNG MỖI CHU KỲ HOP 2 GIÂY (Khối 2s + Suy luận cửa sổ 4s)"),
]

blk_mean_ms = float(step_stats.get("blk_preprocess_2s_total", {}).get("mean_ms", 0) or 0)
blk_p95_ms = float(step_stats.get("blk_preprocess_2s_total", {}).get("p95_ms", 0) or 0)
inf_mean_ms = float(step_stats.get("win_infer_4s_total", {}).get("mean_ms", 0) or 0)
inf_p95_ms = float(step_stats.get("win_infer_4s_total", {}).get("p95_ms", 0) or 0)
inv_mean_ms = float(step_stats.get("win_5_model_invoke_espnn", {}).get("mean_ms", 0) or 0)
inv_p95_ms = float(step_stats.get("win_5_model_invoke_espnn", {}).get("p95_ms", 0) or 0)
hop_mean_ms = float(step_stats.get("hop_2s_active_total", {}).get("mean_ms", 0) or 0)
hop_p95_ms = float(step_stats.get("hop_2s_active_total", {}).get("p95_ms", 0) or 0)

rec0_sum = rec_summaries.get(0, {})

out_lines = [
    "# B3 CAUSAL CONTINUOUS STREAMING MODEL SUMMARY (ESP32-S3)",
    f"Generated: {datetime.datetime.now().astimezone().isoformat()}",
    "Board: ESP32-S3 (Sparkle IoT XH-S3E N16R8, 240 MHz, 16 MB Flash, 8 MB Octal PSRAM)",
    f"Port: {port_arg} (USB-Serial-JTAG)",
    "Model: reports/p56k0/models/PD_vs_HC_raw_qat_fold0.tflite (DS-TCN INT8, QAT fold 0 trained with --causal)",
    "Norm params: reports/p56k0/models/norm_PD_vs_HC_raw_fold0.npz (8 channels: ax, ay, az, wx, wy, wz, |a|, |w|)",
    f"ESP-NN backend: {esp_nn_mode}",
    f"Model-only bit-exact check (n=10 bench_inputs from p56x4): {bitexact_model}",
    "",
    "====================================================================================================",
    "1. TÀI NGUYÊN BỘ NHỚ (FLASH & RAM)",
    "====================================================================================================",
    f"- Flash App Size (idf.py size)  : {flash_img_bytes} (binary: {flash_bin_bytes})",
    f"- Model FlatBuffer Size         : {mem_info.get('model_bytes', 'N/A')} B ({int(mem_info.get('model_bytes', 0))/1024:.2f} KB)",
    f"- TFLM Tensor Arena Used        : {mem_info.get('arena_used', 'N/A')} B / {mem_info.get('arena_alloc', 'N/A')} B (vùng nhớ: {mem_info.get('arena_mem', 'N/A')})",
    f"- DSP Streaming Buffers (SRAM)  : {mem_info.get('dsp_static_buf_bytes', 'N/A')} B (khối 2s s_blk50: 2400B, s_blkbp: 2400B, trạng thái biquad s_biquad_z: {mem_info.get('biquad_state_bytes', '192')}B, cửa sổ 4s s_win_x8: 6400B, s_win_xn: 6400B)",
    f"- Tổng RAM đường ống Streaming  : {mem_info.get('total_e2e_ram_bytes', 'N/A')} B ({int(mem_info.get('total_e2e_ram_bytes', 0))/1024:.2f} KB)",
    f"- Used Static DRAM (idf.py size): {dram_static_bytes}",
    f"- Internal SRAM Free (sau cấp)  : {mem_info.get('internal_free_after', 'N/A')} B / {chip_info.get('internal_total', 'N/A')} B",
    "",
    "====================================================================================================",
    "2. ĐỘ TRỄ MỖI KHỐI 2 GIÂY (TIỀN XỬ LÝ) & MỖI LẦN SUY LUẬN CỬA SỔ 4 GIÂY (HOP 2 GIÂY)",
    f"   Đo trên 100 lần phát lại toàn bộ bản ghi {rec_begins.get(0, {}).get('file', '005_Relaxed_LeftWrist.txt')} (1000 khối 2s, 900 lần suy luận cửa sổ 4s)",
    "====================================================================================================",
    f"{'Bước xử lý':<84} | {'Mean (us)':>10} | {'p50 (us)':>9} | {'p95 (us)':>9} | {'Mean (ms)':>10} | {'p95 (ms)':>10}",
    "-" * 144,
]

for key, label in step_labels:
    st = step_stats.get(key, {})
    mean_us = float(st.get("mean_us", 0) or 0)
    p50_us = int(st.get("p50_us", 0) or 0)
    p95_us = int(st.get("p95_us", 0) or 0)
    mean_ms = float(st.get("mean_ms", 0) or 0)
    p95_ms = float(st.get("p95_ms", 0) or 0)
    out_lines.append(
        f"{label:<84} | {mean_us:10.2f} | {p50_us:9d} | {p95_us:9d} | {mean_ms:10.4f} | {p95_ms:10.4f}"
    )

out_lines.extend([
    "-" * 144,
    f"-> Độ trễ tiền xử lý mỗi khối 2 s (200 mẫu @ 100 Hz -> 100 mẫu @ 50 Hz): Mean = {blk_mean_ms:.3f} ms, p95 = {blk_p95_ms:.3f} ms",
    f"-> Độ trễ mỗi lần suy luận cửa sổ 4 s (Chuẩn hóa + Lượng tử hóa + ESP-NN): Mean = {inf_mean_ms:.3f} ms, p95 = {inf_p95_ms:.3f} ms (riêng invoke ESP-NN: Mean = {inv_mean_ms:.3f} ms, p95 = {inv_p95_ms:.3f} ms)",
    f"-> Tổng thời gian xử lý mỗi chu kỳ hop 2 s (2000 ms)                     : Mean = {hop_mean_ms:.3f} ms, p95 = {hop_p95_ms:.3f} ms (chiếm {hop_mean_ms/2000.0*100.0:.4f}% chu kỳ 2 s)",
    "",
    "====================================================================================================",
    "3. ĐỐI CHIẾU XÁC SUẤT TỪNG CỬA SỔ TRÊN BO VỚI ĐƯỜNGỐNG PYTHON CHUẨN (SCIPY RESAMPLE_POLY + SOSFILT CẢ BẢN GHI)",
    f"   - Bản ghi chính ({rec0_sum.get('file', '005_Relaxed_LeftWrist.txt')}): Bit-exact = {rec0_sum.get('bit_exact', 'N/A')} cửa sổ, sai khác xác suất tối đa = {float(rec0_sum.get('max_abs_dp', 0)):.6f} (max |Δq| = {rec0_sum.get('max_abs_dq', 0)} LSB)",
    f"   - Tổng hợp cả 3 bản ghi đầy đủ (27 cửa sổ x 4s, hop 2s)              : Bit-exact = {overall_info.get('all_records_bit_exact', 'N/A')} cửa sổ, sai khác xác suất tối đa = {float(overall_info.get('max_abs_dp', 0)):.6f} (max |Δq| = {overall_info.get('max_abs_dq', 0)} LSB)",
    "====================================================================================================",
    f"{'Rec':<3} | {'Win':<3} | {'Khung giờ':<10} | {'q_bo':>4} | {'P_bo(PD)':>9} | {'q_py_cont':>9} | {'P_py_cont':>9} | {'Exact':>5} | {'|ΔP_cont|':>9} | {'P_py_reset':>10} | {'ΔP(bo-rst)':>10} | {'P_py_zp':>9} | {'ΔP(bo-zp)':>10}",
    "-" * 138,
])

for w in win_streams:
    t_rng = f"{float(w.get('t_start_s',0)):4.1f}..{float(w.get('t_end_s',0)):4.1f}s"
    out_lines.append(
        f"{w.get('rec',''):<3} | {w.get('win',''):<3} | {t_rng:<10} | "
        f"{int(w.get('q_board',0)):4d} | {float(w.get('prob_board',0)):9.6f} | "
        f"{int(w.get('q_py_cont',0)):9d} | {float(w.get('prob_py_cont',0)):9.6f} | "
        f"{'YES' if int(w.get('bit_exact',0)) else 'NO':>5} | {float(w.get('abs_dp',0)):9.6f} | "
        f"{float(w.get('prob_py_reset',0)):10.6f} | {float(w.get('diff_vs_reset',0)):+10.6f} | "
        f"{float(w.get('prob_py_zp',0)):9.6f} | {float(w.get('diff_vs_zp',0)):+10.6f}"
    )

out_lines.extend([
    "-" * 138,
    "Tóm tắt theo từng bản ghi đầy đủ (20.46 s @ 100 Hz -> 10 khối 2 s -> 9 cửa sổ 4 s hop 2 s):",
])
for r_idx, rs in sorted(rec_summaries.items()):
    rb = rec_begins.get(r_idx, {})
    out_lines.append(
        f"  + Rec {r_idx} ({rb.get('desc', rs.get('file', ''))}): "
        f"Bit-exact = {rs.get('bit_exact', 'N/A')}, max |ΔP| = {float(rs.get('max_abs_dp', 0)):.6f}, "
        f"mean P_bo(PD) = {float(rs.get('mean_prob_board', 0)):.6f} vs mean P_py(PD) = {float(rs.get('mean_prob_py', 0)):.6f}"
    )

out_lines.extend([
    "",
    f"Errors: {', '.join(errors) if errors else 'none'}",
])

summary_txt = "\n".join(out_lines) + "\n"
summary_path.write_text(summary_txt, encoding="utf-8")
print("\n" + summary_txt, flush=True)
if errors or not step_stats:
    sys.exit(1)
PY

echo "[B3] Completed successfully at $(date -Iseconds)"

if [ "$CHAIN_B4" = "chain" ]; then
  echo "[B3] Launching B4 sequentially in background -> $BOARD_DIR/B4_run.log"
  setsid nohup bash "$ROOT/firmware/tools/b4_run.sh" > "$BOARD_DIR/B4_run.log" 2>&1 < /dev/null &
  echo "[B3] B4 launched with PID=$!"
fi
