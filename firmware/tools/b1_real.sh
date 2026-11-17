#!/usr/bin/env bash
# B1-real: Đóng gói + build + nạp + đo 3 mô hình thật (ptq, qat, qat_sr) × 2 cấu hình (nn, ansi) trên ESP32-S3
set -u

ROOT="$HOME/Desktop/PARKINSON"
cd "$ROOT" || exit 1
mkdir -p "$ROOT/reports/board"

PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1)
PORT="${PORT:-/dev/ttyACM0}"
echo "[B1-real] Starting at $(date -Iseconds) on PORT=$PORT"

export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"
if ! "$ROOT/.venv/bin/python" -c "import ai_edge_litert" >/dev/null 2>&1; then
  echo "[B1-real] Installing ai-edge-litert flatbuffers into firmware/.pydeps..."
  "$HOME/.local/bin/uv" pip install --target "$ROOT/firmware/.pydeps" --python "$ROOT/.venv/bin/python" ai-edge-litert flatbuffers
fi

INPUTS="reports/p56a/models/bench_inputs_PD_vs_HC_raw.npy"
MODELS=(ptq qat qat_sr)

run_one_mode() {
  local ten="$1"
  local mode="$2"
  local gen_dir="firmware/b1_bench/gen_real_${ten}"
  local tag="real_${ten}"
  local json_out="$ROOT/reports/board/B1_${tag}_${mode}.json"

  echo "[B1-real] Running run_bench.sh for tag=${tag} mode=${mode}..."
  if ! bash firmware/tools/run_bench.sh "$gen_dir" "$tag" "$mode" "$PORT"; then
    echo "[B1-real] ERROR: run_bench.sh failed for ${tag} (${mode})"
    return 1
  fi

  # Tự động xử lý #ERR AllocateTensors theo ANTIGRAVITY_B1.md nếu có
  if [ -f "$json_out" ] && grep -q "AllocateTensors" "$json_out"; then
    echo "[B1-real] Detected #ERR AllocateTensors in ${tag}_${mode}, increasing ARENA_BYTES in main.cc..."
    sed -i 's/#define ARENA_BYTES (160 \* 1024)/#define ARENA_BYTES (256 \* 1024)/g' "$ROOT/firmware/b1_bench/main/main.cc"
    bash firmware/tools/run_bench.sh "$gen_dir" "$tag" "$mode" "$PORT" || return 1
  fi
  return 0
}

for ten in "${MODELS[@]}"; do
  model_file="reports/p56a/models/PD_vs_HC_raw_${ten}_fold0.tflite"
  gen_dir="firmware/b1_bench/gen_real_${ten}"
  echo "============================================================"
  echo "[B1-real] [1/2] Generating C files for model=${ten} (${model_file})"
  if ! .venv/bin/python firmware/tools/tflite_to_c.py \
      --model "$model_file" \
      --inputs "$INPUTS" \
      --n 50 \
      --out "$gen_dir"; then
    echo "[B1-real] ERROR: tflite_to_c.py failed for ${ten}"
    continue
  fi

  echo "[B1-real] [2/2] Benchmarking model=${ten} (nn then ansi)"
  run_one_mode "$ten" nn
  run_one_mode "$ten" ansi
done

echo "============================================================"
echo "[B1-real] Generating reports/board/B1_REAL_SUMMARY.txt..."
.venv/bin/python - <<'PY'
import json, pathlib, re, datetime

root = pathlib.Path.home() / "Desktop" / "PARKINSON"
board = root / "reports" / "board"
models = ["ptq", "qat", "qat_sr"]
modes = ["nn", "ansi"]

lines = [
    "# B1 REAL MODELS BENCHMARK SUMMARY (ESP32-S3)",
    f"Generated: {datetime.datetime.now().astimezone().isoformat()}",
    "Models: PD_vs_HC_raw_{ptq,qat,qat_sr}_fold0.tflite (DS-TCN INT8, stride (2,1), no dilation)",
    "Inputs: reports/p56a/models/bench_inputs_PD_vs_HC_raw.npy (n=50)",
    "",
    f"{'Model':<10} | {'Config':<6} | {'Bit-Exact':<10} | {'MaxLSB':<6} | {'Latency Mean / p95 (ms)':<25} | {'Arena Used (B)':<14} | {'Flash App Bytes (Image / Bin)':<30} | {'Errors'}",
    "-" * 125,
]

for m in models:
    tag = f"real_{m}"
    for mode in modes:
        jpath = board / f"B1_{tag}_{mode}.json"
        spath = board / f"B1_{tag}_{mode}_size.txt"
        bpath = board / f"B1_{tag}_{mode}_build.log"
        fpath = board / f"B1_{tag}_{mode}_flash.log"

        bitexact = "N/A"
        maxlsb = "N/A"
        lat_str = "N/A"
        arena = "N/A"
        flash_bytes = "N/A"
        errs = []

        if spath.exists():
            stxt = spath.read_text(errors="ignore")
            m_img = re.search(r"Total image size:\s*(\d+)\s*bytes", stxt)
            m_bin = re.search(r"b1_bench\.bin binary size (0x[0-9a-fA-F]+) bytes", stxt)
            img_b = m_img.group(1) if m_img else "?"
            bin_b = f"{int(m_bin.group(1), 16)} ({m_bin.group(1)})" if m_bin else "?"
            flash_bytes = f"{img_b} / {bin_b}"
        else:
            errs.append("missing_size_txt")

        if jpath.exists():
            try:
                d = json.loads(jpath.read_text(errors="ignore"))
                kv = d.get("kv", {})
                n_be = d.get("n_bitexact")
                n_ref = d.get("n_ref", 50)
                n_rec = d.get("n_received", 0)
                bitexact = f"{n_be}/{n_ref}" if n_be is not None else kv.get("matched", "N/A")
                if n_rec < n_ref:
                    errs.append(f"received_{n_rec}/{n_ref}")
                maxlsb = str(d.get("max_abs_lsb_diff", "N/A"))
                mean_us = kv.get("mean")
                p95_us = kv.get("p95")
                if mean_us is not None and p95_us is not None:
                    mean_ms = float(mean_us) / 1000.0
                    p95_ms = float(p95_us) / 1000.0
                    lat_str = f"{mean_ms:.3f} / {p95_ms:.3f} ms"
                arena = str(kv.get("arena_used", "N/A"))
                if d.get("errors"):
                    errs.extend(d["errors"])
            except Exception as e:
                errs.append(f"json_err:{e}")
        else:
            errs.append("missing_json")
            if bpath.exists() and "error:" in bpath.read_text(errors="ignore").lower():
                errs.append("build_error")
            if fpath.exists() and "error" in fpath.read_text(errors="ignore").lower():
                errs.append("flash_error")

        err_str = "; ".join(errs) if errs else "none"
        lines.append(
            f"{m:<10} | {mode:<6} | {bitexact:<10} | {maxlsb:<6} | {lat_str:<25} | {arena:<14} | {flash_bytes:<30} | {err_str}"
        )

summary_txt = "\n".join(lines) + "\n"
(board / "B1_REAL_SUMMARY.txt").write_text(summary_txt)
print(summary_txt)
PY
echo "[B1-real] Finished at $(date -Iseconds)"
