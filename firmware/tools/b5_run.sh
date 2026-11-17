#!/usr/bin/env bash
# B5: 16x8 model (int8 weights, int16 activations) on ESP32-S3 — bit-exactness, latency, arena; ESP-NN and ANSI builds.
# Run: nohup bash firmware/tools/b5_run.sh > reports/board/B5_run.log 2>&1 &
set -u
ROOT="$HOME/Desktop/PARKINSON"; cd "$ROOT" || exit 1
R="$ROOT/reports/board"; P="$ROOT/firmware/b5_a16"; mkdir -p "$R"
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1); PORT="${PORT:-/dev/ttyACM0}"
echo "[B5] start $(date -Iseconds) port=$PORT"
export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"
MODEL="${MODEL_OVERRIDE:-reports/p56q/reports/p56q/models/PD_vs_HC_raw_ptq_a16_fold0.tflite}"
INPUTS="reports/p56a/models/bench_inputs_PD_vs_HC_raw.npy"
.venv/bin/python firmware/tools/tflite_to_c_raw.py --model "$MODEL" --inputs "$INPUTS" --n 50 --out "$P/main/generated" \
  > "$R/B5_gen.log" 2>&1 || { echo "[B5] gen failed"; tail -20 "$R/B5_gen.log"; exit 1; }
cp "$P/main/generated/model_info.json" "$R/B5_model_info.json"
. "$HOME/esp/esp-idf/export.sh" >/dev/null
for MODE in nn ansi; do
  cd "$P"
  if [ "$MODE" = ansi ]; then B=(-B build_ansi -D SDKCONFIG=sdkconfig.ansi -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.ansi"); else B=(-B build); fi
  echo "[B5] build $MODE"
  idf.py "${B[@]}" build > "$R/B5_${MODE}_build.log" 2>&1 || { echo "[B5] build $MODE FAILED"; tail -40 "$R/B5_${MODE}_build.log"; cd "$ROOT"; continue; }
  idf.py "${B[@]}" size > "$R/B5_${MODE}_size.txt" 2>&1 || true
  idf.py "${B[@]}" -p "$PORT" flash > "$R/B5_${MODE}_flash.log" 2>&1 || { echo "[B5] flash $MODE FAILED"; tail -20 "$R/B5_${MODE}_flash.log"; cd "$ROOT"; continue; }
  cd "$ROOT"
  .venv/bin/python firmware/tools/check_b5.py --port "$PORT" --out "$R/B5_${MODE}" --timeout 900 | tail -30
done
{ echo "# B5 SUMMARY — 16x8 model $MODEL"; date -Iseconds
  for MODE in nn ansi; do echo "## $MODE"; grep -E "#(model_bytes|bitexact|latency_us|esp_nn|in_type|ERR)" "$R/B5_${MODE}.log" 2>/dev/null
    grep -E "Total image size" "$R/B5_${MODE}_size.txt" 2>/dev/null; done; } > "$R/B5_SUMMARY.txt"
cat "$R/B5_SUMMARY.txt"
echo "[B5] done $(date -Iseconds)"
