#!/usr/bin/env bash
# B6k: whole fold-0 replay on the board with the specified causal run p56k0 (QAT and PTQ) and causal preprocessing.
# nohup bash firmware/tools/b6_run.sh > reports/board/B6_run.log 2>&1 &
set -u
ROOT="$HOME/Desktop/PARKINSON"; cd "$ROOT" || exit 1
R="$ROOT/reports/board"; P="$ROOT/firmware/b6_fold"; RUN=reports/p56k0
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1); PORT="${PORT:-/dev/ttyACM0}"
export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"
echo "[B6k] start $(date -Iseconds)"
for V in qat ptq; do
  .venv/bin/python firmware/tools/tflite_to_c.py --model $RUN/models/PD_vs_HC_raw_${V}_fold0.tflite --inputs $RUN/models/bench_inputs_PD_vs_HC_raw.npy --n 4 --out "$P/main/generated" > "$R/B6k_${V}_gen.log" 2>&1 || { echo gen fail; continue; }
  ( . "$HOME/esp/esp-idf/export.sh" >/dev/null; cd "$P" && idf.py build > "$R/B6k_${V}_build.log" 2>&1 && idf.py -p "$PORT" flash > "$R/B6k_${V}_flash.log" 2>&1 ) || { echo "[B6k] build/flash $V FAILED"; tail -30 "$R/B6k_${V}_build.log"; continue; }
  sleep 2
  .venv/bin/python firmware/tools/b6_host.py --run $RUN --variant $V --port "$PORT" --causal --out "$R/B6k_${V}" 2>&1 | tail -20
done
echo "[B6k] done $(date -Iseconds)"
