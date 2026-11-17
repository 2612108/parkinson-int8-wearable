#!/usr/bin/env bash
# B6: whole fold-0 replay on the board (specified-architecture zero-phase run p56z0: QAT and PTQ), then 16x8 latency of the same run.
# nohup bash firmware/tools/b6_run.sh > reports/board/B6_run.log 2>&1 &
set -u
ROOT="$HOME/Desktop/PARKINSON"; cd "$ROOT" || exit 1
R="$ROOT/reports/board"; P="$ROOT/firmware/b6_fold"; RUN=reports/p56z0
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1); PORT="${PORT:-/dev/ttyACM0}"
export PYTHONPATH="$ROOT/firmware/.pydeps:${PYTHONPATH:-}"
echo "[B6] start $(date -Iseconds)"
for V in qat ptq; do
  .venv/bin/python firmware/tools/tflite_to_c.py --model $RUN/models/PD_vs_HC_raw_${V}_fold0.tflite --inputs $RUN/models/bench_inputs_PD_vs_HC_raw.npy --n 4 --out "$P/main/generated" > "$R/B6_${V}_gen.log" 2>&1 || { echo gen fail; continue; }
  ( . "$HOME/esp/esp-idf/export.sh" >/dev/null; cd "$P" && idf.py build > "$R/B6_${V}_build.log" 2>&1 && idf.py -p "$PORT" flash > "$R/B6_${V}_flash.log" 2>&1 ) || { echo "[B6] build/flash $V FAILED"; tail -30 "$R/B6_${V}_build.log"; continue; }
  sleep 2
  .venv/bin/python firmware/tools/b6_host.py --run $RUN --variant $V --port "$PORT" --out "$R/B6_${V}" 2>&1 | tail -20
done
echo "[B6] 16x8 latency with specified-architecture model"
MODEL_OVERRIDE=$RUN/models/PD_vs_HC_raw_ptq_a16_fold0.tflite TAG=B5bn bash firmware/tools/b5_run.sh > "$R/B5bn_run.log" 2>&1; tail -12 "$R/B5_SUMMARY.txt"
echo "[B6] done $(date -Iseconds)"
