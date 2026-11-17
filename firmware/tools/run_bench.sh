#!/usr/bin/env bash
# Một lệnh: chép mã C sinh sẵn → build (ESP-NN hoặc ANSI) → nạp → đo → JSON.
#   bash firmware/tools/run_bench.sh <thư mục generated> <tag> <nn|ansi> [port]
# Ví dụ: bash firmware/tools/run_bench.sh firmware/b1_bench/variants/gen_nodil nodil nn
set -euo pipefail
GEN="$1"; TAG="$2"; MODE="$3"; PORT="${4:-/dev/ttyACM0}"
ROOT="$HOME/Desktop/PARKINSON"; P="$ROOT/firmware/b1_bench"; R="$ROOT/reports/board"
mkdir -p "$R"
GEN="$(cd "$ROOT" && cd "$GEN" && pwd)"
rm -rf "$P/main/generated" && cp -r "$GEN" "$P/main/generated"
. "$HOME/esp/esp-idf/export.sh" >/dev/null
cd "$P"
if [ "$MODE" = ansi ]; then
  [ -f sdkconfig.defaults.ansi ] || printf 'CONFIG_NN_OPTIMIZED=n\nCONFIG_NN_ANSI_C=y\n' > sdkconfig.defaults.ansi
  B=(-B build_ansi -D SDKCONFIG=sdkconfig.ansi -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.ansi")
else
  B=(-B build)
fi
idf.py "${B[@]}" build > "$R/B1_${TAG}_${MODE}_build.log" 2>&1 || { tail -40 "$R/B1_${TAG}_${MODE}_build.log"; exit 1; }
idf.py "${B[@]}" size > "$R/B1_${TAG}_${MODE}_size.txt" 2>&1 || true
idf.py "${B[@]}" -p "$PORT" flash > "$R/B1_${TAG}_${MODE}_flash.log" 2>&1 || { tail -20 "$R/B1_${TAG}_${MODE}_flash.log"; exit 1; }
cd "$ROOT"; . .venv/bin/activate
python firmware/tools/check_serial.py --port "$PORT" --ref "$P/main/generated/ref_outputs.npy" \
  --out "$R/B1_${TAG}_${MODE}" --timeout 150 | tail -25
