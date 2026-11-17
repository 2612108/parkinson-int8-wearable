#!/usr/bin/env bash
# Ghi trạng thái 6 kernel R1; nếu COMPLETE thì kéo kết quả nhỏ về reports/<name>/ (một lần).
cd ~/Desktop/PARKINSON || exit 1
K="$(command -v kaggle)"; [ -z "$K" ] && K="$(ls -1 $HOME/.local/bin/kaggle $HOME/*/bin/kaggle $HOME/Desktop/*/.venv/bin/kaggle 2>/dev/null | head -1)"
{ date -Iseconds
for n in p56q0 p56q1; do
  st=$("$K" kernels status nguynthqunhlu/parkinson-$n 2>&1 | tail -1); echo "$n: $st"
  if echo "$st" | grep -qi complete && [ ! -f reports/$n/.pulled ]; then
    mkdir -p reports/$n/_dl && "$K" kernels output nguynthqunhlu/parkinson-$n -p reports/$n/_dl --file-pattern '.*(\.csv|\.json|\.jsonl|\.log)$' >/dev/null 2>&1 \
      && cp -r reports/$n/_dl/reports/$n/. reports/$n/ 2>/dev/null; cp reports/$n/_dl/*.log reports/$n/ 2>/dev/null; touch reports/$n/.pulled; echo "  pulled $n"
  fi
done; } > reports/status_r3.log 2>&1
cat reports/status_r3.log
