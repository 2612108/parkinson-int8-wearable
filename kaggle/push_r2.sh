#!/usr/bin/env bash
# R2: đẩy 4 kernel p56r0-2, p56rr.
cd ~/Desktop/PARKINSON || exit 1
K=""
for c in "$(command -v kaggle)" "$HOME/.local/bin/kaggle" "$HOME/Desktop/PARKINSON/.venv/bin/kaggle" "$HOME/.venv/bin/kaggle"; do
  [ -n "$c" ] && [ -x "$c" ] && { K="$c"; break; }
done
[ -z "$K" ] && K="$(ls -1 $HOME/.local/bin/kaggle $HOME/*/bin/kaggle $HOME/Desktop/*/.venv/bin/kaggle 2>/dev/null | head -1)"
echo "kaggle=$K"
for n in p56r0 p56r1 p56r2 p56rr; do
  echo "== $n $(date -Iseconds)"; "$K" kernels push -p kaggle/$n 2>&1 | tail -3
done
