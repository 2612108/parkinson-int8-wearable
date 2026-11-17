#!/usr/bin/env bash
# R4: p56t0-2, p56u0-1.
cd ~/Desktop/PARKINSON || exit 1
K=""
for c in "$(command -v kaggle)" "$HOME/.local/bin/kaggle" "$HOME/Desktop/PARKINSON/.venv/bin/kaggle" "$HOME/.venv/bin/kaggle"; do
  [ -n "$c" ] && [ -x "$c" ] && { K="$c"; break; }
done
[ -z "$K" ] && K="$(ls -1 $HOME/.local/bin/kaggle $HOME/*/bin/kaggle $HOME/Desktop/*/.venv/bin/kaggle 2>/dev/null | head -1)"
echo "kaggle=$K"
for n in p56t0 p56t1 p56t2 p56u0 p56u1; do
  echo "== $n $(date -Iseconds)"; "$K" kernels push -p kaggle/$n 2>&1 | tail -3
done
