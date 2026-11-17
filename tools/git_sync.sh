#!/usr/bin/env bash
# Commit and push all tracked/untracked (non-ignored) changes of the PARKINSON project to the private GitHub repo.
# usage: bash tools/git_sync.sh "commit message"
set -u
cd "$HOME/Desktop/PARKINSON" || exit 1
export PATH="$HOME/.local/bin:$PATH"
MSG="${1:-Update}"
# refuse files > 50 MB (GitHub limit 100 MB)
find . -path ./.git -prune -o -path ./.venv -prune -o -path ./data -prune -o -path ./firmware/.pydeps -prune -o -type f -size +50M -print | while read -r f; do
  git check-ignore -q "$f" || { echo "skip large: $f"; echo "${f#./}" >> .git/info/exclude; }
done
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WqnLXUDtvpnoCRAFWVXAPF" && echo "committed: $MSG"
fi
git push -q origin HEAD:main 2>&1 | tail -3
git log --oneline -1
