#!/usr/bin/env bash
cd "$HOME/Desktop/PARKINSON"; export PATH="$HOME/.local/bin:$PATH"
cp tools/gitignore.txt .gitignore
git ls-files firmware | awk -F/ '{print $2"/"$3}' | sort | uniq -c | sort -rn | head -12
git rm -r -q --cached firmware >/dev/null 2>&1
git checkout -q --orphan clean
git add -A
git commit -q -m "Initial import: scripts, Kaggle notebooks, firmware sources, results, manuscript (BSPC R6)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WqnLXUDtvpnoCRAFWVXAPF"
git branch -D main -q; git branch -m main
git reflog expire --expire=now --all; git gc -q --prune=now --aggressive
git push -q -f origin main 2>&1 | tail -3
echo "---after"; git log --oneline | head; git ls-files | wc -l
git ls-files firmware | awk -F/ '{print $2"/"$3}' | sort | uniq -c | sort -rn | head -12
git count-objects -vH | grep size-pack
