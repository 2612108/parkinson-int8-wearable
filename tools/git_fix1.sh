#!/usr/bin/env bash
cd "$HOME/Desktop/PARKINSON"; export PATH="$HOME/.local/bin:$PATH"
cp tools/gitignore.txt .gitignore
git rm -r -q --cached firmware/*/build* 2>/dev/null
bash tools/git_sync.sh "Ignore firmware build directories"
git ls-files | wc -l; git ls-files | awk -F/ '{print $1}' | sort | uniq -c
gh api repos/2612108/parkinson-int8-wearable --jq '.size, .private, .default_branch'
