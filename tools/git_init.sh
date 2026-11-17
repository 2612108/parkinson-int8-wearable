#!/usr/bin/env bash
set -eu
cd "$HOME/Desktop/PARKINSON"
export PATH="$HOME/.local/bin:$PATH"
cp tools/gitignore.txt .gitignore
gh auth setup-git
[ -d .git ] || git init -q -b main
git config user.name "Quang-Tuyen Lai"
git config user.email "2612108@student.hcmute.edu.vn"
REPO=parkinson-int8-wearable
if ! gh repo view "2612108/$REPO" >/dev/null 2>&1; then
  gh repo create "$REPO" --private --description "Eight-bit quantization of a tiny wrist-inertial network for Parkinson's disease detection (BSPC submission; private until publication)" >/dev/null
fi
git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/2612108/$REPO.git"
bash tools/git_sync.sh "Initial import: scripts, Kaggle notebooks, firmware sources, results, manuscript (BSPC R6)"
gh repo view "2612108/$REPO" --json name,visibility,url
git count-objects -vH | grep size-pack
