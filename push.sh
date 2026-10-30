#!/usr/bin/env bash
# Publish this folder as a PRIVATE repository under github.com/Xubow-creat.
# Usage:  cd <this folder> && bash push.sh [repo-name]      (default repo name: ActivePAUT)
# Needs either GitHub CLI (`brew install gh && gh auth login`) or an empty repo you created by hand on github.com.
set -euo pipefail
cd "$(dirname "$0")"
OWNER="Xubow-creat"
REPO="${1:-ActivePAUT}"

# 0) git identity must exist before anything is created
if [ -z "$(git config --global user.name || true)" ] || [ -z "$(git config --global user.email || true)" ]; then
  echo "Set your git identity first (use your own name and the email of your GitHub account), then run this script again:"
  echo "  git config --global user.name  'Your Name'"
  echo "  git config --global user.email 'you@example.com'"
  exit 1
fi

# an existing .git is removed only if it holds no commits (leftover of an earlier failed run)
if [ -d .git ]; then
  if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    echo "A .git folder with commits already exists here; refusing to overwrite it."; exit 1
  fi
  rm -rf .git
fi

# 1) refuse to continue if anything big or sensitive slipped in
big=$(find . -type f -size +50M -not -path './.git/*' | head -3)
[ -z "$big" ] || { echo "Files over 50 MB found (GitHub limit 100 MB):"; echo "$big"; exit 1; }
if find . -iname '*SEALED*' -o -name '.venv*' -o -name '*.npz' | grep -q .; then echo "Sealed/venv/npz files present; remove them first."; exit 1; fi

# 2) commit with the identity already configured on this Mac
git init -q -b main
git add -A
git commit -q -m "Initial commit: ActivePAUT simulation code, arim snapshot, SimNDT-modern"
echo "Committed $(git ls-files | wc -l) files."

# 3) create the remote and push
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPO" --private --source=. --remote=origin --push \
     --description "Active (embodied) phased-array ultrasonic testing: Bayesian belief + EIG probe positioning"
else
  echo "GitHub CLI not available or not logged in."
  echo "1) Open https://github.com/new , owner $OWNER, name $REPO, Private, do NOT add README/.gitignore/licence."
  read -r -p "2) Press Enter once the empty repo exists... " _
  git remote add origin "https://github.com/$OWNER/$REPO.git"
  git push -u origin main      # asks for your GitHub username and a personal access token (not your password)
fi
echo "Done: https://github.com/$OWNER/$REPO"
