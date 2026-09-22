#!/usr/bin/env bash
# Publish runtime/status.json (and the control file) to GitHub so the deployed
# monitor can read them.
#
# Deliberately narrow: this commits ONLY the runtime/ files, never source. That keeps
# a live-trading bot's automation from accidentally pushing code changes.
#
# Usage: scripts/publish_status.sh ["commit message"]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MSG="${1:-runtime: status update $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a git repository: $REPO_DIR" >&2
  exit 1
fi

if [ ! -f runtime/status.json ]; then
  echo "runtime/status.json missing -- run a tick first (von-gold tick)" >&2
  exit 1
fi

git add runtime/status.json
[ -f runtime/control.json ] && git add runtime/control.json
# The live decision feed (rewritten every decision by scripts/live_loop.py). Published so the
# dashboard can show what the model is saying right now, not just the last session's record.
[ -f runtime/feed.json ] && git add runtime/feed.json

if git diff --cached --quiet; then
  echo "nothing to publish (runtime unchanged)"
  exit 0
fi

git -c user.name="von-gold bot" -c user.email="bot@localhost" commit -q -m "$MSG" \
  -- runtime/status.json runtime/control.json runtime/feed.json

# Pull any control change made from the web monitor before pushing, so a user flipping
# the switch while we were mid-tick is not overwritten by a non-fast-forward rejection.
git pull --rebase --autostash -q origin "$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null || true
git push -q origin HEAD
echo "published: $MSG"
