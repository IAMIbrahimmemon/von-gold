#!/usr/bin/env bash
# Keep the live decision feed moving.
#
# Runs scripts/live_loop.py at a 10-second decision interval and publishes runtime/feed.json
# to the repo every few minutes so the deployed dashboard stays current.
#
# Why publish on an interval rather than every decision: each publish is a git commit +
# push. At 10s that would be ~8,600 commits/day, which would make the repo history useless.
# The local feed.json is always current; the published copy lags a few minutes by design.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG="$REPO_DIR/runtime/liveloop.log"
mkdir -p "$REPO_DIR/runtime"

# Publish no more often than this many decisions (6 decisions/min at 10s => ~5 min).
PUBLISH_EVERY="${VONGOLD_PUBLISH_EVERY:-30}"

echo "=== live loop start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

# PYTHONPATH is cleared on purpose: an agent/Hermes session exports one that would shadow
# this venv's packages.
exec env -u PYTHONPATH "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/live_loop.py" \
  --interval 10 \
  --publish-every "$PUBLISH_EVERY" \
  >> "$LOG" 2>&1
