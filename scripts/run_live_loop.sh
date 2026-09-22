#!/usr/bin/env bash
# Keep the live paper-trading loop running and publish its feed.
#
# Runs scripts/live_loop.py at a 10-second decision interval. Two transports are used, each
# for what it is good at:
#
#   * ntfy (seconds)      -- the page subscribes with EventSource, so decisions appear
#                            immediately. Display only: a forged message cannot move the
#                            account because money fields come from the next source.
#   * git publish (minutes) -- runtime/feed.json and ledger.jsonl, the authoritative account
#                            state. Throttled because every publish is a commit + push; at 10s
#                            unthrottled that is ~8,600 commits/day and the history becomes
#                            useless.
#
# --live-signals is ON: the target is re-derived from the live price so a move across the
# 200-day trend gate can open or close a position within a cycle rather than overnight.
# --allow-entry is ON so those entrances actually execute; without it the loop is reduce-only
# and would sit flat through a rally. Both are recorded on the dashboard so the mode in force
# is never ambiguous.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG="$REPO_DIR/runtime/liveloop.log"
mkdir -p "$REPO_DIR/runtime"

# Publish every N decisions. At a 10s interval 30 decisions is roughly 6-10 minutes, which
# keeps the authoritative published feed reasonably current without flooding git history.
PUBLISH_EVERY="${VONGOLD_PUBLISH_EVERY:-30}"

# Realtime topic: read from runtime/realtime.json so the loop and the page share one source of
# truth. An empty topic disables realtime publishing (the loop still runs and publishes).
TOPIC=""
if [ -f "$REPO_DIR/runtime/realtime.json" ]; then
  TOPIC="$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("topic",""))' \
    "$REPO_DIR/runtime/realtime.json" 2>/dev/null || true)"
fi

echo "=== live loop start $(date -u +%Y-%m-%dT%H:%M:%SZ) topic=${TOPIC:-none} ===" >> "$LOG"

# PYTHONPATH is cleared on purpose: an agent/Hermes session exports one that would shadow
# this venv's packages.
exec env -u PYTHONPATH "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/live_loop.py" \
  --interval 10 \
  --live-signals \
  --allow-entry \
  --topic "$TOPIC" \
  --publish-every "$PUBLISH_EVERY" \
  >> "$LOG" 2>&1
