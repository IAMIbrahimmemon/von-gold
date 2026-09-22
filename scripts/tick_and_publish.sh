#!/usr/bin/env bash
# One dry-run trading session, end to end:
#   1. pull any switch change made from the web monitor
#   2. run the tick (fetches fresh prices, asks local von, simulates the paper fill)
#   3. publish status back to the repo so the monitor shows it
#
# Designed to be run by launchd / cron on a weekday-evening schedule after the US close.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG="$REPO_DIR/runtime/tick.log"
mkdir -p "$REPO_DIR/runtime"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "=== tick $(stamp) ==="

  # 1. Pick up a control change made from the dashboard. Rebase-safe and non-fatal:
  #    if this fails the tick still runs with the LOCAL control file.
  git pull --rebase --autostash -q origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1 || \
    echo "warn: could not pull control state; using local runtime/control.json"

  # 2. Run the tick. PYTHONPATH is cleared deliberately (a Hermes/agent session exports
  #    one that shadows this venv's packages).
  env -u PYTHONPATH "$REPO_DIR/.venv/bin/python" -c \
    "import sys; sys.path.insert(0,'src'); from vongold.dryrun import main; sys.exit(main())" \
    || echo "error: tick failed with exit $?"

  # 3. Publish.
  bash "$REPO_DIR/scripts/publish_status.sh" "runtime: session $(date -u +%Y-%m-%d)" \
    || echo "warn: publish failed (status is still correct locally)"

  echo "=== done $(stamp) ==="
} >> "$LOG" 2>&1
