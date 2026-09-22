#!/usr/bin/env bash
# Open the local dashboard. The server itself runs as a launchd service
# (local.vongold.monitor); this just waits for it and opens a browser.
set -uo pipefail
URL="http://127.0.0.1:8788/"
for _ in $(seq 1 20); do
  if curl -sf -m 2 -o /dev/null "$URL"; then break; fi
  sleep 0.5
done
if ! curl -sf -m 2 -o /dev/null "$URL"; then
  echo "monitor not responding on 8788; check: launchctl print gui/$(id -u)/local.vongold.monitor" >&2
  exit 1
fi
echo "von-gold monitor: $URL"
open "$URL" 2>/dev/null || true
