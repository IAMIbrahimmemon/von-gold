"""Local dashboard server: serves the monitor with direct disk access to runtime/.

Why this exists
---------------
The deployed page reads the feed over raw.githubusercontent.com, which sends
`Cache-Control: max-age=300` -- a hard 5-minute CDN cache. That is fine for a public showcase
and useless for watching decisions land every 10 seconds.

Serving the page from the same machine as the loop removes the network from the path
entirely: the HTML is read from web/, the state from runtime/, and a decision written a second
ago is visible a second later. No cloud function, no API key, no CDN.

It also mirrors the two endpoints the Vercel deployment provides (`/api/quote`, `/api/quote?what=topic`)
so the page uses one code path in both modes.

Binding: 127.0.0.1 only. This serves `runtime/` -- which contains the account state and the
realtime topic -- so it must never be reachable from the network.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]   # scripts/local_server.py -> repo root
WEB = REPO / "web"
RUNTIME = REPO / "runtime"
HOST, PORT = "127.0.0.1", 8788

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _quote(symbol: str = "GLD") -> dict:
    """Live quote, same upstream the deployed function uses. No key required."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{symbol}?range=1d&interval=5m")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; von-gold-monitor/1.0)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    meta = body["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    ts = meta.get("regularMarketTime")
    import time as _t
    return {
        "symbol": symbol,
        "price": price,
        "prev_close": prev,
        "change": (price - prev) if (price and prev) else None,
        "change_pct": ((price / prev - 1) * 100) if (price and prev) else None,
        "market_time": (_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(ts)) if ts else None),
        "age_seconds": (max(0, int(_t.time() - ts)) if ts else None),
        "source": "yahoo chart api (local proxy)",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal readable
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Never cache locally: the whole point is to see the newest state.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        # --- API: mirrors the deployed /api/quote ---
        if path == "/api/quote":
            what = (qs.get("what") or ["quote"])[0]
            try:
                if what == "topic":
                    cfg = RUNTIME / "realtime.json"
                    if not cfg.exists():
                        return self._json({"topic": None, "transport": None})
                    j = json.loads(cfg.read_text())
                    return self._json({"topic": j.get("topic"), "transport": j.get("transport")})
                sym = ((qs.get("symbol") or ["GLD"])[0] or "GLD").upper()[:12]
                return self._json(_quote(sym))
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 502)

        # --- runtime/ files straight off disk (no CDN) ---
        if path.startswith("/runtime/"):
            target = (RUNTIME / path[len("/runtime/"):]).resolve()
            if not str(target).startswith(str(RUNTIME.resolve())) or not target.is_file():
                return self._send(404, b'{"error":"not found"}', "application/json")
            return self._send(200, target.read_bytes(),
                              MIME.get(target.suffix, "application/octet-stream"))

        # --- static page + assets ---
        rel = path.lstrip("/") or "index.html"
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            target = WEB / "index.html"      # unknown path -> the page
        return self._send(200, target.read_bytes(),
                          MIME.get(target.suffix, "application/octet-stream"))


def main() -> int:
    if not WEB.exists():
        print(f"no web/ directory at {WEB}")
        return 1
    print(f"von-gold local monitor")
    print(f"  http://{HOST}:{PORT}/")
    print(f"  serving {WEB} + {RUNTIME} (local disk, no CDN)")
    print(f"  bound to {HOST} only -- the account state is not exposed to the network")
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
