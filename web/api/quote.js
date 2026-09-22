// Live quote for GLD so the monitor shows current value instead of yesterday's close.
//
// READ-ONLY. Takes no input, changes no state. The previous /api/control endpoint (which
// wrote runtime/control.json with no caller authentication) was removed for exactly that
// reason: a public page must not be able to change what the bot does.
//
// This also serves the realtime topic that the page subscribes to. It lives here rather than
// hardcoded in the HTML so the page and the loop read one source of truth and cannot drift.

const UPSTREAM = "https://query1.finance.yahoo.com/v8/finance/chart/";
const TOPIC_URL =
  "https://raw.githubusercontent.com/IAMIbrahimmemon/von-gold/main/runtime/realtime.json";

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "method not allowed" });
  }

  const what = (req.query && req.query.what) || "quote";

  // --- the realtime topic name ---
  if (what === "topic") {
    res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=120");
    try {
      const r = await fetch(TOPIC_URL, { headers: { "User-Agent": "von-gold-monitor/1.0" } });
      if (!r.ok) return res.status(502).json({ error: `upstream ${r.status}` });
      const cfg = await r.json();
      return res.status(200).json({
        topic: cfg.topic || null,
        transport: cfg.transport || null,
      });
    } catch (e) {
      return res.status(502).json({ error: String((e && e.message) || e) });
    }
  }

  // --- the quote (default) ---
  res.setHeader("Cache-Control", "s-maxage=20, stale-while-revalidate=30");
  const raw = (req.query && req.query.symbol) || "GLD";
  const symbol = String(raw).toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 12);
  if (!symbol) return res.status(400).json({ error: "invalid symbol" });

  try {
    const upstream = await fetch(
      `${UPSTREAM}${encodeURIComponent(symbol)}?range=1d&interval=5m`,
      { headers: { "User-Agent": "Mozilla/5.0 (compatible; von-gold-monitor/1.0)" } }
    );
    if (!upstream.ok) return res.status(502).json({ error: `upstream ${upstream.status}` });
    const body = await upstream.json();
    const meta = body && body.chart && body.chart.result && body.chart.result[0]
      && body.chart.result[0].meta;
    if (!meta || typeof meta.regularMarketPrice !== "number") {
      return res.status(502).json({ error: "no price in upstream response" });
    }
    const price = meta.regularMarketPrice;
    const prevClose = meta.chartPreviousClose ?? meta.previousClose ?? null;
    return res.status(200).json({
      symbol,
      price,
      prev_close: prevClose,
      change: prevClose ? price - prevClose : null,
      change_pct: prevClose ? (price / prevClose - 1) * 100 : null,
      market_time: meta.regularMarketTime
        ? new Date(meta.regularMarketTime * 1000).toISOString() : null,
      age_seconds: meta.regularMarketTime
        ? Math.max(0, Math.floor(Date.now() / 1000 - meta.regularMarketTime)) : null,
      source: "yahoo chart api (proxied)",
    });
  } catch (e) {
    return res.status(502).json({ error: String((e && e.message) || e) });
  }
}
