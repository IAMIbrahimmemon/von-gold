// Live quote for GLD so the monitor shows current value instead of yesterday's close.
//
// READ-ONLY. This endpoint takes a symbol, returns a price, and does nothing else -- there
// is no path for a caller to alter state. The previous /api/control endpoint (which wrote
// runtime/control.json with no caller authentication) was removed for exactly that reason:
// a public page must not be able to change what the bot does.
//
// The quote is proxied through the server rather than fetched by the browser because Yahoo
// does not send CORS headers, and going through here also lets us cache for a few seconds
// so a page left open does not hammer the upstream.

const UPSTREAM = "https://query1.finance.yahoo.com/v8/finance/chart/";
const CACHE_SECONDS = 20;

export default async function handler(req, res) {
  res.setHeader("Cache-Control", `s-maxage=${CACHE_SECONDS}, stale-while-revalidate=30`);
  res.setHeader("Access-Control-Allow-Origin", "*");

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "method not allowed" });
  }

  // Constrain the symbol to a charset rather than interpolating arbitrary user input into
  // the upstream URL.
  const raw = (req.query && req.query.symbol) || "GLD";
  const symbol = String(raw).toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 12);
  if (!symbol) {
    return res.status(400).json({ error: "invalid symbol" });
  }

  try {
    const upstream = await fetch(
      `${UPSTREAM}${encodeURIComponent(symbol)}?range=1d&interval=5m`,
      { headers: { "User-Agent": "Mozilla/5.0 (compatible; von-gold-monitor/1.0)" } }
    );
    if (!upstream.ok) {
      return res.status(502).json({ error: `upstream ${upstream.status}` });
    }

    const body = await upstream.json();
    const result = body && body.chart && body.chart.result && body.chart.result[0];
    const meta = result && result.meta;
    if (!meta || typeof meta.regularMarketPrice !== "number") {
      return res.status(502).json({ error: "no price in upstream response" });
    }

    const price = meta.regularMarketPrice;
    const prevClose = meta.chartPreviousClose ?? meta.previousClose ?? null;
    const marketTime = meta.regularMarketTime
      ? new Date(meta.regularMarketTime * 1000).toISOString()
      : null;

    return res.status(200).json({
      symbol,
      price,
      prev_close: prevClose,
      change: prevClose ? price - prevClose : null,
      change_pct: prevClose ? (price / prevClose - 1) * 100 : null,
      market_time: marketTime,
      // The true freshness signal: how old the upstream quote is. The browser uses this to
      // label the price honestly instead of implying a live tick it may not have.
      age_seconds: meta.regularMarketTime
        ? Math.max(0, Math.floor(Date.now() / 1000 - meta.regularMarketTime))
        : null,
      source: "yahoo chart api (proxied)",
    });
  } catch (e) {
    return res.status(502).json({ error: String((e && e.message) || e) });
  }
}
