/**
 * Control endpoint for the von-gold dry-run monitor.
 *
 * GET  /api/control  -> current switch state (read from the repo, no auth needed)
 * POST /api/control  -> write runtime/control.json to the repo (needs GITHUB_TOKEN)
 *
 * Why the repo is the transport: the trading loop runs on the user's Mac, behind no
 * inbound port. The dashboard runs on Vercel. A versioned file in the repo is the
 * simplest channel that (a) works through NAT, (b) keeps a human-readable audit trail
 * of every switch flip, and (c) fails closed when unavailable.
 *
 * Required env var (Vercel project settings, NOT in this repo):
 *   GITHUB_TOKEN  - fine-grained PAT with "Contents: read and write" on the repo only.
 */

const REPO = process.env.GITHUB_REPO || "IAMIbrahimmemon/von-gold";
const BRANCH = process.env.GITHUB_BRANCH || "main";
const PATH = "runtime/control.json";
const API = "https://api.github.com";

const DEFAULTS = {
  enabled: false,
  reason: "default (serverless read)",
  max_exposure: 1.0,
  kill_if_drawdown_exceeds: 0.25,
  symbol: "GLD",
};

async function readControl() {
  // Unauthenticated raw read: works even before a token is configured.
  const url = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/${PATH}?t=${Date.now()}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) return { ...DEFAULTS };
  try {
    return { ...DEFAULTS, ...(await r.json()) };
  } catch {
    return { ...DEFAULTS, reason: "control.json unparseable" };
  }
}

async function writeControl(token, body) {
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "von-gold-monitor",
    "X-GitHub-Api-Version": "2022-11-28",
  };

  // Read-modify-write so the guard-rail fields survive a switch flip.
  const cur = await fetch(`${API}/repos/${REPO}/contents/${PATH}?ref=${BRANCH}`, { headers });
  let sha;
  let existing = {};
  if (cur.ok) {
    const j = await cur.json();
    sha = j.sha;
    try {
      existing = JSON.parse(Buffer.from(j.content, "base64").toString("utf8"));
    } catch {}
  } else if (cur.status !== 404) {
    const t = await cur.text();
    throw new Error(`read failed ${cur.status}: ${t.slice(0, 200)}`);
  }

  const next = {
    ...DEFAULTS,
    ...existing,
    enabled: body.enabled === true,
    reason: body.reason || (body.enabled ? "enabled from web monitor" : "stopped from web monitor"),
    changed_by: body.changed_by || "web-monitor",
    changed_at: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
  };
  // Keep a human's guard rails if they tightened them in the file directly.
  if (typeof body.max_exposure === "number") next.max_exposure = body.max_exposure;
  if (typeof body.kill_if_drawdown_exceeds === "number") {
    next.kill_if_drawdown_exceeds = body.kill_if_drawdown_exceeds;
  }

  const content = Buffer.from(JSON.stringify(next, null, 2) + "\n").toString("base64");
  const put = await fetch(`${API}/repos/${REPO}/contents/${PATH}`, {
    method: "PUT",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({
      message: `control: switch ${next.enabled ? "ON" : "OFF"} (${next.changed_by})`,
      content,
      branch: BRANCH,
      ...(sha ? { sha } : {}),
    }),
  });
  if (!put.ok) {
    const t = await put.text();
    throw new Error(`write failed ${put.status}: ${t.slice(0, 300)}`);
  }
  return next;
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "GET") {
    const control = await readControl();
    return res.status(200).json(control);
  }

  if (req.method === "POST") {
    const token = process.env.GITHUB_TOKEN;
    if (!token) {
      return res.status(500).json({
        error:
          "GITHUB_TOKEN is not configured on this deployment, so the switch cannot be " +
          "written. Add a fine-grained PAT with Contents:read+write as a Vercel env var, " +
          "or flip the switch locally with: von-gold tick --enable / --disable",
      });
    }
    try {
      const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
      const next = await writeControl(token, body);
      return res.status(200).json(next);
    } catch (e) {
      return res.status(502).json({ error: String(e.message || e) });
    }
  }

  res.setHeader("Allow", "GET, POST");
  return res.status(405).json({ error: "method not allowed" });
}
