# Deployment

Live dashboard (read-only): **https://von-gold-monitor.vercel.app**

| | |
|---|---|
| Repo | https://github.com/IAMIbrahimmemon/von-gold (public) |
| Vercel project | `von-gold-monitor` (`prj_2KIY6iIhP4L9m3rFEd7LijG9s6Ab`) |
| Production URL | https://von-gold-monitor.vercel.app |
| Root directory | `web/` |
| Deployed files | `index.html`, `vercel.json`, `package.json`, `api/control.js` |
| Status source | `raw.githubusercontent.com/IAMIbrahimmemon/von-gold/main/runtime/status.json` |

## How it was deployed

Deployed by **file upload via the Composio Vercel API**, not by git integration, because
the Vercel account has no GitHub account connected (`no_github_account_connected`). Git
source deployments require linking GitHub to Vercel first.

A file-upload deployment is a **snapshot**: it does not redeploy when the repo changes.
The dashboard reads live state from `raw.githubusercontent.com` on every page load, so the
data is always current even though the page itself is static. To ship a UI change, redeploy.

## Gotcha worth remembering

Vercel's create-deployment API ignored the `encoding: "base64"` field on the file objects
and wrote the base64 text through literally. The only file that failed loudly was
`vercel.json` (`invalid_vercel_json`) because it is the only one Vercel parses as JSON --
the HTML and JS would have shipped silently corrupt. **Send `files[].data` as plain UTF-8
text, not base64.**

## The on/off switch does not work remotely yet

`POST /api/control` writes `runtime/control.json` back to the repo, which needs a
`GITHUB_TOKEN` environment variable in the Vercel project (fine-grained PAT, *Contents:
read and write* on this repo only). Without it the deployed page is read-only and the
switch must be flipped locally:

```bash
cd ~/Documents/GitHub/von-gold
env -u PYTHONPATH .venv/bin/python -m vongold.cli disable --force
```

Do not add that token unless you're comfortable with the deployed page being able to
write to the repo.
