# Deploy

How PixelRAG runs in production. (Host-specific details — the actual machine, secrets, live
state — are intentionally **not** in this public repo; they live in an out-of-repo file the
root `CLAUDE.md` imports on the deploy host.)

## Topology
- **Frontend** (`web/`) — deployed on Vercel from `main`, automatically.
- **Search API** (`serve/`) — `pixelrag serve` behind nginx, run **blue-green** (two slots).
- **Chat agent** (`web/agent-server.mjs`) — a Node service that calls the search API.

## CD — self-hosted runner (egress-only)
Continuous deployment uses a **self-hosted GitHub Actions runner** on the deploy box. The runner
dials *out* to GitHub (no inbound access, no SSH keys or hostnames in the repo/secrets), so the
machine stays private and works behind a firewall.

`.github/workflows/deploy.yml` triggers **only** on push to `main` (post-merge = trusted code) and
manual dispatch — never `pull_request` — so fork PRs can't run on the runner. All CI stays on
GitHub-hosted runners.

On each deploy, `deploy/deploy.sh` fast-forwards the checkout and restarts **only what changed**:
- `uv.lock` → `uv sync`
- `web/agent-server.mjs` → restart the agent (cheap)
- `serve/**` → flagged only (a search-index reload is expensive — use blue-green instead)

It refuses to run unless the checkout is on a clean `main` (the deploy box doubles as a dev
machine, so it must never clobber in-progress work).

## Blue-green search API
Two slots — **blue** and **green** — each an independent `pixelrag serve` on its own port, fronted
by an nginx `upstream` (`pixelrag-api-upstream.conf` → `/etc/nginx/conf.d/`). To roll out a new
index or model with zero downtime:

1. Bring up the idle slot with the new config (`pixelrag-api-green.service` is the green slot).
2. `deploy/api-switch.sh <port>` — health- and smoke-checks the target, flips the nginx upstream
   with a graceful reload (no dropped connections), and repoints + restarts the agent.
3. **Rollback** = `deploy/api-switch.sh <other-port>`.

This is preferred over restarting a slot in place, which reloads the (large) FAISS index and would
mean minutes of downtime.

## Files
- `deploy.sh` — CD restart logic (invoked by the Deploy workflow)
- `api-switch.sh` — blue-green cutover + rollback
- `pixelrag-api.service`, `pixelrag-api-green.service` — the two search-API slots
- `pixelrag-agent.service` — the chat agent
- `pixelrag-api-upstream.conf` — nginx upstream the switch script rewrites
