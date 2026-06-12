#!/usr/bin/env bash
# Blue-green switch for the PixelRAG search API.
#
# Points BOTH api.pixelrag.ai (via nginx) and the agent backend (direct on
# localhost) at the chosen slot, but only after the target passes a health
# check and a smoke query. Rollback is just switching back to the other slot.
#
#   blue  = base model   (pixelrag-api@blue.service)
#   green = LoRA model   (pixelrag-api@green.service)
#
# Usage: deploy/api-switch.sh <blue|green>
set -euo pipefail
SLOT="${1:?usage: api-switch.sh <blue|green>}"
ENV_FILE="/etc/pixelrag/api-${SLOT}.env"
UPSTREAM=/etc/nginx/conf.d/pixelrag-api-upstream.conf
AGENT_DROPIN=/etc/systemd/system/pixelrag-agent.service.d/backend.conf

# Resolve port from the slot's env file.
[ -f "$ENV_FILE" ] || { echo "ABORT: $ENV_FILE not found"; exit 1; }
PORT=$(grep -oP '^PORT=\K\d+' "$ENV_FILE")
[ -n "${PORT:-}" ] || { echo "ABORT: PORT not found in $ENV_FILE"; exit 1; }

base="http://127.0.0.1:${PORT}"

# 1. Target must be healthy.
echo "checking ${base}/health ..."
curl -fsS "${base}/health" >/dev/null || { echo "ABORT: :${PORT} is not healthy"; exit 1; }

# 2. Smoke query — the target must return sane results before taking traffic.
echo "smoke query ..."
hits=$(curl -fsS -X POST "${base}/search" -H 'Content-Type: application/json' \
  -d '{"queries":[{"text":"Albert Einstein"}],"n_docs":3}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['results'][0]['hits']))")
[ "${hits:-0}" -ge 1 ] || { echo "ABORT: smoke query returned no hits"; exit 1; }
echo "smoke ok (${hits} hits)"

# 3. Flip nginx (api.pixelrag.ai) — graceful reload, zero dropped connections.
echo "upstream pixelrag_api { server 127.0.0.1:${PORT}; }" | sudo tee "$UPSTREAM" >/dev/null
sudo nginx -t && sudo nginx -s reload

# 4. Repoint the agent (it calls the API directly on localhost) + restart.
sudo mkdir -p "$(dirname "$AGENT_DROPIN")"
printf '[Service]\nEnvironment=PIXELRAG_SEARCH_URL=http://localhost:%s\n' "$PORT" | sudo tee "$AGENT_DROPIN" >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart pixelrag-agent.service

echo "SWITCHED: api.pixelrag.ai + agent -> 127.0.0.1:${PORT} (slot=${SLOT})"
