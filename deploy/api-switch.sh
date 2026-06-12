#!/usr/bin/env bash
# Blue-green switch for the PixelRAG search API.
#
# Points BOTH api.pixelrag.ai (via nginx) and the agent backend (direct on
# localhost) at the chosen slot, but only after the target passes a health
# check and a smoke query. Rollback is just switching back to the other port.
#
#   blue  = 30001  (base model,  pixelrag-api.service)
#   green = 30002  (LoRA model,  pixelrag-api-green.service)
#
# Usage: deploy/api-switch.sh <port>
set -euo pipefail
PORT="${1:?usage: api-switch.sh <port>  (30001=blue/base, 30002=green/lora)}"
UPSTREAM=/etc/nginx/conf.d/pixelrag-api-upstream.conf
AGENT_DROPIN=/etc/systemd/system/pixelrag-agent.service.d/backend.conf

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

echo "SWITCHED: api.pixelrag.ai + agent -> 127.0.0.1:${PORT}"
