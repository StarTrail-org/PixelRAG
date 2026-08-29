#!/usr/bin/env bash
# Blue-green switch for the PixelRAG search API, host side.
#
# Points this host's nginx upstream and the agent backend (direct on localhost)
# at the chosen slot, after the target passes a health check and a smoke query.
# Rollback is just switching back to the other port.
#
# SCOPE — read this before using it to resolve an outage. If public traffic
# reaches this host through something in front of it (a relay, a reverse proxy,
# a tunnel endpoint), then that layer selects the slot for public requests and
# this script does not touch it. Running this alone then moves the host and the
# agent while the public route stays where it was, which looks like a completed
# switch and serves the old slot. Set PIXELRAG_INGRESS_SWITCH_HOOK to a script
# that moves that layer too; it is called with the chosen port and must exit 0.
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

# 3. Flip this host's nginx — graceful reload, zero dropped connections.
echo "upstream pixelrag_api { server 127.0.0.1:${PORT}; }" | sudo tee "$UPSTREAM" >/dev/null
sudo nginx -t && sudo nginx -s reload

# 4. Repoint the agent (it calls the API directly on localhost) + restart.
sudo mkdir -p "$(dirname "$AGENT_DROPIN")"
printf '[Service]\nEnvironment=PIXELRAG_SEARCH_URL=http://localhost:%s\n' "$PORT" | sudo tee "$AGENT_DROPIN" >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart pixelrag-agent.service

# 5. Whatever fronts this host has its own idea of which slot is live, and it
# is the one public traffic obeys. Moving it is deployment-specific, so it is
# delegated rather than assumed absent.
if [ -n "${PIXELRAG_INGRESS_SWITCH_HOOK:-}" ]; then
  echo "running ingress hook: ${PIXELRAG_INGRESS_SWITCH_HOOK} ${PORT}"
  "${PIXELRAG_INGRESS_SWITCH_HOOK}" "${PORT}" || {
    echo "ABORT: ingress hook failed — the host moved but public traffic may still be on the old slot" >&2
    exit 1
  }
  echo "SWITCHED: host nginx + agent + public ingress -> 127.0.0.1:${PORT}"
else
  echo "SWITCHED: host nginx + agent -> 127.0.0.1:${PORT}"
  echo "NOTE: public ingress not touched. If anything fronts this host, move it too,"
  echo "      then verify the public endpoint rather than the local one."
fi
