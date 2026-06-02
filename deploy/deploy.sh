#!/usr/bin/env bash
# PixelRAG self-hosted CD.
#
# Runs on the deploy box via the GitHub Actions self-hosted runner after a push
# to main (see .github/workflows/deploy.yml). The working tree has already been
# fast-forwarded to origin/main by the workflow; this script decides what to
# (re)start based on what actually changed, so a docs-only push costs nothing
# and the expensive 216G search index is never reloaded automatically.
#
# Usage: deploy/deploy.sh [BEFORE_SHA]
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$HOME/.nix-profile/bin:$PATH"

BEFORE="${1:-}"
mkdir -p logs
LOG=logs/cd.log
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "$(ts) $*" | tee -a "$LOG"; }

AFTER=$(git rev-parse HEAD)
say "deploy: ${BEFORE:-<none>} -> ${AFTER}"

if [ -n "$BEFORE" ] && git cat-file -e "${BEFORE}^{commit}" 2>/dev/null; then
  CHANGED=$(git diff --name-only "$BEFORE" "$AFTER")
else
  # First run / unknown base: treat the tip commit's files as the change set.
  CHANGED=$(git show --name-only --pretty="" "$AFTER")
fi
say "changed: $(echo "$CHANGED" | tr '\n' ' ')"

changed() { echo "$CHANGED" | grep -qE "$1"; }

# 1. Python deps. MUST include the extras the box runs (the search API needs the
# `serve` extra: torch, transformers, faiss-cpu). A bare `uv sync` installs only
# the light core and would UNINSTALL torch/faiss, breaking the serves on their
# next restart. `all` = embed+serve+index (no gpu — there's no CUDA here).
if changed '^uv\.lock$'; then
  say "uv.lock changed -> uv sync --extra all"
  uv sync --extra all >>"$LOG" 2>&1 || say "WARN: uv sync failed"
fi

# 2. Agent backend — cheap restart, safe to automate.
if changed '^web/agent-server\.mjs$'; then
  say "agent-server.mjs changed -> restart pixelrag-agent"
  sudo systemctl restart pixelrag-agent.service
  sleep 2
  say "pixelrag-agent: $(systemctl is-active pixelrag-agent.service)"
fi

# 3. Search API — restarting reloads a 216G index (minutes of downtime), so we
#    NEVER auto-restart it. Just flag that a manual restart is needed.
if changed '^serve/'; then
  say "NOTICE: serve/ changed — 'sudo systemctl restart pixelrag-api' needed manually (216G reload)"
fi

# The web frontend deploys via Vercel; nothing to do here for web/ changes
# other than the agent backend handled above.
say "deploy done"
