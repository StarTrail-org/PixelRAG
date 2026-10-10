# Codex chat backend

The standalone agent server supports `CHAT_BACKEND=claude` (the existing
default) and `CHAT_BACKEND=codex`. Both use the same search and screenshot
handlers and expose the existing `/chat` SSE contract to the Next.js proxy.
The Codex path uses the logged-in Codex CLI via `codex exec --json`, with a
per-conversation stdio MCP adapter for `pixelrag_search` and `pixelrag_tile`.
It supports text history and an uploaded image, search/gallery events,
disconnect cancellation, tool-call limits, and a wall-time timeout.

Prerequisites: Node 22+, the existing web dependencies, Codex CLI 0.159.1
(the tested version), and `codex login` under the service's OS account.
No OpenAI API key is required when using that account's ChatGPT login.

```sh
CHAT_BACKEND=codex CODEX_BIN=/home/zlab-ssd1/.local/bin/codex \
  PIXELRAG_SEARCH_URL=http://localhost:30001 AGENT_PORT=30011 \
  /home/yichuan/.nix-profile/bin/node web/agent-server.mjs
```

`CHAT_CODEX_MODEL` optionally selects a model; otherwise Codex chooses its
default. `CHAT_CODEX_TIMEOUT_MS` defaults to 180000. `CHAT_MAX_TURNS` bounds
MCP tool calls for Codex. Existing request/concurrency limits apply to both
providers. `CHAT_MAX_BUDGET_USD` and Claude thinking-token limits apply only
to Claude; Codex subscription use is bounded by request, tool, and time limits.

The model runs in an empty temporary directory with a read-only sandbox,
shell, web search, browser, external apps, plugins, and image-file reading
disabled. Only the two PixelRAG MCP tools are preapproved. A random local
bearer token isolates each conversation's tool endpoint. Tool arguments
are validated with the same Zod schemas as Claude's tools.

The production drop-in `pixelrag-agent-codex.conf` runs the original
`/home/yichuan/visrag` checkout under `zlab-ssd1`, the account with the
working Codex login. It preserves the existing port, rate limits, CORS,
and search-backend configuration. Adjust its paths and account before
using it on another host. Do not copy authentication tokens between users.

An administrator can apply it with:

```sh
sudo install -m 0644 deploy/pixelrag-agent-codex.conf \
  /etc/systemd/system/pixelrag-agent.service.d/codex.conf
sudo systemctl daemon-reload
sudo systemctl restart pixelrag-agent.service
curl --fail http://localhost:30010/health
curl --fail http://localhost:30010/ready
```

The health response includes `backend: "codex"`. Removing the drop-in and
restarting the service restores the original Claude deployment.

Validation:

```sh
node --test web/lib/*.test.mjs
node --check web/agent-server.mjs
```

Official documentation: [Non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode)
and [MCP tool configuration](https://learn.chatgpt.com/docs/config-file/config-reference).
