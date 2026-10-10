// A stdio MCP adapter for a single conversation's local tool endpoint.
// No filesystem, shell, or arbitrary URL tools are exposed to the model.
import readline from "node:readline"

const [endpoint, token] = process.argv.slice(2)
const reply = (id, result) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, result }) + "\n")
for await (const line of readline.createInterface({ input: process.stdin })) {
  let request
  try {
    request = JSON.parse(line)
    if (request.id === undefined) continue
    if (request.method === "initialize") {
      reply(request.id, { protocolVersion: request.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: "pixelrag", version: "1.0.0" } })
    } else if (request.method === "ping") {
      reply(request.id, {})
    } else if (["tools/list", "tools/call"].includes(request.method)) {
      const response = await fetch(endpoint, {
        method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify(request), signal: AbortSignal.timeout(65000),
      })
      if (!response.ok) throw new Error(`Tool endpoint HTTP ${response.status}`)
      reply(request.id, await response.json())
    } else {
      process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: request.id, error: { code: -32601, message: "Method not found" } }) + "\n")
    }
  } catch (error) {
    if (request?.id !== undefined) process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: request.id, error: { code: -32603, message: String(error) } }) + "\n")
  }
}
