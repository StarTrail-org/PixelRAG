import http from "node:http"
import { spawn } from "node:child_process"
import { randomBytes } from "node:crypto"
import { mkdtemp, writeFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import readline from "node:readline"
import { z } from "zod"

export function codexTool(name, description, shape, handler) {
  const schema = z.object(shape)
  return { name, description, inputSchema: z.toJSONSchema(schema), annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false }, call: (args) => handler(schema.parse(args)) }
}

export async function runCodex({ textPrompt, uploadedImage, systemPrompt, tools, send, signal, maxToolCalls = 12, onDiagnostic = () => {} }) {
  const dir = await mkdtemp(join(tmpdir(), "pixelrag-codex-"))
  const token = randomBytes(32).toString("hex")
  let calls = 0
  let child
  let timedOut = false
  let exited = false
  const bridge = http.createServer(async (req, res) => {
    const respond = (status, data) => { res.writeHead(status, { "Content-Type": "application/json" }); res.end(JSON.stringify(data)) }
    if (req.method !== "POST" || req.headers.authorization !== `Bearer ${token}`) return respond(403, {})
    try {
      let body = ""
      for await (const chunk of req) {
        body += chunk
        if (body.length > 65536) return respond(413, {})
      }
      const request = JSON.parse(body)
      if (request.method === "tools/list") return respond(200, { tools: tools.map(({ name, description, inputSchema, annotations }) => ({ name, description, inputSchema, annotations })) })
      if (request.method !== "tools/call") return respond(400, {})
      if (++calls > maxToolCalls) return respond(200, { isError: true, content: [{ type: "text", text: "Tool limit reached. Finish the answer using the tiles already read." }] })
      const tool = tools.find((t) => t.name === request.params?.name)
      if (!tool) throw new Error("Unknown tool")
      respond(200, await tool.call(request.params.arguments || {}))
    } catch (error) {
      respond(200, { isError: true, content: [{ type: "text", text: String(error) }] })
    }
  })
  let timer
  let forceKill
  const cancel = () => {
    if (child && !exited) {
      child.kill("SIGTERM")
      if (!forceKill) forceKill = setTimeout(() => { if (!exited) child.kill("SIGKILL") }, 2000)
    }
  }
  try {
    await new Promise((resolve, reject) => { bridge.once("error", reject); bridge.listen(0, "127.0.0.1", resolve) })
    const endpoint = `http://127.0.0.1:${bridge.address().port}`
    const args = ["exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--json", "--sandbox", "read-only", "--cd", dir,
      "--disable", "shell_tool", "--disable", "code_mode", "--enable", "code_mode_host", "--disable", "multi_agent", "--disable", "apps", "--disable", "plugins", "--disable", "view_image", "--disable", "browser_use_external", "--disable", "browser_use_full_cdp_access", "--disable", "browser_use", "--disable", "computer_use", "--disable", "image_generation",
      "-c", "approval_policy=\"never\"", "-c", "web_search=\"disabled\"",
      "-c", `developer_instructions=${JSON.stringify(systemPrompt)}`,
      "-c", `mcp_servers.pixelrag.command=${JSON.stringify(process.execPath)}`,
      "-c", `mcp_servers.pixelrag.args=${JSON.stringify([fileURLToPath(new URL("./codex-mcp.mjs", import.meta.url)), endpoint, token])}`,
      "-c", "mcp_servers.pixelrag.required=true",
      "-c", 'mcp_servers.pixelrag.tools.pixelrag_search.approval_mode="approve"',
      "-c", 'mcp_servers.pixelrag.tools.pixelrag_tile.approval_mode="approve"']
    if (process.env.CHAT_CODEX_MODEL) args.push("--model", process.env.CHAT_CODEX_MODEL)
    if (uploadedImage) {
      const match = uploadedImage.match(/^data:(image\/(?:png|jpeg|webp|gif));base64,(.+)$/is)
      if (!match) throw new Error("Unsupported uploaded image; use a PNG, JPEG, WebP, or GIF data URL")
      const path = join(dir, "uploaded-image")
      await writeFile(path, Buffer.from(match[2], "base64"), { mode: 0o600 })
      args.push("--image", path)
    }
    args.push("-")
    if (signal?.aborted) throw new Error("Chat cancelled")
    child = spawn(process.env.CODEX_BIN || "codex", args, { cwd: dir, stdio: ["pipe", "pipe", "pipe"] })
    let diagnostics = ""
    child.stderr.on("data", (chunk) => {
      const safe = String(chunk).replaceAll(token, "[redacted]")
      diagnostics = (diagnostics + safe).slice(-4000)
      onDiagnostic(safe)
    })
    const completed = new Promise((resolve, reject) => {
      child.once("error", reject)
      child.once("close", (code) => {
        exited = true
        clearTimeout(forceKill)
        code === 0 ? resolve() : reject(new Error(signal?.aborted ? "Chat cancelled" : timedOut ? "Codex request timed out" : `Codex failed (${code}): ${diagnostics}`))
      })
    })
    // Attach a rejection handler immediately, before consuming stdout.
    completed.catch(() => {})
    signal?.addEventListener("abort", cancel, { once: true })
    timer = setTimeout(() => { timedOut = true; cancel() }, Number(process.env.CHAT_CODEX_TIMEOUT_MS || 180000))
    child.stdin.on("error", () => {})
    child.stdin.end(textPrompt)
    let sentText = false
    let failure
    for await (const line of readline.createInterface({ input: child.stdout })) {
      let event
      try { event = JSON.parse(line) } catch { continue }
      if (event.type === "item.completed" && event.item?.type === "agent_message" && event.item.text) {
        send("text", { text: event.item.text }); sentText = true
      } else if (event.type === "item.completed" && event.item?.type === "reasoning" && event.item.text) {
        send("thinking", { text: event.item.text })
      } else if (event.type === "turn.failed" || event.type === "error") {
        failure = event.error?.message || event.message || "Codex turn failed"
      }
    }
    await completed
    if (failure) throw new Error(failure)
    if (!sentText) throw new Error("Codex returned no answer")
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener("abort", cancel)
    cancel()
    bridge.closeAllConnections()
    await new Promise((resolve) => bridge.close(resolve))
    await rm(dir, { recursive: true, force: true })
  }
}
