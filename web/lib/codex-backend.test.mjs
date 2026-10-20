import test from "node:test"
import assert from "node:assert/strict"
import { mkdtemp, writeFile, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { z } from "zod"
import { runCodex, codexTool } from "./codex-backend.mjs"

// A fake CLI uses the actual stdio adapter, making this test cover the MCP
// protocol, tool validation, and SSE event conversion without model calls.
const fakeCli = `
import {spawn} from 'node:child_process';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const args=process.argv.slice(2);
assert(args.includes('read-only'));
assert(args.includes('shell_tool'));
if(args.includes('--image')) assert((await readFile(args[args.indexOf('--image')+1])).length>0);
const config=args.find(x=>x.startsWith('mcp_servers.pixelrag.args='));
const bridgeArgs=JSON.parse(config.slice(config.indexOf('=')+1));
const child=spawn(process.execPath,bridgeArgs,{stdio:['pipe','pipe','inherit']});
const lines=readline.createInterface({input:child.stdout});
const iterator=lines[Symbol.asyncIterator]();
let id=0;
async function call(method,params={}) {
 child.stdin.write(JSON.stringify({jsonrpc:'2.0',id:++id,method,params})+'\\n');
 return JSON.parse((await iterator.next()).value).result;
}
try {
 const init=await call('initialize',{protocolVersion:'2024-11-05'});
 assert.equal(init.serverInfo.name,'pixelrag');
 child.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\\n');
 const list=await call('tools/list');
 assert.equal(list.tools[0].name,'pixelrag_search');
 const result=await call('tools/call',{name:'pixelrag_search',arguments:{query:'Taj Mahal'}});
 console.log(JSON.stringify({type:'item.completed',item:{type:'reasoning',text:'Reading tiles'}}));
 console.log(JSON.stringify({type:'item.completed',item:{type:'agent_message',text:JSON.stringify(result)}}));
 if(process.env.PIXELRAG_TEST_FAILURE) console.log(JSON.stringify({type:'turn.failed',error:{message:'test failure'}}));
} finally {child.stdin.end();lines.close();}
`

test("Codex CLI adapter calls MCP tools and emits SSE-compatible events", async () => {
  const dir = await mkdtemp(join(tmpdir(), "codex-test-"))
  const cli = join(dir, "cli.mjs")
  const previous = process.env.CODEX_BIN
  try {
    await writeFile(cli, `#!${process.execPath}\n${fakeCli}`, { mode: 0o700 })
    process.env.CODEX_BIN = cli
    const events = []
    const options = {
      textPrompt: "What is the Taj Mahal?", systemPrompt: "Read tiles.",
      tools: [codexTool("pixelrag_search", "Search", { query: z.string() }, async (args) => {
        events.push(["searching", args])
        return { content: [{ type: "text", text: "Retrieved " + args.query }] }
      })],
      send: (event, data) => events.push([event, data]),
    }
    await runCodex(options)
    assert.equal(events[0][0], "searching")
    assert.equal(events[1][0], "thinking")
    assert.match(events[2][1].text, /Retrieved Taj Mahal/)
    process.env.PIXELRAG_TEST_FAILURE = "1"
    await assert.rejects(runCodex(options), /test failure/)
    delete process.env.PIXELRAG_TEST_FAILURE
    const controller = new AbortController()
    controller.abort()
    await assert.rejects(runCodex({ ...options, signal: controller.signal }), /cancelled/)
    await assert.rejects(runCodex({ ...options, uploadedImage: "invalid image" }), /Unsupported uploaded image/)
    await runCodex({ ...options, uploadedImage: "data:image/png;base64,iVBORw0KGgo=" })
    events.length = 0
    await runCodex({ ...options, maxToolCalls: 0 })
    assert.match(events.at(-1)[1].text, /Tool limit reached/)
    assert.equal(events.filter(([event]) => event === "searching").length, 0)
    // Exercise cancellation of an already-running subprocess and the timeout.
    await writeFile(cli, `#!${process.execPath}\nsetInterval(() => {}, 1000)`, { mode: 0o700 })
    const running = new AbortController()
    const cancelTimer = setTimeout(() => running.abort(), 100)
    await assert.rejects(runCodex({ ...options, signal: running.signal }), /cancelled/)
    clearTimeout(cancelTimer)
    process.env.CHAT_CODEX_TIMEOUT_MS = "100"
    await assert.rejects(runCodex(options), /timed out/)
    delete process.env.CHAT_CODEX_TIMEOUT_MS
    process.env.CODEX_BIN = join(dir, "nonexistent-cli")
    await assert.rejects(runCodex(options), /ENOENT/)
  } finally {
    if (previous === undefined) delete process.env.CODEX_BIN
    else process.env.CODEX_BIN = previous
    delete process.env.PIXELRAG_TEST_FAILURE
    delete process.env.CHAT_CODEX_TIMEOUT_MS
    await rm(dir, { recursive: true, force: true })
  }
})

test("MCP schemas reject invalid tile coordinates before invoking retrieval", async () => {
  let called = false
  const tool = codexTool("pixelrag_tile", "Read tile", { article_id: z.number().int() }, async () => { called = true })
  assert.equal(tool.inputSchema.type, "object")
  assert.throws(() => tool.call({ article_id: "wrong" }))
  assert.equal(called, false)
})
