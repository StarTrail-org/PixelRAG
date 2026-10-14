// Retrieval failure classification, shared by the two chat backends
// (`web/agent-server.mjs` in production, the inline SDK path in
// `web/app/api/chat/route.ts` for local dev) so the contract cannot drift
// between them.
//
// The distinction that matters is NOT "did the search return hits" but
// "was the index reached at all":
//
//   - reached, no hits   -> a real answer. The model should say so.
//   - never reached      -> there is nothing to ground an answer in.
//
// The second case used to be handed to the model as an ordinary tool result
// ("Search API error: 502"), which reads as information rather than as a
// failure. The model would then answer from its own memory and the turn would
// end as a success, so a dead retrieval backend produced confident, ungrounded
// answers and a clean log line. PixelRAG's entire contract is that answers come
// from retrieved tiles, so an unreachable index has to end the turn as
// degraded, not as done.

export const RETRIEVAL_OK = "ok"
export const RETRIEVAL_BACKEND_DOWN = "backend_down"
export const RETRIEVAL_BAD_REQUEST = "bad_request"

// 5xx and 429 are the backend failing or shedding load; the request itself was
// fine, so retrying with different arguments cannot help. 4xx means this call
// was malformed — the model can usefully correct that one itself.
export function classifyStatus(status) {
  if (status >= 200 && status < 300) return RETRIEVAL_OK
  if (status >= 500 || status === 429) return RETRIEVAL_BACKEND_DOWN
  return RETRIEVAL_BAD_REQUEST
}

// A throw from fetch is a connection refused, DNS failure, or the abort signal
// firing — the index was never reached in any of those cases.
export function classifyThrow() {
  return RETRIEVAL_BACKEND_DOWN
}

export function createRetrievalHealth() {
  return { ok: 0, backendDown: 0, badRequest: 0, lastError: null }
}

export function recordRetrieval(health, outcome, detail = null) {
  if (outcome === RETRIEVAL_OK) health.ok += 1
  else if (outcome === RETRIEVAL_BACKEND_DOWN) {
    health.backendDown += 1
    health.lastError = detail
  } else {
    health.badRequest += 1
    health.lastError = detail
  }
  return health
}

// True when the turn produced no grounded retrieval at all: the backend failed
// and nothing ever came back from the index. A turn that failed once and then
// succeeded is still grounded, so it is not degraded.
export function isUngrounded(health) {
  return health.backendDown > 0 && health.ok === 0
}

// What the model is told when the index is unreachable. It has to be an
// instruction, not a status line: the failure mode being fixed is the model
// treating an error string as context and answering anyway.
export function backendDownInstruction(reason) {
  return [
    `RETRIEVAL BACKEND UNAVAILABLE (${reason}).`,
    "The visual Wikipedia index cannot be reached, so there are no tiles to read.",
    "Do NOT answer from your own knowledge, and do NOT retry the search.",
    "Tell the user that PixelRAG's search backend is temporarily unavailable, then stop.",
  ].join(" ")
}

export const UNGROUNDED_CLIENT_MESSAGE =
  "PixelRAG's retrieval backend is unavailable, so no Wikipedia tiles could be read for this answer."
