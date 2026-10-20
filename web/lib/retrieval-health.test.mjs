import test from "node:test"
import assert from "node:assert/strict"

import {
  RETRIEVAL_OK,
  RETRIEVAL_BACKEND_DOWN,
  RETRIEVAL_BAD_REQUEST,
  classifyStatus,
  classifyThrow,
  createRetrievalHealth,
  recordRetrieval,
  isUngrounded,
  backendDownInstruction,
} from "./retrieval-health.mjs"

test("2xx is a reached index", () => {
  assert.equal(classifyStatus(200), RETRIEVAL_OK)
  assert.equal(classifyStatus(204), RETRIEVAL_OK)
})

test("5xx and 429 mean the index was not reached", () => {
  for (const status of [500, 502, 503, 504, 429]) {
    assert.equal(classifyStatus(status), RETRIEVAL_BACKEND_DOWN, `status ${status}`)
  }
})

test("4xx is the call's own fault, not the backend's", () => {
  for (const status of [400, 404, 422]) {
    assert.equal(classifyStatus(status), RETRIEVAL_BAD_REQUEST, `status ${status}`)
  }
})

test("a fetch throw is always an unreached index", () => {
  assert.equal(classifyThrow(new TypeError("fetch failed")), RETRIEVAL_BACKEND_DOWN)
})

test("a turn whose only search hit a dead backend is ungrounded", () => {
  const health = createRetrievalHealth()
  recordRetrieval(health, RETRIEVAL_BACKEND_DOWN, "connection refused")
  assert.equal(isUngrounded(health), true)
  assert.equal(health.lastError, "connection refused")
})

test("empty results from a live backend are grounded, not degraded", () => {
  // The regression this guards: "no hits" must stay an ordinary answer.
  const health = createRetrievalHealth()
  recordRetrieval(health, RETRIEVAL_OK)
  assert.equal(isUngrounded(health), false)
})

test("a failure followed by a success is still grounded", () => {
  const health = createRetrievalHealth()
  recordRetrieval(health, RETRIEVAL_BACKEND_DOWN, "502")
  recordRetrieval(health, RETRIEVAL_OK)
  assert.equal(isUngrounded(health), false)
})

test("a malformed call alone does not mark the turn ungrounded", () => {
  const health = createRetrievalHealth()
  recordRetrieval(health, RETRIEVAL_BAD_REQUEST, "400")
  assert.equal(isUngrounded(health), false)
})

test("a clean turn is grounded", () => {
  assert.equal(isUngrounded(createRetrievalHealth()), false)
})

test("the model is instructed to stop, not merely informed", () => {
  const text = backendDownInstruction("connection refused")
  assert.match(text, /connection refused/)
  assert.match(text, /Do NOT answer from your own knowledge/)
  assert.match(text, /temporarily unavailable/)
})
