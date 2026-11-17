// Type surface for the shared classifier. The implementation stays plain
// `.mjs` because `web/agent-server.mjs` imports it directly under node, with
// no build step between them.

export type RetrievalOutcome = "ok" | "backend_down" | "bad_request"

export interface RetrievalHealth {
  ok: number
  backendDown: number
  badRequest: number
  lastError: string | null
}

export declare const RETRIEVAL_OK: "ok"
export declare const RETRIEVAL_BACKEND_DOWN: "backend_down"
export declare const RETRIEVAL_BAD_REQUEST: "bad_request"
export declare const UNGROUNDED_CLIENT_MESSAGE: string

export declare function classifyStatus(status: number): RetrievalOutcome
export declare function classifyThrow(err?: unknown): RetrievalOutcome
export declare function createRetrievalHealth(): RetrievalHealth
export declare function recordRetrieval(
  health: RetrievalHealth,
  outcome: RetrievalOutcome,
  detail?: string | null
): RetrievalHealth
export declare function isUngrounded(health: RetrievalHealth): boolean
export declare function backendDownInstruction(reason: string): string
