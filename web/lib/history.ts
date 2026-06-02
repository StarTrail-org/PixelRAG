// Per-user, client-side query history kept in localStorage. Two independent
// namespaces: "search" (the visual search box) and "ask" (the agent chat).
type Kind = "search" | "ask"

const MAX = 10
const keyFor = (kind: Kind) => `pixelrag-${kind}-history`

export function getHistory(kind: Kind = "search"): string[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(keyFor(kind))
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter((s): s is string => typeof s === "string").slice(0, MAX)
  } catch {
    return []
  }
}

export function addHistory(query: string, kind: Kind = "search"): void {
  if (typeof window === "undefined") return
  const trimmed = query.trim()
  if (!trimmed) return
  try {
    const prev = getHistory(kind)
    const deduped = prev.filter((s) => s !== trimmed)
    const next = [trimmed, ...deduped].slice(0, MAX)
    localStorage.setItem(keyFor(kind), JSON.stringify(next))
  } catch {
    // localStorage unavailable — silently ignore
  }
}

export function clearHistory(kind: Kind = "search"): void {
  if (typeof window === "undefined") return
  try {
    localStorage.removeItem(keyFor(kind))
  } catch {
    // silently ignore
  }
}
