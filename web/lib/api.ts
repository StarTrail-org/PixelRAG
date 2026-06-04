import type { SearchRequest, SearchResponse, StatusResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  return fetchApi<SearchResponse>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // NOTE: the visual grid intentionally does NOT set articles_only — the
    // Portal/Featured-picture pages it would drop are often the most visually
    // striking tiles (panoramas, paintings), which is the point of this view.
    // The chat agent DOES filter (it reads article text to answer questions).
    body: JSON.stringify(req),
  });
}

export async function getStatus(): Promise<StatusResponse> {
  return fetchApi<StatusResponse>("/status");
}

export async function getHealth(): Promise<{ status: string }> {
  return fetchApi<{ status: string }>("/health");
}

export function tileUrl(hit: {
  article_id: number;
  tile_index: number;
  chunk_index: number;
}): string {
  return `${API_BASE}/tile/${hit.article_id}/${hit.tile_index}/${hit.chunk_index}`;
}

export async function reconstruct(
  vectorIds: number[]
): Promise<{ embeddings: number[][] }> {
  return fetchApi<{ embeddings: number[][] }>("/reconstruct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vector_ids: vectorIds }),
  });
}
