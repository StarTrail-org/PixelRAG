import { z } from "zod"
import { RETRIEVAL_OK, RETRIEVAL_BACKEND_DOWN, classifyStatus, classifyThrow, recordRetrieval, backendDownInstruction } from "./retrieval-health.mjs"

// An unreachable index is not information for the model to work around: it
// ends the turn. Recorded on the turn's health so the stream can close as
// degraded instead of done.
function retrievalDown(health, onEvent, label, detail) {
  recordRetrieval(health, RETRIEVAL_BACKEND_DOWN, detail)
  onEvent("search_unavailable", { query: label, reason: detail })
  return { content: [{ type: "text", text: backendDownInstruction(detail) }], isError: true }
}

function safeDecodeURIComponent(str) {
  try {
    return decodeURIComponent(str)
  } catch {
    return str
  }
}

export function createTools(onEvent, uploadedImage, health, tool, SEARCH_URL) {
  const searchTool = tool(
    "pixelrag_search",
    "Search the visual Wikipedia index by text, by the user's uploaded image, or BOTH combined. When the user uploaded an image, you MUST set use_uploaded_image=true AND provide a text query to get joint image+text retrieval — this gives the best results. Returns ranked results with article URLs, tile positions, and `pages` — the article's valid tile:chunk ranges (e.g. '0:0-7,1:0-4' = tile 0 has chunks 0-7, tile 1 has chunks 0-4). Use this first, then pixelrag_tile to view tiles.",
    {
      query: z.string().optional().describe("Natural language search query. Omit only when searching purely by an uploaded image."),
      use_uploaded_image: z.boolean().optional().describe("Set true to include the user's uploaded image in the search (visual similarity). ALWAYS combine with a text query for best results — set this AND provide a query string in the same call."),
      n_results: z.number().int().min(1).max(20).optional().describe("Number of results (default 5)"),
    },
    async (args) => {
      if (args.use_uploaded_image && !uploadedImage) {
        return { content: [{ type: "text", text: "No image was uploaded in this conversation — use a text query instead." }] }
      }
      const searchByImage = Boolean(args.use_uploaded_image && uploadedImage)
      if (!searchByImage && !args.query) {
        return { content: [{ type: "text", text: "Provide a text query, or set use_uploaded_image:true when the user uploaded an image." }] }
      }
      // Text and image can be combined in one query for joint image+text retrieval.
      const queryObj = {}
      if (searchByImage) queryObj.image = uploadedImage
      if (args.query) queryObj.text = args.query
      const label = searchByImage && args.query ? `${args.query} + uploaded image` : args.query || "uploaded image"
      onEvent("searching", { query: label })
      let resp
      try {
        resp = await fetch(`${SEARCH_URL}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Source": "chat" },
          body: JSON.stringify({ queries: [queryObj], n_docs: args.n_results ?? 5, articles_only: true }),
          signal: AbortSignal.timeout(30000),
        })
      } catch (err) {
        return retrievalDown(health, onEvent, label, String(err))
      }
      const outcome = classifyStatus(resp.status)
      if (outcome === RETRIEVAL_BACKEND_DOWN) {
        return retrievalDown(health, onEvent, label, `HTTP ${resp.status}`)
      }
      if (outcome !== RETRIEVAL_OK) {
        recordRetrieval(health, outcome, `HTTP ${resp.status}`)
        return { content: [{ type: "text", text: `Search API error: ${resp.status}` }] }
      }
      recordRetrieval(health, RETRIEVAL_OK)
      const data = await resp.json()
      const hits = data.results?.[0]?.hits ?? []
      const results = hits.map((h) => {
        const slug = h.url.includes("/wiki/") ? h.url.split("/wiki/").pop() : h.url
        return {
          title: safeDecodeURIComponent(slug || "").replace(/_/g, " "),
          url: h.url.startsWith("http") ? h.url : `https://en.wikipedia.org/wiki/${slug}`,
          score: Math.round(h.score * 1000) / 1000,
          article_id: h.article_id,
          tile_index: h.tile_index,
          chunk_index: h.chunk_index,
          pages: h.article_pages,
        }
      })
      onEvent("search_results", { query: label, hits })
      return {
        content: [{ type: "text", text: JSON.stringify({ query: label, results, count: results.length }, null, 2) }],
      }
    }
  )

  const tileTool = tool(
    "pixelrag_tile",
    "View a Wikipedia screenshot tile by its coordinates. Returns the tile as an image so you can read the visual content. Only request coordinates within the article's `pages` ranges from search results (e.g. pages '0:0-7,1:0-4' means tile 1 ends at chunk 4) — coordinates beyond them do not exist.",
    {
      article_id: z.number().int().describe("Article ID from search results"),
      tile_index: z.number().int().describe("Tile index from search results"),
      chunk_index: z.number().int().describe("Chunk index from search results"),
    },
    async (args) => {
      const tileUrl = `${SEARCH_URL}/tile/${args.article_id}/${args.tile_index}/${args.chunk_index}`
      try {
        const resp = await fetch(tileUrl, { signal: AbortSignal.timeout(30000) })
        // The agent pages through articles by guessing chunk coordinates, so
        // 404s are normal exploration — only surface tiles that actually load,
        // otherwise the chat gallery renders broken images.
        if (!resp.ok) {
          if (classifyStatus(resp.status) === RETRIEVAL_BACKEND_DOWN) {
            recordRetrieval(health, RETRIEVAL_BACKEND_DOWN, `tile HTTP ${resp.status}`)
          }
          return { content: [{ type: "text", text: `Tile not found: ${resp.status}` }] }
        }
        onEvent("viewing_tile", { article_id: args.article_id, tile_index: args.tile_index, chunk_index: args.chunk_index })
        const buffer = await resp.arrayBuffer()
        const base64 = Buffer.from(buffer).toString("base64")
        const mimeType = resp.headers.get("content-type") || "image/png"
        return { content: [{ type: "image", data: base64, mimeType }] }
      } catch (err) {
        recordRetrieval(health, classifyThrow(err), `tile ${err}`)
        return { content: [{ type: "text", text: `Failed to fetch tile: ${err}` }] }
      }
    }
  )

  return [searchTool, tileTool]
}
