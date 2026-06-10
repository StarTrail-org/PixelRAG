"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { ApiPlayground } from "@/components/ApiPlayground"

interface Field {
  name: string
  type: string
  required?: boolean
  description?: string
  children?: Field[]
}

interface Endpoint {
  id: string
  method: "GET" | "POST"
  path: string
  summary: string
  description: string
  requestFields?: Field[]
  responseFields?: Field[]
  curlPrefix: string
  defaultBody?: string
  defaultParams?: string
  buildPath?: (body: string, params: string) => string
}

const endpoints: Endpoint[] = [
  {
    id: "search",
    method: "POST",
    path: "/search",
    summary: "Search for visually similar tiles",
    description:
      "Submit one or more queries (text, image, or embedding) and retrieve the top matching tiles. Each hit returns article_id, tile_index, and chunk_index — pass these to GET /tile/... to fetch the actual screenshot.",
    requestFields: [
      { name: "queries", type: "Query[]", required: true, description: "Array of query objects", children: [
        { name: "text", type: "string", description: "Text search query" },
        { name: "image", type: "string", description: "Base64-encoded image" },
        { name: "embedding", type: "number[]", description: "Pre-computed embedding vector" },
      ]},
      { name: "n_docs", type: "number", description: "Number of results to return (default: 20)" },
      { name: "nprobe", type: "number", description: "FAISS nprobe override" },
      { name: "min_tile_height", type: "number", description: "Filter out tiles shorter than this" },
      { name: "articles_only", type: "boolean", description: "Drop Wikipedia meta pages (Portal:, List_of_, disambiguation, …) — keeps real articles" },
      { name: "instruction", type: "string", description: "Custom embedding instruction" },
    ],
    responseFields: [
      { name: "results", type: "QueryResult[]", required: true, description: "One result per query", children: [
        { name: "hits", type: "Hit[]", required: true, description: "Ranked list of matches", children: [
          { name: "score", type: "number", required: true, description: "Cosine similarity" },
          { name: "vector_id", type: "number", required: true, description: "FAISS vector index" },
          { name: "article_id", type: "number", required: true, description: "Wikipedia article ID" },
          { name: "tile_index", type: "number", required: true, description: "Which 8192px tile" },
          { name: "chunk_index", type: "number", required: true, description: "Which 1024px chunk within tile" },
          { name: "y_offset", type: "number", required: true, description: "Y position in page (px)" },
          { name: "tile_height", type: "number", required: true, description: "Chunk height (px)" },
          { name: "path", type: "string", required: true, description: "Tile file path on server" },
          { name: "url", type: "string", required: true, description: "Wikipedia article slug" },
        ]},
      ]},
    ],
    curlPrefix: `curl -X POST https://pixelrag.ai/api/search \\
  -H "Content-Type: application/json"`,
    defaultBody: JSON.stringify({ queries: [{ text: "solar system" }], n_docs: 5 }, null, 2),
  },
  {
    id: "tile",
    method: "GET",
    path: "/tile/{article_id}/{tile_index}/{chunk_index}",
    summary: "Retrieve a tile image",
    description:
      "Fetches the screenshot image for a /search hit — pass the article_id, tile_index, and chunk_index straight from a search result. Returns PNG binary data; embed it with an <img> tag. The example below is the top of the Albert Einstein article.",
    requestFields: [
      { name: "article_id", type: "number", required: true, description: "Wikipedia article ID" },
      { name: "tile_index", type: "number", required: true, description: "Which 8192px tile (0-based)" },
      { name: "chunk_index", type: "number", required: true, description: "Which 1024px chunk within tile (0-based)" },
    ],
    responseFields: [
      { name: "(binary)", type: "image/png", required: true, description: "PNG image data" },
    ],
    curlPrefix: `curl https://pixelrag.ai/api/tile`,
    defaultParams: "article_id=698618&tile_index=0&chunk_index=0",
    buildPath: (_body, params) => {
      const p = new URLSearchParams(params)
      return `/tile/${p.get("article_id") || "698618"}/${p.get("tile_index") || "0"}/${p.get("chunk_index") || "0"}`
    },
  },
  {
    id: "status",
    method: "GET",
    path: "/status",
    summary: "Get index status and configuration",
    description:
      "Returns metadata about the FAISS index including vector count, dimension, model, and configuration.",
    responseFields: [
      { name: "total_vectors", type: "number", required: true, description: "Total indexed vectors" },
      { name: "dimension", type: "number", required: true, description: "Embedding dimension" },
      { name: "nlist", type: "number", required: true, description: "IVF cluster count" },
      { name: "nprobe", type: "number", required: true, description: "Search probe count" },
      { name: "model", type: "string", required: true, description: "Embedding model name" },
      { name: "index_dir", type: "string", required: true, description: "Index directory path" },
      { name: "tiles_dir", type: "string", required: true, description: "Tiles directory path" },
      { name: "index_built_at", type: "string", required: true, description: "ISO 8601 timestamp" },
      { name: "index_size_bytes", type: "number", required: true, description: "FAISS index file size" },
      { name: "metadata_size_bytes", type: "number", required: true, description: "Metadata NPZ size" },
    ],
    curlPrefix: `curl https://pixelrag.ai/api/status`,
  },
  {
    id: "health",
    method: "GET",
    path: "/health",
    summary: "Health check",
    description:
      "Returns {\"status\": \"ok\"} when the server is running.",
    responseFields: [
      { name: "status", type: "string", required: true, description: "Always \"ok\"" },
    ],
    curlPrefix: `curl https://pixelrag.ai/api/health`,
  },
]

export default function DocsPage() {
  const initialId = typeof window !== "undefined" ? window.location.hash.slice(1) : ""
  const [activeId, setActiveId] = React.useState<string>(
    endpoints.find((e) => e.id === initialId)?.id ?? "overview"
  )
  const active = endpoints.find((e) => e.id === activeId)

  function selectEndpoint(id: string) {
    setActiveId(id)
    window.history.replaceState(null, "", `#${id}`)
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-0 px-4 py-8 sm:px-6 md:flex-row md:py-10">
      {/* Sidebar */}
      <aside className="hidden w-56 shrink-0 pr-6 md:block">
        <nav className="mb-6 space-y-1">
          <button
            onClick={() => selectEndpoint("overview")}
            className={cn(
              "flex w-full items-center rounded-md px-2.5 py-1.5 text-left text-sm transition-colors",
              activeId === "overview"
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
            )}
          >
            Overview
          </button>
        </nav>
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Endpoints
        </h2>
        <nav className="space-y-1">
          {endpoints.map((ep) => (
            <button
              key={ep.id}
              onClick={() => selectEndpoint(ep.id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors",
                activeId === ep.id
                  ? "bg-muted font-medium text-foreground"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
            >
              <MethodBadge method={ep.method} />
              <span className="truncate font-mono text-xs">{ep.path}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* Mobile endpoint selector */}
      <div className="mb-6 md:hidden">
        <select
          value={activeId}
          onChange={(e) => selectEndpoint(e.target.value)}
          className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="overview">Overview</option>
          {endpoints.map((ep) => (
            <option key={ep.id} value={ep.id}>
              {ep.method} {ep.path}
            </option>
          ))}
        </select>
      </div>

      {/* Main content */}
      <div className="min-w-0 flex-1">
        {active ? (
          <>
            <div className="flex items-center gap-3">
              <MethodBadge method={active.method} />
              <h1 className="font-mono text-lg font-semibold">{active.path}</h1>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">{active.summary}</p>

            <div className="mt-6 space-y-6">
              {/* Description */}
              <p className="text-sm leading-relaxed text-foreground/80">
                {active.description}
              </p>

              {/* Try it — most useful, put first */}
              <ApiPlayground
                key={active.id}
                method={active.method}
                path={active.path}
                curlPrefix={active.curlPrefix}
                defaultBody={active.defaultBody}
                defaultParams={active.defaultParams}
                buildPath={active.buildPath}
              />

              {/* Schema — Request + Response side by side when both exist */}
              {active.requestFields ? (
                <div className="grid gap-4 lg:grid-cols-2">
                  <Section title="Request">
                    <FieldTable fields={active.requestFields} />
                  </Section>
                  {active.responseFields && (
                    <Section title="Response">
                      <FieldTable fields={active.responseFields} />
                    </Section>
                  )}
                </div>
              ) : active.responseFields ? (
                <Section title="Response">
                  <FieldTable fields={active.responseFields} />
                </Section>
              ) : null}
            </div>
          </>
        ) : (
          <OverviewSection onSelect={selectEndpoint} />
        )}
      </div>
    </div>
  )
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground">
      {children}
    </code>
  )
}

// Minimal, dependency-free shell highlighter for the curl examples — colors
// comments, commands, flags, quoted strings, and URLs.
function ShellBlock({ code }: { code: string }) {
  const TOKEN =
    /(#.*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(https?:\/\/[^\s'"]+)|(\bcurl\b)|(\s-{1,2}[A-Za-z][\w-]*)/g
  const nodes: React.ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  let key = 0
  while ((m = TOKEN.exec(code)) !== null) {
    if (m.index > last) nodes.push(code.slice(last, m.index))
    const [full, comment, str, url, cmd] = m
    const cls = comment
      ? "italic text-muted-foreground/60"
      : str
        ? "text-green-600 dark:text-green-400"
        : url
          ? "text-blue-600 dark:text-blue-400"
          : cmd
            ? "font-medium text-purple-600 dark:text-purple-400"
            : "text-amber-600 dark:text-amber-400" // flag
    nodes.push(
      <span key={key++} className={cls}>
        {full}
      </span>
    )
    last = m.index + full.length
  }
  if (last < code.length) nodes.push(code.slice(last))
  return (
    <pre className="mt-3 overflow-x-auto rounded-xl border border-border/60 bg-card p-4 text-xs leading-relaxed text-foreground/80">
      <code>{nodes}</code>
    </pre>
  )
}

function FlowStep({
  n,
  method,
  path,
  title,
  onSelect,
  children,
}: {
  n: number
  method: "GET" | "POST"
  path: string
  title: string
  onSelect: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onSelect}
      className="flex w-full gap-4 rounded-xl border border-border/60 bg-card/40 p-4 text-left transition-colors hover:border-border hover:bg-muted/40"
    >
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
        {n}
      </span>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <MethodBadge method={method} />
          <code className="truncate font-mono text-xs text-foreground">{path}</code>
        </div>
        <p className="mt-2 text-sm font-medium text-foreground">{title}</p>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{children}</p>
      </div>
    </button>
  )
}

function OverviewSection({ onSelect }: { onSelect: (id: string) => void }) {
  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">API Overview</h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        PixelRAG indexes 8.28M Wikipedia articles as{" "}
        <strong className="font-medium text-foreground">screenshot tiles</strong> and retrieves over
        the images directly. Using the API is a two-step flow: search for the tiles that match your
        query, then fetch each tile&apos;s screenshot.
      </p>

      <div className="mt-8 space-y-3">
        <FlowStep
          n={1}
          method="POST"
          path="/search"
          title="Find matching tiles"
          onSelect={() => onSelect("search")}
        >
          Send a natural-language question or an image. Each hit comes back with{" "}
          <Code>article_id</Code>, <Code>tile_index</Code>, and <Code>chunk_index</Code>.
        </FlowStep>
        <FlowStep
          n={2}
          method="GET"
          path="/tile/{article_id}/{tile_index}/{chunk_index}"
          title="Fetch the screenshot"
          onSelect={() => onSelect("tile")}
        >
          Pass those three IDs straight from a search hit to get the PNG — embed it with an{" "}
          <Code>&lt;img&gt;</Code> tag.
        </FlowStep>
      </div>

      <h2 className="mt-10 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Example
      </h2>
      <p className="mt-2 text-sm text-muted-foreground">
        Search, then fetch the top hit&apos;s screenshot:
      </p>
      <ShellBlock
        code={`# 1 — search with a text query
curl -X POST https://pixelrag.ai/api/search \\
  -H "Content-Type: application/json" \\
  -d '{"queries": [{"text": "How does photosynthesis work?"}], "n_docs": 5}'

# → hits[0] = { "article_id": 5878188, "tile_index": 1, "chunk_index": 0, ... }

# 2 — fetch that hit's screenshot
curl https://pixelrag.ai/api/tile/5878188/1/0 --output tile.png

# 3 — or search with an image: base64-encode any local file inline
curl -X POST https://pixelrag.ai/api/search \\
  -H "Content-Type: application/json" \\
  -d "{\\"queries\\": [{\\"image\\": \\"$(base64 < photo.jpg | tr -d '\\n')\\"}], \\"n_docs\\": 5}"

# 4 — hybrid: combine text + image in one query (joint multimodal retrieval)
curl -X POST https://pixelrag.ai/api/search \\
  -H "Content-Type: application/json" \\
  -d "{\\"queries\\": [{\\"text\\": \\"impressionist oil painting\\", \\"image\\": \\"$(base64 < photo.jpg | tr -d '\\n')\\"}], \\"n_docs\\": 5}"`}
      />

      <h2 className="mt-10 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Base URL
      </h2>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        <Code>https://pixelrag.ai/api</Code> — or query the index server directly at{" "}
        <Code>https://api.pixelrag.ai</Code>.
      </p>

      <h2 className="mt-10 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Endpoints
      </h2>
      <div className="mt-3 space-y-1.5">
        {endpoints.map((ep) => (
          <button
            key={ep.id}
            onClick={() => onSelect(ep.id)}
            className="flex w-full items-center gap-3 rounded-lg border border-border/40 px-3 py-2 text-left text-sm transition-colors hover:border-border hover:bg-muted/40"
          >
            <MethodBadge method={ep.method} />
            <code className="shrink-0 font-mono text-xs text-foreground">{ep.path}</code>
            <span className="truncate text-xs text-muted-foreground">{ep.summary}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function MethodBadge({ method }: { method: "GET" | "POST" }) {
  return (
    <Badge
      variant="secondary"
      className={cn(
        "shrink-0 font-mono text-[0.6rem] font-bold uppercase",
        method === "POST"
          ? "bg-green-500/15 text-green-700 dark:bg-green-500/20 dark:text-green-400"
          : "bg-blue-500/15 text-blue-700 dark:bg-blue-500/20 dark:text-blue-400"
      )}
    >
      {method}
    </Badge>
  )
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      {children}
    </div>
  )
}


function TypeBadge({ type }: { type: string }) {
  const color = type.includes("[]")
    ? "text-amber-400 bg-amber-400/10"
    : type === "number"
      ? "text-blue-400 bg-blue-400/10"
      : type === "string" || type.startsWith("image/")
        ? "text-green-400 bg-green-400/10"
        : "text-purple-400 bg-purple-400/10"
  return (
    <span className={cn("rounded px-1.5 py-0.5 font-mono text-[10px] font-medium", color)}>
      {type}
    </span>
  )
}

function FieldRow({ field, depth = 0 }: { field: Field; depth?: number }) {
  return (
    <>
      <div
        className="flex items-start gap-3 border-b border-border/30 px-3 py-2.5 text-xs"
        style={{ paddingLeft: `${12 + depth * 16}px` }}
      >
        <div className="flex shrink-0 items-center gap-2" style={{ minWidth: "120px" }}>
          <span className="font-mono font-medium text-foreground">{field.name}</span>
          {!field.required && (
            <span className="text-[10px] italic text-muted-foreground/60">optional</span>
          )}
        </div>
        {field.description && (
          <span className="min-w-0 flex-1 truncate text-muted-foreground" title={field.description}>
            {field.description}
          </span>
        )}
        <TypeBadge type={field.type} />
      </div>
      {field.children?.map((child) => (
        <FieldRow key={child.name} field={child} depth={depth + 1} />
      ))}
    </>
  )
}

function FieldTable({ fields }: { fields: Field[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/60 bg-card">
      {fields.map((field) => (
        <FieldRow key={field.name} field={field} />
      ))}
    </div>
  )
}
