"use client"

import * as React from "react"
import {
  Send,
  Search,
  ExternalLink,
  Loader2,
  RotateCcw,
  Eye,
  ArrowRight,
  Maximize2,
  Brain,
  ChevronRight,
  ImagePlus,
  X,
} from "lucide-react"
import { tileUrl } from "@/lib/api"
import { motion, AnimatePresence } from "framer-motion"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ModeToggle } from "@/components/ModeToggle"

interface SearchResult {
  query: string
  hits: {
    score: number
    article_id: number
    tile_index: number
    chunk_index: number
    url: string
    tile_height: number
  }[]
}

interface TileView {
  article_id: number
  tile_index: number
  chunk_index: number
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  image?: string
  thinking?: string
  searches?: SearchResult[]
  searching?: string
  tiles?: TileView[]
  viewingTile?: boolean
}

const EXAMPLES = [
  { q: "Explain Van Gogh's The Starry Night", icon: "01" },
  { q: "How do you read the periodic table?", icon: "02" },
  { q: "Tell me about the Taj Mahal", icon: "03" },
  { q: "介绍一下兵马俑", icon: "04" },
]

function ChatPageInner() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [input, setInput] = React.useState("")
  const [image, setImage] = React.useState<string | undefined>()
  const [isStreaming, setIsStreaming] = React.useState(false)
  const messagesEndRef = React.useRef<HTMLDivElement>(null)
  const inputRef = React.useRef<HTMLTextAreaElement>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const abortRef = React.useRef<AbortController | null>(null)
  const handleSendRef = React.useRef<((text?: string, imageOverride?: string) => void) | null>(null)
  const didInitRef = React.useRef(false)

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  React.useEffect(() => {
    scrollToBottom()
  }, [messages])

  React.useEffect(() => {
    handleSendRef.current = handleSend
  })

  // Auto-send query handed off from the Search page (Ask mode)
  React.useEffect(() => {
    if (didInitRef.current) return
    const q = searchParams.get("q")
    // The attached image (if any) is passed via sessionStorage, since a
    // base64 image can't fit in a URL query param.
    let pendingImage: string | undefined
    if (searchParams.get("img")) {
      try {
        pendingImage = sessionStorage.getItem("pixelrag:pending-image") || undefined
        sessionStorage.removeItem("pixelrag:pending-image")
      } catch {
        pendingImage = undefined
      }
    }
    if (q || pendingImage) {
      didInitRef.current = true
      router.replace("/chat", { scroll: false })
      handleSendRef.current?.(q || "", pendingImage)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleSend(text?: string, imageOverride?: string) {
    const query = (text ?? input).trim()
    const img = imageOverride ?? image
    if ((!query && !img) || isStreaming) return
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: query, image: img }
    const assistantMsg: ChatMessage = { id: crypto.randomUUID(), role: "assistant", content: "", searches: [] }
    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setInput("")
    setImage(undefined)
    setIsStreaming(true)
    const allMessages = [
      ...messages.filter((m) => m.content).map((m) => ({ role: m.role, content: m.content })),
      { role: "user" as const, content: query, ...(img ? { image: img } : {}) },
    ]
    const abort = new AbortController()
    abortRef.current = abort
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: allMessages }),
        signal: abort.signal,
      })
      if (!resp.ok) { const err = await resp.json(); throw new Error(err.error || `HTTP ${resp.status}`) }
      const reader = resp.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""
        let eventType = ""
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7)
          else if (line.startsWith("data: ")) handleSSEEvent(assistantMsg.id, eventType, JSON.parse(line.slice(6)))
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: m.content || `Error: ${err}`, searching: undefined } : m))
      }
    } finally { setIsStreaming(false); abortRef.current = null }
  }

  function handleSSEEvent(msgId: string, event: string, data: Record<string, unknown>) {
    setMessages((prev) => prev.map((m) => {
      if (m.id !== msgId) return m
      switch (event) {
        case "thinking": return { ...m, thinking: (m.thinking || "") + (data.text as string) }
        case "text": return { ...m, content: m.content + (data.text as string) }
        case "searching": return { ...m, searching: data.query as string }
        case "search_results": return { ...m, searching: undefined, searches: [...(m.searches || []), { query: data.query as string, hits: data.hits as SearchResult["hits"] }] }
        case "viewing_tile": return { ...m, viewingTile: true, tiles: [...(m.tiles || []), { article_id: data.article_id as number, tile_index: data.tile_index as number, chunk_index: data.chunk_index as number }] }
        case "done": return { ...m, searching: undefined, viewingTile: false }
        case "error": return { ...m, content: m.content || `Error: ${data.message}`, searching: undefined }
        default: return m
      }
    }))
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  function handleReset() {
    if (abortRef.current) abortRef.current.abort()
    setMessages([]); setIsStreaming(false); setInput(""); setImage(undefined); inputRef.current?.focus()
  }

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) return
    const reader = new FileReader()
    reader.onload = (e) => setImage(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  function handlePaste(e: React.ClipboardEvent) {
    const file = Array.from(e.clipboardData.items)
      .find((it) => it.type.startsWith("image/"))
      ?.getAsFile()
    if (file) { e.preventDefault(); handleFile(file) }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-page flex h-[calc(100vh-3.5rem)] flex-col">
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <EmptyState onExample={handleSend} onSearchMode={() => router.push("/")} />
        ) : (
          <div className="mx-auto max-w-[720px] px-5 py-8">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                  {msg.role === "user" ? <UserMessage content={msg.content} image={msg.image} /> : <AssistantMessage message={msg} isStreaming={isStreaming && msg.id === messages[messages.length - 1]?.id} />}
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="chat-input-area border-t border-[var(--chat-border)]">
        <div className="mx-auto flex max-w-[720px] items-end gap-2 px-5 py-3.5">
          {messages.length > 0 && (
            <button onClick={handleReset} className="mb-1.5 rounded-lg p-2 text-[var(--chat-muted)] transition-colors hover:text-[var(--chat-fg)]" title="New conversation">
              <RotateCcw className="h-4 w-4" />
            </button>
          )}
          <div
            className="chat-input flex min-h-[50px] flex-1 flex-col rounded-2xl border border-[var(--chat-border)] bg-[var(--chat-input-bg)] px-4 py-1.5 transition-all focus-within:border-[var(--chat-accent)] focus-within:shadow-[0_0_0_3px_var(--chat-accent-glow)]"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {image && (
              <div className="relative mt-1.5 w-fit">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image} alt="attachment" className="max-h-24 rounded-lg border border-[var(--chat-border)] object-contain" />
                <button
                  onClick={() => setImage(undefined)}
                  className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--chat-fg)] text-[var(--chat-bg)] shadow"
                  title="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            )}
          <div className="flex items-end">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = "" }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming}
              className="mb-1.5 mr-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-[var(--chat-muted)] transition-colors hover:bg-[var(--chat-card-hover)] hover:text-[var(--chat-fg)] disabled:opacity-30"
              title="Attach image"
            >
              <ImagePlus className="h-4 w-4" />
            </button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px" }}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="Ask anything, or paste an image…"
              rows={1}
              className="flex-1 resize-none bg-transparent py-2 text-[14px] leading-relaxed text-[var(--chat-fg)] outline-none placeholder:text-[var(--chat-muted)]"
              disabled={isStreaming}
            />
            <button
              onClick={() => handleSend()}
              disabled={(!input.trim() && !image) || isStreaming}
              className="mb-1.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[var(--chat-accent)] text-white transition-all hover:brightness-110 disabled:opacity-20"
            >
              {isStreaming ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
            </button>
          </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  )
}

/* ─── Empty State ─── */

function EmptyState({ onExample, onSearchMode }: { onExample: (q: string) => void; onSearchMode: () => void }) {
  return (
    <div className="relative flex h-full flex-col items-center justify-center px-6">
      {/* Background mesh */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-1/4 top-1/4 h-[400px] w-[400px] rounded-full bg-[var(--chat-accent)] opacity-[0.03] blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 h-[300px] w-[300px] rounded-full bg-[var(--chat-warm)] opacity-[0.04] blur-[100px]" />
        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-[0.015]" style={{ backgroundImage: "linear-gradient(var(--chat-fg) 1px, transparent 1px), linear-gradient(90deg, var(--chat-fg) 1px, transparent 1px)", backgroundSize: "60px 60px" }} />
      </div>

      <div className="relative z-10 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-[var(--chat-border)] px-5 py-2 text-[11px] font-medium tracking-[0.15em] text-[var(--chat-muted)]" style={{ fontFamily: "var(--font-mono)" }}>
            <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" /><span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" /></span>
            PIXELRAG AGENT
          </div>
        </motion.div>

        <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }} className="font-display text-5xl font-light leading-[1.15] tracking-tight text-[var(--chat-fg)] sm:text-6xl">
          Search Wikipedia
          <br />
          <span className="bg-gradient-to-r from-[var(--chat-accent)] to-[var(--chat-warm)] bg-clip-text text-transparent">
            visually.
          </span>
        </motion.h1>

        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mx-auto mt-5 max-w-md text-[14px] leading-relaxed text-[var(--chat-secondary)]">
          Ask a question. I&apos;ll search 28.1M Wikipedia screenshot tiles,
          read the visual results, and synthesize an answer.
        </motion.p>
      </div>

      {/* Mode toggle */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="relative z-10 mt-8">
        <ModeToggle mode="ask" onChange={(m) => { if (m === "search") onSearchMode() }} />
      </motion.div>

      {/* Examples */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="relative z-10 mt-8 grid w-full max-w-lg grid-cols-2 gap-3">
        {EXAMPLES.map(({ q, icon }) => (
          <button
            key={q}
            onClick={() => onExample(q)}
            className="group flex items-start gap-3 rounded-xl border border-[var(--chat-border)] bg-[var(--chat-card)] p-4 text-left transition-all hover:border-[var(--chat-accent-dim)] hover:bg-[var(--chat-card-hover)]"
          >
            <span className="mt-0.5 shrink-0 font-mono text-[11px] text-[var(--chat-accent)] opacity-50 group-hover:opacity-100">{icon}</span>
            <span className="text-[13px] leading-snug text-[var(--chat-secondary)] group-hover:text-[var(--chat-fg)]">
              {q}
            </span>
            <ArrowRight className="ml-auto mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--chat-muted)] opacity-0 transition-all group-hover:translate-x-0.5 group-hover:opacity-60" />
          </button>
        ))}
      </motion.div>
    </div>
  )
}

/* ─── Messages ─── */

function UserMessage({ content, image }: { content: string; image?: string }) {
  return (
    <div className="mb-5 flex flex-col items-end gap-1.5">
      {image && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={image} alt="attached" className="max-h-48 max-w-[80%] rounded-2xl rounded-br-md border border-[var(--chat-border)] object-contain" />
      )}
      {content && (
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-[var(--chat-accent)] px-4 py-2.5 text-[14px] leading-relaxed text-white shadow-lg shadow-[var(--chat-accent-glow)]">
          {content}
        </div>
      )}
    </div>
  )
}

function AssistantMessage({ message, isStreaming }: { message: ChatMessage; isStreaming: boolean }) {
  return (
    <div className="mb-8">
      {message.thinking && (
        <ThinkingTrace text={message.thinking} active={isStreaming && !message.content} />
      )}
      {message.searches?.map((s, i) => <SearchCard key={i} result={s} />)}
      {message.tiles && message.tiles.length > 0 && <TileGallery tiles={message.tiles} loading={message.viewingTile} />}

      {message.searching && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-4 flex items-center gap-2.5 rounded-lg border border-[var(--chat-border)] bg-[var(--chat-card)] px-3.5 py-2.5">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--chat-accent)]" />
          <span className="text-[12px] text-[var(--chat-secondary)]">Searching <span className="font-medium text-[var(--chat-fg)]">&ldquo;{message.searching}&rdquo;</span></span>
        </motion.div>
      )}

      {message.content && (
        <div className="chat-prose prose prose-sm max-w-none text-[14px] leading-[1.8] text-[var(--chat-secondary)] prose-headings:font-display prose-headings:font-normal prose-headings:tracking-tight prose-headings:text-[var(--chat-fg)] prose-h2:mt-6 prose-h2:text-lg prose-h3:mt-4 prose-h3:text-[15px] prose-p:text-[var(--chat-secondary)] prose-a:text-[var(--chat-accent)] prose-a:no-underline hover:prose-a:underline prose-strong:font-semibold prose-strong:text-[var(--chat-fg)] prose-code:rounded-md prose-code:bg-[var(--chat-card)] prose-code:px-1.5 prose-code:py-0.5 prose-code:text-[12px] prose-code:text-[var(--chat-secondary)] prose-code:before:content-none prose-code:after:content-none prose-pre:border prose-pre:border-[var(--chat-border)] prose-pre:bg-[var(--chat-card)] prose-pre:rounded-xl prose-li:marker:text-[var(--chat-muted)] prose-hr:border-[var(--chat-border)] prose-blockquote:border-l-[var(--chat-accent-dim)] prose-blockquote:text-[var(--chat-muted)] prose-th:font-mono prose-th:text-[11px] prose-th:font-medium prose-th:uppercase prose-th:tracking-wider prose-th:text-[var(--chat-muted)] prose-td:text-[var(--chat-secondary)] prose-table:text-[13px]">
          <Markdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children }) => (<a href={href} target="_blank" rel="noopener noreferrer">{href?.includes("wikipedia.org") ? decodeURIComponent(href.split("/wiki/").pop() || String(children)).replace(/_/g, " ") : children}</a>) }}>
            {message.content}
          </Markdown>
        </div>
      )}

      {isStreaming && !message.content && !message.searching && (
        <div className="flex items-center gap-2.5 py-3">
          <div className="flex gap-1.5">
            {[0, 1, 2].map((i) => (<motion.span key={i} className="h-1.5 w-1.5 rounded-full bg-[var(--chat-accent)]" animate={{ opacity: [0.2, 0.8, 0.2] }} transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.15 }} />))}
          </div>
          {message.tiles && message.tiles.length > 0 && !message.viewingTile && (
            <span className="text-[12px] text-[var(--chat-muted)]">Reading the screenshots and writing your answer…</span>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Search Card ─── */

function SearchCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = React.useState(false)
  const shown = expanded ? result.hits : result.hits.slice(0, 5)

  return (
    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="mb-4 overflow-hidden rounded-xl border border-[var(--chat-border)] bg-[var(--chat-card)]">
      <button onClick={() => setExpanded(!expanded)} className="flex w-full items-center gap-2 border-b border-[var(--chat-border)] px-4 py-2.5 text-left">
        <Search className="h-3.5 w-3.5 text-[var(--chat-accent)]" />
        <span className="text-[12px] text-[var(--chat-secondary)]">&ldquo;{result.query}&rdquo;</span>
        <span className="ml-auto rounded-full bg-[var(--chat-accent)] bg-opacity-10 px-2 py-0.5 font-mono text-[10px] tabular-nums text-[var(--chat-accent)]">{result.hits.length}</span>
      </button>
      <div className="flex gap-1 overflow-x-auto p-2 scrollbar-thin">
        {shown.map((hit, i) => {
          const slug = hit.url.includes("/wiki/") ? hit.url.split("/wiki/").pop() : hit.url
          const title = decodeURIComponent(slug || "").replace(/_/g, " ")
          const fullUrl = hit.url.startsWith("http") ? hit.url : `https://en.wikipedia.org/wiki/${slug}`
          return (
            <a key={i} href={fullUrl} target="_blank" rel="noopener noreferrer" className="group relative shrink-0 overflow-hidden rounded-lg transition-transform hover:scale-[1.02]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={tileUrl(hit)} alt={title} className="h-24 w-36 object-cover object-top" loading="lazy" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
              <div className="absolute inset-x-0 bottom-0 translate-y-1 px-2 pb-1.5 opacity-0 transition-all group-hover:translate-y-0 group-hover:opacity-100">
                <span className="flex items-center gap-1 text-[10px] font-medium text-white"><span className="truncate">{title}</span><ExternalLink className="h-2.5 w-2.5 shrink-0 opacity-60" /></span>
              </div>
              <div className="absolute right-1 top-1 rounded bg-black/60 px-1 py-0.5 font-mono text-[8px] tabular-nums text-white/60 backdrop-blur-sm">{hit.score.toFixed(2)}</div>
            </a>
          )
        })}
      </div>
      {!expanded && result.hits.length > 5 && (
        <button onClick={() => setExpanded(true)} className="w-full border-t border-[var(--chat-border)] py-2 text-center text-[11px] text-[var(--chat-muted)] transition-colors hover:text-[var(--chat-accent)]">
          Show {result.hits.length - 5} more results
        </button>
      )}
    </motion.div>
  )
}

/* ─── Thinking Trace ─── */

function ThinkingTrace({ text, active }: { text: string; active: boolean }) {
  // Auto-expanded while reasoning streams; user can collapse/expand anytime.
  const [open, setOpen] = React.useState(true)
  const [userToggled, setUserToggled] = React.useState(false)
  const bodyRef = React.useRef<HTMLDivElement>(null)

  // Once the answer starts (active=false), auto-collapse unless user touched it.
  React.useEffect(() => {
    if (!active && !userToggled) setOpen(false)
  }, [active, userToggled])

  // Keep the streaming trace scrolled to the latest line.
  React.useEffect(() => {
    if (open && active && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight
    }
  }, [text, open, active])

  return (
    <div className="mb-3 overflow-hidden rounded-xl border border-[var(--chat-border)] bg-[var(--chat-card)]">
      <button
        onClick={() => { setOpen((o) => !o); setUserToggled(true) }}
        className="flex w-full items-center gap-2 px-3.5 py-2.5 text-left"
      >
        <Brain className={`h-3.5 w-3.5 ${active ? "text-[var(--chat-accent)]" : "text-[var(--chat-muted)]"}`} />
        <span className="text-[12px] font-medium text-[var(--chat-secondary)]">
          {active ? "Thinking…" : "Thought process"}
        </span>
        {active && (
          <span className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <motion.span key={i} className="h-1 w-1 rounded-full bg-[var(--chat-accent)]" animate={{ opacity: [0.2, 0.8, 0.2] }} transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.15 }} />
            ))}
          </span>
        )}
        <ChevronRight className={`ml-auto h-3.5 w-3.5 text-[var(--chat-muted)] transition-transform ${open ? "rotate-90" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div
              ref={bodyRef}
              className="max-h-56 overflow-y-auto border-t border-[var(--chat-border)] px-3.5 py-3 scrollbar-thin"
            >
              <p className="whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-[var(--chat-muted)]">
                {text}
                {active && <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-[var(--chat-accent)] align-middle" />}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ─── Tile Gallery ─── */

// Vivid, varied phrasing for "Claude is reading the screenshots with its eyes".
const READING_VERBS = [
  "Looking at",
  "Reading",
  "Studying",
  "Examining",
  "Poring over",
  "Peering at",
  "Taking in",
  "Scanning",
]

function TileGallery({ tiles, loading }: { tiles: TileView[]; loading?: boolean }) {
  const n = tiles.length
  // Seed by the first tile so the verb stays put as more tiles stream in,
  // but differs from answer to answer.
  const verb = READING_VERBS[(tiles[0]?.article_id ?? n) % READING_VERBS.length]
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-5">
      <div className="mb-2.5 flex items-center gap-2">
        <Eye className="h-3.5 w-3.5 text-[var(--chat-warm)]" />
        <span className="text-[12px] font-medium text-[var(--chat-secondary)]">
          {verb} {n} screenshot{n > 1 ? "s" : ""}
        </span>
        {loading && <Loader2 className="h-3 w-3 animate-spin text-[var(--chat-warm)]" />}
      </div>
      <div className="flex gap-2.5 overflow-x-auto pb-1 scrollbar-thin">
        {tiles.map((t, i) => (
          <motion.a key={i} href={tileUrl(t)} target="_blank" rel="noopener noreferrer"
            title="Open full-size tile (right-click to copy or save)"
            initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.08 }}
            className="group/tile relative block shrink-0 overflow-hidden rounded-xl border-2 border-[var(--chat-warm)] border-opacity-20 shadow-lg shadow-[var(--chat-warm)]/5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={tileUrl(t)} alt={`Tile ${t.article_id}/${t.tile_index}/${t.chunk_index}`} className="h-48 w-80 object-cover object-top transition-transform duration-300 group-hover/tile:scale-[1.02]" loading="lazy" />
            <div className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-md bg-black/55 px-2 py-1 text-[10px] font-medium text-white opacity-0 backdrop-blur-sm transition-opacity group-hover/tile:opacity-100">
              <Maximize2 className="h-2.5 w-2.5" /> Open
            </div>
            <div className="flex items-center gap-2 bg-[var(--chat-card)] px-3 py-1.5">
              <span className="relative flex h-1.5 w-1.5"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--chat-warm)] opacity-40" /><span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[var(--chat-warm)]" /></span>
              <span className="font-mono text-[10px] tabular-nums text-[var(--chat-muted)]">{t.article_id}:{t.tile_index}:{t.chunk_index}</span>
            </div>
          </motion.a>
        ))}
      </div>
    </motion.div>
  )
}
