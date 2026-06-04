"use client"

import * as React from "react"
import { Search, X, ImagePlus, ArrowLeft, Clock, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getHistory, clearHistory } from "@/lib/history"

// The embedding model is trained on natural-language questions (SimpleQA-style),
// which retrieve the target page most precisely — Ask mode leans on that. The
// search grid deliberately uses bare, slightly ambiguous titles instead: "The
// Starry Night" fans out to Van Gogh + Munch + Millet + the Dutch Sterrennacht,
// a far more striking visual spread than one painting repeated five times.
const EXAMPLE_QUERIES = [
  "The Starry Night",
  "Periodic table",
  "Taj Mahal",
  "The Great Wave off Kanagawa",
  "兵马俑",
]

const ASK_EXAMPLES = [
  "Explain Van Gogh's The Starry Night",
  "How do you read the periodic table?",
  "Tell me about the Taj Mahal",
  "介绍一下兵马俑",
]

interface SearchBarProps {
  onSearch: (query: string, image?: string) => void
  onReset?: () => void
  isLoading: boolean
  hasResults?: boolean
  defaultValue?: string
  mode?: "search" | "ask"
}

export function SearchBar({ onSearch, onReset, isLoading, hasResults, defaultValue = "", mode = "search" }: SearchBarProps) {
  const [query, setQuery] = React.useState(defaultValue)
  const [imageData, setImageData] = React.useState<string | undefined>()
  const [imageName, setImageName] = React.useState<string>("")
  const [isDragOver, setIsDragOver] = React.useState(false)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const inputWrapperRef = React.useRef<HTMLDivElement>(null)
  const [showHistory, setShowHistory] = React.useState(false)
  const [history, setHistory] = React.useState<string[]>([])
  const blurTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  // Refresh history from localStorage whenever dropdown might show
  function refreshHistory() {
    setHistory(getHistory())
  }

  // Cmd+K / Ctrl+K global shortcut to focus search input
  React.useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        const input = inputWrapperRef.current?.querySelector("input")
        input?.focus()
      }
    }
    window.addEventListener("keydown", handleKey)
    return () => window.removeEventListener("keydown", handleKey)
  }, [])

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) return
    setImageName(file.name)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImageData(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim() && !imageData) return
    onSearch(query.trim(), imageData)
  }

  function clearImage() {
    setImageData(undefined)
    setImageName("")
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setIsDragOver(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    setIsDragOver(false)
  }

  function handlePaste(e: React.ClipboardEvent) {
    const file = Array.from(e.clipboardData.items)
      .find((it) => it.type.startsWith("image/"))
      ?.getAsFile()
    if (file) {
      e.preventDefault()
      handleFile(file)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl space-y-3">
      <div
        className={`search-bar-glow flex items-center gap-2 rounded-xl border bg-card p-2 transition-colors ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-border"
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onPaste={handlePaste}
      >
        {hasResults && onReset && (
          <button
            type="button"
            onClick={() => {
              setQuery("")
              setImageData(undefined)
              setImageName("")
              onReset()
            }}
            className="shrink-0 rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="New search"
          >
            <ArrowLeft className="h-4 w-4" />
          </button>
        )}
        {imageData && (
          <div className="relative shrink-0">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageData}
              alt={imageName}
              className="h-10 w-10 rounded-md object-cover"
            />
            <button
              type="button"
              onClick={clearImage}
              className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-destructive-foreground"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        )}
        <div ref={inputWrapperRef} className="relative flex-1">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              refreshHistory()
              setShowHistory(true)
            }}
            onBlur={() => {
              blurTimeoutRef.current = setTimeout(() => setShowHistory(false), 150)
            }}
            placeholder={mode === "ask" ? "Ask anything about Wikipedia..." : "Search Wikipedia with pixels..."}
            className="w-full border-0 bg-transparent pr-10 shadow-none focus-visible:ring-0 focus-visible:border-transparent"
          />
          <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
            ⌘K
          </kbd>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
          }}
        />
        {mode === "search" && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
          >
            <ImagePlus className="h-4 w-4" />
          </Button>
        )}
        <Button type="submit" size="default" disabled={isLoading || (!query.trim() && !imageData)}>
          {mode === "ask" ? <Sparkles className="h-4 w-4" /> : <Search className="h-4 w-4" />}
          {mode === "ask" ? "Ask" : isLoading ? "Searching..." : "Search"}
        </Button>
      </div>
      {/* History dropdown */}
      {showHistory && !query && !hasResults && history.length > 0 && (
        <div className="w-full rounded-lg border border-border bg-card shadow-lg">
          <div className="py-1">
            {history.map((item) => (
              <button
                key={item}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault() // prevent blur
                  if (blurTimeoutRef.current) clearTimeout(blurTimeoutRef.current)
                  setQuery(item)
                  setShowHistory(false)
                  onSearch(item)
                }}
                className="flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-sm text-foreground/90 transition-colors hover:bg-muted"
              >
                <Clock className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate">{item}</span>
              </button>
            ))}
          </div>
          <div className="border-t border-border px-3 py-1.5">
            <button
              type="button"
              onMouseDown={(e) => {
                e.preventDefault()
                if (blurTimeoutRef.current) clearTimeout(blurTimeoutRef.current)
                clearHistory()
                setHistory([])
                setShowHistory(false)
              }}
              className="text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              Clear history
            </button>
          </div>
        </div>
      )}
      {!hasResults && <div className="flex items-start gap-1.5 text-xs text-muted-foreground">
        <span className="mt-1 shrink-0">Try:</span>
        <div className="flex flex-wrap gap-1.5">
          {(mode === "ask" ? ASK_EXAMPLES : EXAMPLE_QUERIES).map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => {
                setQuery(q)
                onSearch(q)
              }}
              className="query-chip rounded-full border border-border px-2.5 py-0.5 hover:border-primary/60 hover:text-foreground hover:bg-primary/5"
            >
              {q}
            </button>
          ))}
        </div>
      </div>}
    </form>
  )
}
