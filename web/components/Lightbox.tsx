"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronLeft, ChevronRight, X, ExternalLink, ArrowLeft } from "lucide-react"
import { tileUrl } from "@/lib/api"
import type { Hit } from "@/lib/types"
import { Button } from "@/components/ui/button"

interface LightboxProps {
  hit: Hit
  allHits: Hit[]
  onClose: () => void
  onNavigate: (hit: Hit) => void
}

export function Lightbox({ hit, allHits, onClose, onNavigate }: LightboxProps) {
  const currentIndex = allHits.findIndex((h) => h.vector_id === hit.vector_id)
  const hasPrev = currentIndex > 0
  const hasNext = currentIndex < allHits.length - 1

  // Pan and zoom state
  const [scale, setScale] = React.useState(1)
  const [translate, setTranslate] = React.useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = React.useState(false)
  const dragStart = React.useRef({ x: 0, y: 0 })
  const translateStart = React.useRef({ x: 0, y: 0 })

  // Derive article title and full URL
  const raw = hit.url
  const slug = raw.includes("/wiki/") ? raw.split("/wiki/").pop()! : raw
  const title = decodeURIComponent(slug).replace(/_/g, " ") || `Article #${hit.article_id}`
  const articleUrl = raw.startsWith("http") ? raw : `https://en.wikipedia.org/wiki/${encodeURIComponent(slug)}`

  // Lock body scroll while lightbox is open
  React.useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => { document.body.style.overflow = prev }
  }, [])

  // Reset pan/zoom when navigating to a different tile — use key prop on parent instead
  // (handled by passing key={hit.vector_id} to Lightbox in page.tsx)

  // Keyboard navigation
  React.useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
      if (e.key === "ArrowLeft" && hasPrev) onNavigate(allHits[currentIndex - 1])
      if (e.key === "ArrowRight" && hasNext) onNavigate(allHits[currentIndex + 1])
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [onClose, onNavigate, allHits, currentIndex, hasPrev, hasNext])

  // Zoom via mouse wheel (native event to allow preventDefault on non-passive listener)
  const imageAreaRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    const el = imageAreaRef.current
    if (!el) return
    function onWheel(e: WheelEvent) {
      e.preventDefault()
      e.stopPropagation()
      setScale((prev) => Math.max(0.5, Math.min(5, prev - e.deltaY * 0.001)))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [])

  // Pan via drag
  function handleMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return
    setIsDragging(true)
    dragStart.current = { x: e.clientX, y: e.clientY }
    translateStart.current = { ...translate }
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!isDragging) return
    setTranslate({
      x: translateStart.current.x + (e.clientX - dragStart.current.x),
      y: translateStart.current.y + (e.clientY - dragStart.current.y),
    })
  }

  function handleMouseUp() {
    setIsDragging(false)
  }

  return (
    <AnimatePresence>
      <motion.div
        key="lightbox-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex"
        onClick={onClose}
      >
        {/* Dark backdrop */}
        <div className="absolute inset-0 bg-black/80" />

        {/* Content container */}
        <motion.div
          key="lightbox-content"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className="relative z-10 flex w-full"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Image area */}
          <div className="relative flex flex-1 flex-col">
            {/* Back button bar — prominent, always visible */}
            <div className="absolute left-0 right-0 top-0 z-20 flex items-center justify-between px-4 py-3" style={{ background: "linear-gradient(to bottom, rgba(0,0,0,0.55) 0%, transparent 100%)" }}>
              <button
                onClick={onClose}
                className="flex items-center gap-2 rounded-lg bg-white/10 px-3 py-1.5 text-sm font-medium text-white backdrop-blur-sm transition-colors hover:bg-white/20"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-white/70 backdrop-blur-sm transition-colors hover:bg-white/10 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div
              ref={imageAreaRef}
              className="flex flex-1 items-center justify-center overflow-hidden"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ cursor: isDragging ? "grabbing" : "grab" }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
              src={tileUrl(hit)}
              alt={`Tile ${hit.tile_index}`}
              className="max-h-[90vh] max-w-full select-none"
              draggable={false}
              style={{
                transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
              }}
            />
          </div>
          </div>

          {/* Metadata sidebar */}
          <div className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-l border-border bg-card p-6">
            {/* Close button */}
            <div className="flex justify-end">
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <h2 className="font-display text-lg font-semibold">{title}</h2>

            <a
              href={articleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-primary hover:underline"
            >
              Open article <ExternalLink className="h-3 w-3" />
            </a>

            <div className="space-y-3 text-sm">
              <MetaRow label="Score" value={hit.score.toFixed(3)} accent />
              <MetaRow label="Rank" value={`${currentIndex + 1} of ${allHits.length}`} />
              <MetaRow label="Tile index" value={String(hit.tile_index)} />
              <MetaRow label="Tile height" value={`${hit.tile_height}px`} />
              <MetaRow label="Y offset" value={`${hit.y_offset}px`} />
              <MetaRow label="Vector ID" value={String(hit.vector_id)} />
              <MetaRow label="Article ID" value={String(hit.article_id)} />
            </div>

            {/* Navigation buttons */}
            <div className="mt-auto flex gap-2 pt-4">
              <Button
                variant="outline"
                className="flex-1"
                disabled={!hasPrev}
                onClick={() => hasPrev && onNavigate(allHits[currentIndex - 1])}
              >
                <ChevronLeft className="h-4 w-4" />
                Prev
              </Button>
              <Button
                variant="outline"
                className="flex-1"
                disabled={!hasNext}
                onClick={() => hasNext && onNavigate(allHits[currentIndex + 1])}
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

function MetaRow({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent?: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={
          accent
            ? "font-mono font-semibold text-[var(--pixelrag-score)]"
            : "font-mono"
        }
      >
        {value}
      </span>
    </div>
  )
}
