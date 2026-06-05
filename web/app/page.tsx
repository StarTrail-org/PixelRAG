"use client"

import * as React from "react"
import { Suspense } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { search } from "@/lib/api"
import { addHistory } from "@/lib/history"
import { groupHitsByArticle } from "@/lib/types"
import type { Hit, ArticleGroup } from "@/lib/types"
import { SearchBar } from "@/components/SearchBar"
import { SearchControls, type SearchOptions } from "@/components/SearchControls"
import { ModeToggle, type QueryMode } from "@/components/ModeToggle"
import { AboutSection } from "@/components/AboutSection"
import { ResultGroup } from "@/components/ResultGroup"
import { ComparePanel } from "@/components/ComparePanel"
import { Lightbox } from "@/components/Lightbox"
import { AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"

function SearchPageContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const initialQuery = searchParams.get("q") ?? ""

  const [groups, setGroups] = React.useState<ArticleGroup[]>([])
  const [allHits, setAllHits] = React.useState<Hit[]>([])
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [resultMeta, setResultMeta] = React.useState<{
    count: number
    latencyMs: number
  } | null>(null)
  const [selectedHits, setSelectedHits] = React.useState<Set<number>>(new Set())
  const [lightboxHit, setLightboxHit] = React.useState<Hit | null>(null)
  const [hasSearched, setHasSearched] = React.useState(false)
  const [showCompare, setShowCompare] = React.useState(false)
  const [searchOptions, setSearchOptions] = React.useState<SearchOptions>({
    n_docs: 20,
  })
  const [mode, setMode] = React.useState<QueryMode>("search")

  const handleSearchRef = React.useRef<((query: string, image?: string) => void) | null>(null)

  async function handleSearch(query: string, image?: string) {
    // Ask mode → hand off to the agent conversation with the query (and image)
    if (mode === "ask" && (query || image)) {
      const params = new URLSearchParams()
      if (query) params.set("q", query)
      // A base64 image is far too large for a URL query param, so stash it in
      // sessionStorage and let the chat page pick it up — otherwise an
      // image-based question like "who is this man" loses its image entirely.
      if (image) {
        try {
          sessionStorage.setItem("pixelrag:pending-image", image)
          params.set("img", "1")
        } catch {
          // sessionStorage unavailable — text still goes through.
        }
      }
      router.push(`/chat?${params.toString()}`)
      return
    }

    setIsLoading(true)
    setError(null)
    setSelectedHits(new Set())
    setLightboxHit(null)

    const startTime = performance.now()
    try {
      const res = await search({
        queries: [{ text: query || undefined, image: image || undefined }],
        n_docs: searchOptions.n_docs,
        nprobe: searchOptions.nprobe,
        min_tile_height: searchOptions.min_tile_height,
        instruction: searchOptions.instruction,
      })
      const elapsed = performance.now() - startTime
      const hits = res.results[0]?.hits ?? []
      setAllHits(hits)
      setGroups(groupHitsByArticle(hits))
      setResultMeta({ count: hits.length, latencyMs: Math.round(elapsed) })
      setHasSearched(true)

      // Save to search history
      if (query) {
        addHistory(query)
      }

      // Push the results page as its own history entry so the browser back
      // button returns to the clean search page (not whatever preceded it,
      // e.g. /chat). lastRunQuery guards the searchParams effect below from
      // re-running the search we just did.
      if (query) {
        lastRunQuery.current = query
        router.push(`?q=${encodeURIComponent(query)}`, { scroll: false })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed")
      setAllHits([])
      setGroups([])
      setResultMeta(null)
      setHasSearched(true)
    } finally {
      setIsLoading(false)
    }
  }

  React.useEffect(() => {
    handleSearchRef.current = handleSearch
  })

  function resetSearch() {
    setGroups([])
    setAllHits([])
    setResultMeta(null)
    setError(null)
    setHasSearched(false)
    setSelectedHits(new Set())
    setShowCompare(false)
    setLightboxHit(null)
    router.push("/", { scroll: false })
  }

  // The URL's ?q= drives search state, so back/forward work naturally:
  // arriving at ?q= (mount, forward, or back) runs that search; navigating
  // back to "/" clears the results.
  const lastRunQuery = React.useRef<string | null>(null)
  React.useEffect(() => {
    const q = searchParams.get("q")
    if (q && q !== lastRunQuery.current) {
      lastRunQuery.current = q
      handleSearchRef.current?.(q)
    } else if (!q && lastRunQuery.current) {
      lastRunQuery.current = null
      setGroups([])
      setAllHits([])
      setResultMeta(null)
      setError(null)
      setHasSearched(false)
      setSelectedHits(new Set())
      setShowCompare(false)
      setLightboxHit(null)
    }
  }, [searchParams])

  function handleSelectHit(hit: Hit) {
    setSelectedHits((prev) => {
      const next = new Set(prev)
      if (next.has(hit.vector_id)) {
        next.delete(hit.vector_id)
      } else {
        next.add(hit.vector_id)
      }
      return next
    })
  }

  function handleClickHit(hit: Hit) {
    setLightboxHit(hit)
  }

  return (
    <div className="min-h-[calc(100vh-3.5rem)]">
      {/* Hero / Search section — collapses when results are showing */}
      <div className={`flex flex-col items-center gap-4 px-6 transition-all duration-300 ${hasSearched && (groups.length > 0 || error) ? "pb-4 pt-6" : "pb-8 pt-16"}`}>
        {!(hasSearched && (groups.length > 0 || error)) && (
          <div className="relative text-center">
            <div
              className="reveal mb-5 inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground backdrop-blur-sm"
              style={{ animationDelay: "0ms" }}
            >
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-primary" />
              </span>
              28.1M screenshot tiles · indexed
            </div>
            <h1
              className="reveal relative font-display text-5xl font-light leading-[1.05] tracking-tight sm:text-7xl"
              style={{ animationDelay: "80ms" }}
            >
              Search Wikipedia
              <br />
              <span className="gradient-text font-normal italic">by what it looks like.</span>
            </h1>
            <p
              className="reveal relative mx-auto mt-5 max-w-md text-[15px] leading-relaxed text-muted-foreground"
              style={{ animationDelay: "160ms" }}
            >
              Pixel-level visual retrieval over Wikipedia screenshots — search
              with text or an image.
            </p>
          </div>
        )}
        {!(hasSearched && groups.length > 0) && (
          <div className="reveal" style={{ animationDelay: "220ms" }}>
            <ModeToggle mode={mode} onChange={setMode} />
          </div>
        )}
        <div className={`w-full max-w-2xl ${hasSearched && (groups.length > 0 || error) ? "" : "reveal"}`} style={{ animationDelay: "280ms" }}>
          <SearchBar onSearch={handleSearch} onReset={resetSearch} isLoading={isLoading} hasResults={hasSearched && groups.length > 0} defaultValue={initialQuery} mode={mode} />
        </div>
        {mode === "search" && !(hasSearched && groups.length > 0) && <SearchControls options={searchOptions} onChange={setSearchOptions} />}
      </div>

      {/* Narrative landing — only on the initial (pre-search) state */}
      {!hasSearched && !isLoading && (
        <div className="reveal" style={{ animationDelay: "360ms" }}>
          <AboutSection />
        </div>
      )}

      {/* Status bar */}
      {resultMeta && (
        <div className="border-b border-border/50 px-6 pb-4">
          <div className="mx-auto flex max-w-7xl items-center gap-4 text-sm">
            <span className="font-medium tabular-nums">
              {resultMeta.count} result{resultMeta.count !== 1 ? "s" : ""}
            </span>
            <span className="text-muted-foreground/40">·</span>
            <span className="tabular-nums text-muted-foreground">
              {(resultMeta.latencyMs / 1000).toFixed(2)}s
            </span>
            {groups.length > 0 && (
              <>
                <span className="text-muted-foreground/40">·</span>
                <span className="text-muted-foreground">
                  {groups.length} article{groups.length !== 1 ? "s" : ""}
                </span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="mx-auto max-w-7xl px-6 py-8">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
            {error}
          </div>
        </div>
      )}

      {/* Empty state */}
      {hasSearched && !error && allHits.length === 0 && !isLoading && (
        <div className="mx-auto max-w-7xl px-6 py-16 text-center">
          <p className="text-muted-foreground">No results found. Try a different query.</p>
        </div>
      )}

      {/* Loading skeletons */}
      {isLoading && (
        <div className="mx-auto max-w-7xl space-y-6 px-6 py-5">
          {Array.from({ length: 3 }).map((_, gi) => (
            <div key={gi} className="space-y-2.5">
              <div className="h-5 w-48 animate-pulse rounded bg-muted" />
              <div className="flex gap-3 overflow-hidden">
                {Array.from({ length: 4 }).map((_, ci) => (
                  <div
                    key={ci}
                    className="shrink-0 overflow-hidden rounded-lg border border-border bg-card"
                  >
                    <div className="h-48 w-72 animate-pulse bg-muted" />
                    <div className="flex items-center justify-between border-t border-border px-3 py-1.5">
                      <div className="h-3 w-12 animate-pulse rounded bg-muted" />
                      <div className="h-3 w-8 animate-pulse rounded bg-muted" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Results */}
      {!isLoading && groups.length > 0 && (
        <div className="mx-auto max-w-7xl space-y-6 px-6 py-5">
          {groups.map((group) => (
            <ResultGroup
              key={group.article_id}
              group={group}
              selectedHits={selectedHits}
              onSelectHit={handleSelectHit}
              onClickHit={handleClickHit}
            />
          ))}
        </div>
      )}

      {/* Compare floating bar */}
      {selectedHits.size >= 2 && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
          <div className="flex items-center gap-3 rounded-full border border-border bg-card px-5 py-2.5 shadow-lg">
            <span className="text-sm font-medium">
              {selectedHits.size} tiles selected
            </span>
            <Button
              size="sm"
              onClick={() => setShowCompare(true)}
            >
              Compare
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedHits(new Set())}
            >
              Clear
            </Button>
          </div>
        </div>
      )}

      {/* Compare panel */}
      <AnimatePresence>
        {showCompare && selectedHits.size >= 2 && (
          <ComparePanel
            hits={allHits.filter((h) => selectedHits.has(h.vector_id))}
            allHits={allHits}
            onClose={() => setShowCompare(false)}
          />
        )}
      </AnimatePresence>

      {/* Lightbox */}
      {lightboxHit && (
        <Lightbox
          key={lightboxHit.vector_id}
          hit={lightboxHit}
          allHits={allHits}
          onClose={() => setLightboxHit(null)}
          onNavigate={(hit) => setLightboxHit(hit)}
        />
      )}
    </div>
  )
}

export default function SearchPage() {
  return (
    <Suspense>
      <SearchPageContent />
    </Suspense>
  )
}
