# Screenshot Throughput Optimization

Batch screenshot capture of Full English Wikipedia (18.7M articles, ~30M tiles)
with headless Chrome on AMD EPYC 7763 (128 cores, 995GB RAM).

## 1. Results

| Metric | Value |
|--------|-------|
| **E2E throughput** | **109 t/s** (JPEG written to disk) |
| **Capture throughput** | 98 t/s (raw BGRA to /dev/shm) |
| Tile size | 875 × 8192 px |
| Output format | JPEG q50, avg 305 KB/tile |
| Storage (30M tiles) | **8.5 TB** |
| Processing time (30M tiles) | **77 hours single machine, < 1 day on 4 machines** |
| Correctness | 100% pixel-verified against ground truth |

Best config: 48 Chrome workers, work-stealing queue, JPEG file output via
`rawFilePath` with `.jpg` extension (Chrome encodes JPEG in ThreadPool, writes
directly to disk — no base64, no websocket transfer, no external compression).

## 2. Architecture

```
kiwix-serve (ZIM)  →  48 Chrome workers  →  JPEG files on disk
    localhost:9461       875×8192 viewport     305 KB/tile avg
                         CDP websocket         109 tiles/s
```

Each Chrome process: navigate → fonts.ready + rAF → `Page.captureScreenshot`
with `rawFilePath=/path/tile.jpg` → Chrome encodes JPEG in async ThreadPool →
writes to disk → returns immediately → next article.

48 workers share the same `user-data-dir` (HTTP cache). Articles distributed
via asyncio work-stealing queue. `Page.frameStoppedLoading` event ensures
navigation is complete before capture.

## 3. Pipeline Bottleneck Analysis

The system is a two-stage pipeline analyzed via closed queueing model (Little's
Law + USL contention curve).

```
Stage         Workers  Per-op    Capacity     Bottleneck?
────────────────────────────────────────────────────────
Nav           48       186ms     258 pg/s     No
Capture       48       321ms     150 t/s      ← YES
Compress      async    ~10ms     ∞            No (ThreadPool)
Disk write    async    ~1ms      ∞            No

Steady-state: C/T_c(C) = 48/321ms ≈ 150 t/s theoretical
Actual (200 articles): 95 t/s (pipeline startup/drain bubble)
Actual (500 articles): 109 t/s (less bubble)
```

**Capture is the bottleneck.** Nav latency does not affect steady-state
throughput — verified by reducing nav from 186ms to 92ms with no throughput
change.

Per-capture breakdown (48 concurrent, measured via Chromium instrumentation):

| Component | 1 worker | 48 concurrent | Notes |
|-----------|----------|---------------|-------|
| ForceRedraw IPC | 95ms | 181ms | 8-hop async roundtrip |
| DrawRenderPass | 57ms | 62ms | Composite 136 quads |
| CopyDrawnRenderPass | 18ms | 46ms | memcpy 28MB |
| JPEG encode | 0 | ~10ms | ThreadPool, async |
| **Total** | **170ms** | **321ms** | |

IPC dominates at 48c (56% of capture time). This is OS scheduling overhead:
48 Chrome processes × 5 threads = 240 threads on 128 cores.

Contention curve `C/T_c(C)` converges at ~125-130 t/s regardless of C:

| Concurrent | T_c | C/T_c |
|------------|-----|-------|
| 24 | 200ms | 120 |
| 32 | 260ms | 123 |
| 48 | 321ms | **150** |
| 64 | 500ms | 128 |

## 4. Optimizations and Ablation

Each optimization measured on 200 articles, 100% correct, same hardware.

| # | Optimization | Throughput | Δ | Key insight |
|---|-------------|-----------|---|-------------|
| 0 | Baseline (Playwright, sleep 30ms) | 20 t/s | — | Node.js IPC layer |
| 1 | Direct CDP websocket | 23 t/s | +14% | Bypass Playwright |
| 2 | + `fonts.ready` + eager images | 28 t/s | +22% | Event-driven, no polling |
| 3 | + `rawFilePath` (Chromium patch) | 33 t/s | +18% | Bypass PNG/JPEG encode mutex |
| 4 | + Multi-worker (48w sequential) | 79 t/s | +140% | Linear scaling to ~48w |
| 5 | + Phased strategy (semaphore) | 96 t/s | +22% | Reduce capture contention |
| 6 | + Work-stealing queue | 98 t/s | +2% | Better load balancing |
| 7 | + `.jpg` rawFilePath (JPEG in ThreadPool) | 95 t/s | −3% | E2E with compression |
| 8 | + 500 articles (steady-state) | **109 t/s** | +15% | Amortize pipeline bubble |

Cumulative: 20 → 109 t/s = **5.5× improvement**.

### Chromium patches (5 files, 285 lines)

| Patch | Impact | Description |
|-------|--------|-------------|
| `rawFilePath` | +18% | Async raw BGRA write to /dev/shm via ThreadPool |
| `.jpg` auto-detect | e2e JPEG | JPEG encode in ThreadPool when path ends with .jpg |
| `directClip` | per-tile parallel | CopyFromSurface(src_rect) without emulation change |
| `skipRedraw` | −5ms latency | ForceRedrawWithCallback → CopyFromSurface |

## 5. Approaches That Did Not Work

| Approach | Result | Why it failed |
|----------|--------|---------------|
| `--in-process-gpu` | 120 t/s, 90% correct | Compositor surface race: about:blank captured instead of real page. ForceRedraw callback fires before compositor activates new frame. |
| `--single-process` | 168 t/s, 74% correct | Renderer thread contention across tabs in shared process |
| Two-tab pipelining | 8 t/s | Chrome UI thread serializes ForceRedraw across tabs in same process |
| `directClip` without ForceRedraw | 93% correct | Compositor frame stale without explicit redraw |
| Per-user-data-dir Chrome | 8 t/s | Each process starts with cold HTTP cache → thundering herd on kiwix |
| SwiftShader GPU compositor | −17% | CPU-based Vulkan slower than Chrome's software rasterizer |
| GPU on lab machines (H200/B200) | Blocked | `/dev/dri` permissions, no nvidia-container-toolkit |
| External ProcessPoolExecutor JPEG | 40 t/s | Cross-process IPC overhead; pool workers starved during capture |
| Firefox (Playwright) | 2.6× slower | Same IPC overhead, different engine |
| CEF OSR | ~12 t/s | Xvfb + 2.6s/page overhead |
| Servo | N/A | Not production-ready (stub package) |
| Skip `fonts.ready` | +4% only | Nav is not the bottleneck |
| 4096px tiles | Higher t/s but lower mpix/s | Fixed ForceRedraw overhead per tile |

### `--in-process-gpu` deep dive

Eliminates GPU process IPC → per-capture drops from 321ms to 175ms → 120 t/s.
But 5-10% of captures get about:blank content (correct dimensions, wrong pixels).

Root cause: ForceRedraw's presentation feedback fires after `SubmitCompositorFrame`
but before viz activates the new surface. `CopyFromSurface` reads the old surface.
About:blank renders at 875×8192 (same as real page due to persistent viewport
emulation), making detection impossible from dimensions alone.

Tried: SwapPromise (fires earlier), RequestRepaintOnNewSurface (new LocalSurfaceId),
bitmap dimension retry, pixel content check, `Page.frameNavigated` (reliable but
times out at 48w igpu), `Page.frameStoppedLoading` (reliable but fires for
sub-frames). None achieved 100% correct at 48 workers.

## Reproducing

**Measure with the bench harness, not a hand-rolled loop.** `tiles_per_s` here is
**capture-only**: the bench's timer starts *after* `strategy.setup()` brings the Chrome
workers up. At 48 workers, that startup is ~49s of serial Chrome launches — it is setup
cost, not throughput, so it must not be counted. A naive end-to-end loop that includes the
48-worker startup reports ~13 t/s (the startup tax), which is not the capture rate.
Re-measured on the reference box (EPYC 7763, 128c) with the harness below: **130 t/s
capture-only** (200 maxi-ZIM pages, 48 workers, `fmt="raw"`).

```python
from pixelrag_render.strategies.cdp_phased import CDPPhasedStrategy
from pixelrag_render.bench import Bench

# Benchmark (with GT pixel verification)
bench = Bench(zim_path="...", chrome_path="...", output_dir="./results",
              kiwix_url="http://localhost:9461")
strategy = CDPPhasedStrategy(chrome_path="...", n_workers=48, capture_limit=48, fmt="raw")
result = await bench.run(strategy)  # {"tiles_per_s": 98, "correct_pct": 100, ...}

# Production (JPEG files on disk)
# Use rawFilePath with .jpg extension for Chrome-side JPEG encode:
await conn.cdp("Page.captureScreenshot", {
    "rawFilePath": "/output/tile.jpg",  # .jpg → JPEG encode in ThreadPool
    "fromSurface": True, "optimizeForSpeed": True,
    "clip": {"x": 0, "y": 0, "width": 875, "height": 8192, "scale": 1}
})
```

Requires custom Chromium build. Patch + build instructions: `chromium/README.md`.
