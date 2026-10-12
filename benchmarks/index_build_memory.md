# Index build memory study: measured opt-in prototype

## Status and scope

Inspected on 2026-10-11 and rechecked on 2026-10-12 at `b4fe76301465eaee3d8ac75e0bec54664e53c072`.
GitHub's upstream `main` endpoint returned the same SHA. Reviewed open/closed
PR titles (up to 200 entries), plus PR #140's body and discussion. No newer
streaming builder was found. [PR #155](https://github.com/StarTrail-org/PixelRAG/pull/155)
is merged and already provides IVFPQ. [PR #140](https://github.com/StarTrail-org/PixelRAG/pull/140#issuecomment-5461339035)
is open; its maintainer discussion reports a structured-key dedup slowdown and
asks for a separate resolution. This prototype deliberately preserves the
current packed-key collisions and int64 overflow; it does not resolve that PR.

**Bounded synthetic measurements (2026-10-12).** The focused legacy allocation
failure was reproduced, 47 prototype/neighboring tests passed, and four benchmark
configurations completed on the reviewed code using the existing lockfile. No embedding models were downloaded/trained, paid or embedding
APIs called, or production services touched. Only FAISS index training on local
generated data was performed. This remains a proposal, not production integration.

Production files, CLI defaults, dependencies and lockfile are unchanged.
The experiment lives only in `benchmarks/`; tests import real NumPy, FAISS,
`pixelrag_embed.index`, and the prototype. Missing FAISS causes a collection
error in the new suite, not a silent skip.

## Existing behavior

`_merge_all_shards` loads the compressed `.npz` embeddings even during its shape
scan: `mmap_mode` does not memory-map compressed ZIP members. It then allocates
an `N_raw × d` float32 matrix, metadata arrays, packed keys and unique positions.
If there are duplicates, advanced indexing allocates another retained-vector
matrix before the raw matrix can be freed. `build_ivf` trains and adds in
100,000-row batches *after* this merge. PQ compresses the final index, not those
input matrices. `build_qdrant` also merges first and only then uploads batches.

`FaissBackend` in `serve/src/pixelrag_serve/backends.py` uses FAISS row IDs to
index five `metadata.npz` arrays. Their names, dtypes and first-occurrence order
must remain aligned. This prototype preserves that format; it does not reduce
the serving process's metadata loading footprint.

## Prototype design

1. Sort `shard_*.npz` lexically. Read one shard at a time, casting metadata just
   as the legacy merge does. Compute the same packed signed int64 key:
   `article_id * 100_000_000 + tile_index * 10_000 + chunk_index`.
2. Insert each key into an on-disk SQLite primary-key table with a 2 MiB page
   cache. Keep only the first occurrence. Save per-shard survivor positions to
   disk. No corpus-sized Python set or in-memory metadata table is retained.
3. In that first pass, reservoir-sample surviving float32 vectors, with a fixed
   seed and a configured maximum sample count. Train the same CPU IVFFlat or
   IVFPQ classes and explicitly retain the requested IP/L2 metric.
4. Release the reservoir, reread shards, and add saved survivors in stable order
   in bounded batches. Stream metadata into five raw files, then copy them in
   bounded chunks into `.npy` members of a compatible uncompressed `.npz`.
5. Write index, metadata and summary in a temporary directory on the target
   filesystem. Rename the completed directory into a **new** destination only.
   Exceptions clean temporary output; an existing destination is rejected.

This is a CPU-only experiment, not a reusable backend API. Per-row SQLite and
reservoir work can be slow. Their CPU and disk costs must be measured before
considering integration. Reservoir sampling is uniform without replacement,
but uses a different RNG/ordering from legacy `np.random.choice`; identical
seeds do not imply identical trained centroids or ANN neighbors.

The bound is **not constant total memory**. Live memory includes one full
uncompressed shard (and conversion copies), a training sample or add batch,
SQLite's cache, per-shard selections/keys, FAISS training scratch and the final
in-memory index. The shard-path/signature list scales with the number of files.
The prototype does not stream inside an individual huge compressed archive.
A shard-size contract or a new chunkable input format would be needed for that.

For IVFFlat, stored vectors alone cost about `4 * N_unique * d` bytes, plus
8-byte IDs, centroids, inverted-list capacity slack and other overhead. Eliminating
an extra raw input matrix does not eliminate this footprint. For PQ, codes cost
approximately `ceil(pq_m * pq_nbits / 8)` bytes/vector plus IDs and codebooks;
training/shard buffers may then dominate. These are storage estimates, not RSS
measurements or a prediction for a 30M-vector deployment.

Temporary disk use includes the SQLite database/journal, 8-byte survivor row
positions, 24 bytes/vector of raw metadata, another metadata archive, and the
final index. Disk exhaustion must leave the destination unpublished. No resume
support is implemented. SIGKILL may leave a temporary directory; rename gives
visibility atomicity on the same filesystem, not fsync-backed crash durability.
Concurrent builders targeting the same path are unsupported. Inputs must be
immutable; checks after training and addition re-list the shards and compare
paths, file identities, sizes and mtimes. They detect common changes but are
not a locked snapshot or content-hash guarantee.

## Reproduction

Follow `CLAUDE.md` and CI's core-plus-dev setup. The dev extra supplies NumPy,
FAISS, pytest and local/mock backend testing dependencies without torch. The
locked sync below creates `.venv` without changing project dependencies or the
lockfile. Subsequent project commands intentionally use `--no-sync`; Ruff uses
CI's `uvx` workflow (0.17.0 for these checks).

```sh
uv --version
uv sync --locked --extra dev --python 3.12

# Red: real legacy builder should hit the full-corpus allocation assertion.
PIXELRAG_STUDY_BASELINE=1 uv run --no-sync pytest -q \
  tests/test_index_build_memory.py::test_does_not_allocate_corpus_matrix

# Green and neighboring builder/metadata-consumer coverage.
uv run --no-sync pytest -q tests/test_index_build_memory.py \
  tests/test_ivfpq_index.py tests/test_serve_backends.py \
  tests/test_serve_reconstruct.py tests/test_department_search.py

# Each work directory must be absent. These only generate local random data.
uv run --no-sync python benchmarks/index_build_memory.py \
  --work-dir /tmp/pixelrag-memory-study-flat-review-20261012
uv run --no-sync python benchmarks/index_build_memory.py \
  --work-dir /tmp/pixelrag-memory-study-pq-review-20261012 --pq-m 16

# Repeat at half the corpus volume with fixed shard/training sizes.
uv run --no-sync python benchmarks/index_build_memory.py \
  --work-dir /tmp/pixelrag-memory-study-flat-small-review-20261012 --rows 32768
uv run --no-sync python benchmarks/index_build_memory.py \
  --work-dir /tmp/pixelrag-memory-study-pq-small-review-20261012 --rows 32768 --pq-m 16

uvx ruff check benchmarks/__init__.py benchmarks/streaming_ivf.py \
  benchmarks/index_build_memory.py tests/test_index_build_memory.py
uvx ruff format --check benchmarks/__init__.py \
  benchmarks/streaming_ivf.py benchmarks/index_build_memory.py \
  tests/test_index_build_memory.py
git diff --check
```

Default fixture: 65,536 rows × 256 dimensions, 4,096 rows/shard, normalized
float32 Gaussian vectors, about 5% duplicate keys with different vector/metadata
payloads, seed 2026. Raw vector volume is 64 MiB; generator memory is shard-sized.
The driver caps raw corpus, shard and reservoir matrices at 256 MiB each; this
is not a bound on whole-process RSS or total retained disk output. Default
settings use 8,192 training rows, 64 lists, nprobe 8, a 1,024-row prototype add
batch, one CPU thread and three repeats with alternating builder order. Legacy
keeps its production add batch; a local mock overrides only its CPU thread
count. PQ uses 16 subquantizers and 4 bits in the example above.

Every builder invocation runs in a fresh interpreter. Generation and validation
are outside those processes. `build_wall_seconds` surrounds only the builder;
peak RSS is the process lifetime high-water mark including imports and build,
recorded before any validation. `resource.getrusage(...).ru_maxrss` is **bytes
on macOS, KiB on Linux**. The JSON preserves the raw value and unit, and converts
to MiB by dividing bytes by 1,048,576. Other platforms are rejected. Thread count,
Python/NumPy/FAISS versions and OS are recorded. Repeats share OS file caches;
these are not cold-cache measurements. No performance threshold is asserted.

`results.json` contains per-run peak RSS, wall time, index/metadata/total output
bytes, retained vector count, all five metadata-array equality checks against
the legacy merge, ANN top-10 set overlap between builders, and overlap with an
exact `IndexFlatIP` oracle on 128 independently generated normalized queries.
For corpora below ten vectors, metric names report the actual `k = min(10, N)`.
It also reports overlap when all IVF lists are probed (exact for IVFFlat, still
quantized for PQ). Missing neighbor IDs never count as matches. Validation
intentionally allocates the complete *bounded fixture* outside measured workers.
The small pytest fixtures additionally reconstruct every IVFFlat row to verify
vector/metadata alignment, exercise duplicate keys/collisions/overflow, both
metrics and PQ, inject a failed index write and shard mutations, verify bounded
repeatable sampling/addition, and load output through the real serving backend
with normal and memory-mapped indexes.

## Validation record (2026-10-12)

| Check | Actual result |
| --- | --- |
| `uv --version` | 0.13.0 (Homebrew, aarch64-apple-darwin) |
| `uv sync --locked --extra dev --python 3.12` | Exit 0; CPython 3.12.13, 42 packages installed; lockfile unchanged |
| Baseline-mode focused test | Expected exit 1: **1 failed in 0.08s**, at production `index.py:78`, assertion `builder allocated the full raw input matrix`, shape `(384, 16)` |
| Prototype plus four neighboring suites (command above) | Exit 0: **47 passed in 0.73s, 0 skipped** |
| IVFFlat/PQ at 32,768/65,536 rows | All four commands exit 0; 24 fresh builder processes, 12 paired validations completed |
| Ruff lint and format checks | Exit 0; all lint checks passed, all four Python files formatted |
| `git diff --check` | Exit 0 |

Review regressions were demonstrated before their fixes: adding a shard after
training did not raise (**1 failed in 0.22s**); a four-vector benchmark mislabeled
its overlap as top-10 (**1 failed in 0.09s**). Both pass in the final suite.
The shard checks now also reject same-size/mtime replacements and additions
during the second pass, leaving no published output. Tests verify a 96-row
training bound, 17-row add batches, first-wins sampling, deterministic seeds,
and both IVFFlat/PQ serving compatibility with normal and mmap loading.

Diff review against upstream `b4fe763` found only this benchmark package, the
study/report artifacts and `tests/test_index_build_memory.py`; no production,
retrieval-expansion, rendering, dependency or lockfile changes are included.
No missing-import skips were taken. The full browser/ML suite and Linux runs
were not executed; the selected neighboring suites cover IVF/PQ, serving
backends, reconstruction and department search.

## Measurements

Hardware: Apple M4, 10 logical CPUs, 16 GiB RAM. OS: macOS 26.5.1 arm64.
Python 3.12.13, NumPy 2.4.5, FAISS 1.14.2. Each builder used one FAISS CPU
thread. Each reviewed configuration ran three pairs sequentially, alternating
builder order. The stored reports supersede the pre-review measurements. The
machine was not reserved exclusively for benchmarks; timings are short, share
filesystem caches, and should not be extrapolated to production.

Values below are **median [minimum, maximum]**, across three repeats. Peak RSS
includes process imports and allocator retention, not just live vector buffers.

| Index / raw rows | Legacy peak RSS, MiB | Prototype peak RSS, MiB | Legacy build, seconds | Prototype build, seconds |
| --- | --- | --- | --- | --- |
| IVFFlat / 32,768 | 171.72 [171.72, 171.75] | 107.83 [107.80, 107.84] | 0.217 [0.214, 0.218] | 0.197 [0.195, 0.197] |
| IVFFlat / 65,536 | 268.62 [268.61, 268.77] | 142.80 [142.70, 142.83] | 0.435 [0.428, 0.440] | 0.391 [0.388, 0.397] |
| IVFPQ / 32,768 | 166.11 [165.00, 166.12] | 73.20 [73.17, 73.59] | 0.305 [0.267, 0.317] | 0.259 [0.247, 0.278] |
| IVFPQ / 65,536 | 263.98 [263.97, 264.11] | 74.48 [74.44, 74.58] | 0.487 [0.485, 0.488] | 0.428 [0.426, 0.428] |

At the larger volume, the observed median RSS difference is 125.83 MiB for
IVFFlat and 189.50 MiB for PQ. These are whole-builder comparisons, not a
component-level allocation profile. The legacy duplicate-removal copy matters:
this fixture has 5% duplicate keys, so both a 64 MiB raw matrix and a roughly
60.8 MiB retained matrix can coexist during dedup. A duplicate-free fixture
would avoid that particular copy and need its own measurements.

With fixed shard/sample sizes, doubling rows increased prototype IVFFlat median
RSS by 34.97 MiB and its index file by about 30.64 MiB. PQ prototype median RSS
changed by 1.28 MiB. This is consistent with retaining the final IVFFlat index
while removing corpus-sized input buffers; it is not proof of a constant total
RSS bound or a prediction for 30M vectors. Larger and representative-data runs,
including dedup disk/CPU profiling, remain necessary before adoption.

Both builders retained exactly **31,130 / 62,260 vectors** at the respective
volumes. All five metadata arrays, their order and dtypes matched the legacy
merge in every run. File sizes were repeat-stable and equal between builders
except for summary JSON (the prototype adds experiment/seed fields).

| Index / raw rows | Index bytes, either builder | Metadata bytes, either builder | Total output bytes, legacy / prototype |
| --- | --- | --- | --- |
| IVFFlat / 32,768 | 32,192,347 | 748,416 | 32,941,133 / 32,941,204 |
| IVFFlat / 65,536 | 64,318,507 | 1,495,536 | 65,814,401 / 65,814,472 |
| IVFPQ / 32,768 | 580,692 | 748,416 | 1,329,470 / 1,329,541 |
| IVFPQ / 65,536 | 1,078,772 | 1,495,536 | 2,574,658 / 2,574,729 |

Top-10 set overlap fractions below were identical across all three repeats.
Queries are independent normalized Gaussian vectors, not self-queries.

| Index / raw rows | ANN vs exact, legacy / prototype | ANN between builders | All lists vs exact, legacy / prototype |
| --- | --- | --- | --- |
| IVFFlat / 32,768 | 0.274219 / 0.266406 | 0.227344 | 1.000000 / 1.000000 |
| IVFFlat / 65,536 | 0.278906 / 0.287500 | 0.228125 | 1.000000 / 1.000000 |
| IVFPQ / 32,768 | 0.008594 / 0.005469 | 0.000000 | 0.008594 / 0.005469 |
| IVFPQ / 65,536 | 0.010156 / 0.006250 | 0.001563 | 0.010938 / 0.006250 |

All-list IVFFlat agreement and small-fixture reconstruction tests support row
alignment; approximate rankings are **not equivalent**. Different reservoir
samples/order change training. The deliberately small 16-subquantizer, 4-bit PQ
codes perform poorly on this high-dimensional random fixture; this setting is
not a production recommendation. Neither the slight IVFFlat differences nor PQ
results establish a semantic-retrieval improvement. Matched-sample training and
representative-data recall are separate required experiments.

Raw reports (configuration, platform, each run and validation):

- [IVFFlat, 32,768 rows](results/index-build-memory-20261012-flat-32768.json)
- [IVFFlat, 65,536 rows](results/index-build-memory-20261012-flat-65536.json)
- [IVFPQ, 32,768 rows](results/index-build-memory-20261012-pq-32768.json)
- [IVFPQ, 65,536 rows](results/index-build-memory-20261012-pq-65536.json)

Generated shards, indexes and per-worker logs remain in the four `/tmp` work
directories listed in the reproduction commands; only small JSON reports are
included in the repository. Use new work-directory names to rerun.

## Staged maintainer decision

1. Review the completed red/green checks and four bounded studies, including
   the poor synthetic PQ overlap. Reproduce on another machine and measure a
   duplicate-free fixture and representative data before setting any resource
   budget or performance target. Inspect dedup disk/CPU cost at larger volumes.
2. Decide the dedup identity/bounds separately with PR #140. This study preserves
   current behavior, including known collisions. Compare disk-backed dedup cost
   against alternatives before choosing a production implementation. Consider a
   matched-training-sample experiment to isolate sampling-induced ANN variation.
3. Only after measurements, propose an explicit opt-in production builder with
   supported resource budgets, input validation, snapshot/locking policy, disk
   requirements and publication/recovery guarantees. Keep defaults until reviewed.
4. Consider external-memory FAISS inverted lists/sharded indexes separately if
   the final IVFFlat index itself exceeds RAM. Consider Qdrant streaming in a
   separate change: retain current UUID/payload mapping, first-wins input dedup,
   append/recreate semantics, and address partial remote uploads/retries. No
   Qdrant client is contacted in this study.

Open questions: acceptable disk/CPU tradeoff; maximum shard size; minimum PQ
training volume; dedup migration; metadata format longevity; replacement and
crash-recovery policy; representative real-data recall evaluation. Synthetic
neighbor agreement is not evidence of semantic-retrieval quality.

## Proposed PR (draft only, not submitted)

**Title:** `bench(index): prototype two-pass IVF construction and memory study`

**Body:** IVF/PQ builds currently merge the complete raw float32 corpus before
batched addition. Add an isolated two-pass prototype and bounded synthetic
benchmark using disk-backed legacy dedup, reservoir training, ordered addition
and compatible streamed metadata. Production defaults remain unchanged.

Validation: reproduce the legacy allocation failure; 47 focused/neighboring tests
pass with no skips; Ruff and whitespace checks pass. Four configurations include
three fresh-process pairs each. At 65,536 × 256 rows, median RSS was
268.62 → 142.80 MiB (IVFFlat) and 263.98 → 74.48 MiB (PQ); raw reports include
wall time, counts, metadata, sizes and neighbor overlap. Limits: final index
memory remains, sampling changes ANN results, and random-data PQ overlap is poor.
The prototype requires immutable shards and a new output directory; Linux,
production-scale recall and crash recovery remain unvalidated.

## Maintainer discussion draft (not posted)

Would an opt-in bounded-input-memory builder be useful with this tradeoff?
The reviewed laptop runs remove much of the merge overhead while preserving
first-wins row order and the metadata format. IVFFlat still stores full vectors
in memory. SQLite moves dedup state to disk with a bounded page cache, but adds
per-row SQL work and scratch-space requirements that need measurement at scale.
I suggest keeping this as a benchmark/proposal until those costs are understood.

Two policy decisions should stay separate. This study retains the packed-int64
key semantics, including collisions/overflow, so it does not absorb #140.
Reservoir sampling is reproducible for a fixed seed but changes trained
centroids and ANN neighbors compared with the current sampling method. Is that
acceptable for an explicitly opt-in rebuild, or should a matched-training-sample
experiment be required first? Neither current overlap results nor the small
wall-time differences establish real-data retrieval or production speed gains.

Before integration, I propose agreeing on maximum shard/training sizes, scratch
disk budget, and minimum representative-data recall evidence. The prototype
publishes only complete new directories; replacing an existing index, concurrent
writers, locked input snapshots and crash-durable recovery need their own policy.
If the final IVFFlat index is the RAM bottleneck, external-memory inverted lists
are a separate project. Qdrant streaming/retry semantics should also be reviewed
separately. No production default change is proposed by this contribution.

## Proposed issue message (draft only, not posted)

**Title:** Measure and design bounded-input-memory index construction

`_merge_all_shards` allocates the complete raw float32 corpus before IVF/PQ or
Qdrant batching. PQ (#155) reduces final index storage but does not remove this
input allocation. Could we first review a bounded synthetic memory study and
an opt-in two-pass prototype before changing defaults? The draft compares fresh
processes on peak RSS, build time, output size, metadata order and neighbor
overlap, preserving existing packed-key dedup pending #140. On a 65,536 × 256
synthetic fixture, three-repeat median RSS was 268.62 → 142.80 MiB for IVFFlat
and 263.98 → 74.48 MiB for PQ. Both retained 62,260 rows with identical metadata;
IVFFlat matched exact search when probing all lists, but ANN neighbors differ
and aggressive PQ has poor overlap on random data. These laptop measurements
are not production-scale or semantic-retrieval claims. Decisions needed include
dedup storage cost, shard/sample limits, publication and recovery, and whether
final IVFFlat memory requires a separate external-memory design. This proposal
does not include the #140 collision fix or a production backend rewrite.
