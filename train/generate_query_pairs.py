#!/usr/bin/env python3
"""
Generate synthetic query-chunk pairs for contrastive learning.

Samples informative Wikipedia pages from kiwix tiles, sends screenshot chunks
to Gemini to generate factual Q&A pairs with source_type labels.

Output: JSONL with {query, answer, source_type, subject, source_sentence,
                     chunk_path, url, title, chunk_index, tiles_dir}

Prerequisites:
  - kiwix_tiles directory with Wikipedia screenshot tiles and index.jsonl
  - Google Cloud ADC (gcloud auth application-default login) for Vertex AI,
    OR set GOOGLE_API_KEY for direct Gemini API access

Usage:
  python generate_query_pairs.py \
      --tiles-dir /path/to/kiwix_tiles \
      --num-pages 1000 \
      --output batches/batch_000.jsonl

  # Batched generation (non-overlapping slices):
  python generate_query_pairs.py \
      --tiles-dir /path/to/kiwix_tiles \
      --batch-index 0 --total-batches 100 \
      --num-pages 2000 \
      --output batches/batch_000.jsonl

Ported from Vis-RAG/agent/scripts/contrastive/generate_query_pairs.py
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from PIL import Image

# Vertex AI config — requires gcloud ADC (gcloud auth application-default login)
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "wise-coyote-478119-h0")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

MODEL_PRICING = {
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "gemini-2.5-pro-preview-03-25": (1.25, 10.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-flash-lite-preview": (1.00, 4.00),
    "gemini-2.0-flash-001": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
}
DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_CONCURRENT = 40
IMG_JPEG_QUALITY = 85

# ── Page filtering ───────────────────────────────────────────────────
SKIP_PATTERNS = [
    r"disambiguation",
    r"category:",
    r"template:",
    r"wikipedia:",
    r"portal:",
    r"file:",
    r"help:",
    r"talk:",
    r"module:",
    r"draft:",
    r"_deaths$",
    r"_births$",
]

SKIP_CONTENT_PATTERNS = [
    r"\belection\b",
    r"\belections\b",
    r"\breferendum\b",
    r"\bprimary\b",
    r"\bby-election\b",
    r"\bcouncil election\b",
    r"^list of ",
    r"^lists of ",
    r"\bdiscography\b",
    r"\bfilmography\b",
    r"\btrack listing\b",
    r"_discography$",
    r"_filmography$",
    r"\bseason\b.*\bleague\b",
    r"\bleague season\b",
    r"\bfootball league\b",
    r"\bnba season\b",
    r"\bnfl season\b",
    r"\bcensus\b",
    r"\bdemographic\b",
    r"^list of .* episodes",
    r"\bepisodes of\b",
    r"\bgovernors? of\b",
    r"\bmayors? of\b",
    r"\bprime ministers? of\b",
    r"\bcareer statistics\b",
    r"\bplayer statistics\b",
]

SKIP_RE = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]
SKIP_CONTENT_RE = [re.compile(p, re.IGNORECASE) for p in SKIP_CONTENT_PATTERNS]

BAD_QUESTION_PATTERNS = [
    r"\baccording to the\b",
    r"\baccording to this\b",
    r"\bvisible\b",
    r"\btrack listing\b",
    r"\blisted (here|above|below|in the table)\b",
    r"\bthe table (shows|lists|above|below)\b",
    r"\bin the (following|above) table\b",
    r"\bshown in\b",
    r"^what is (the )?listed",
    r"^what (are|is) listed",
    r"\bpage (shows|lists|includes)\b",
    r"\bthe film\b(?!\s+[A-Z\"])",
    r"\bthe song\b(?!\s+[A-Z\"])",
    r"\bthe album\b(?!\s+[A-Z\"])",
    r"\bthe book\b(?!\s+[A-Z\"])",
    r"\bthe team\b(?!\s+[A-Z\"])",
    r"\bthe show\b(?!\s+[A-Z\"])",
    r"\bthe series\b(?!\s+[A-Z\"])",
    r"\bthe station\b(?!\s+[A-Z\"])",
    r"\bthe school\b(?!\s+[A-Z\"])",
    r"\bthe match\b(?!\s+[A-Z\"])",
    r"\bthe game\b(?!\s+[A-Z\"])",
    r"\bthe competition\b(?!\s+[A-Z\"])",
    r"\bthe episode\b(?!\s+[A-Z\"\d])",
    r"\bthe production\b(?!\s+[A-Z\"])",
    r"\bthe tournament\b(?!\s+[A-Z\d\"])",
    r"^(when|where|who|what|how|why) (was|is|were|did|does|has|have) (it|they|this|that|he|she)\b",
]
BAD_Q_RE = [re.compile(p, re.IGNORECASE) for p in BAD_QUESTION_PATTERNS]


def get_page_chunk_count(entry: dict, tiles_root: Path) -> int:
    cached = entry.get("_chunk_count")
    if cached is not None:
        return cached

    tiles_dir = tiles_root / entry["tiles_dir"]
    chunks_json = tiles_dir / "chunks.json"
    if not chunks_json.exists():
        entry["_chunk_count"] = 0
        return 0

    with open(chunks_json) as f:
        meta = json.load(f)

    chunk_count = len(meta.get("chunks", []))
    entry["_chunk_count"] = chunk_count
    return chunk_count


def is_informative_page(entry: dict) -> bool:
    if entry.get("page_height", 0) < 3000:
        return False
    if entry.get("num_tiles", 0) < 1:
        return False
    if not entry.get("complete", False):
        return False

    title_lower = entry["title"].lower()
    url_lower = entry.get("url", "").lower()
    check = title_lower + " " + url_lower

    for pat in SKIP_RE:
        if pat.search(check):
            return False
    for pat in SKIP_CONTENT_RE:
        if pat.search(check):
            return False

    return True


def is_natural_question(qa: dict) -> bool:
    q = qa.get("query", "")
    for pat in BAD_Q_RE:
        if pat.search(q):
            return False
    a = qa.get("answer", "")
    if a and a[-1] in ("→", "…", "–", "/", "(", ","):
        return False
    s = qa.get("source_sentence", "") or ""
    src_type = qa.get("source_type", "prose")
    if not s:
        return False
    if s.rstrip()[-1:] in ("(", ",", "–", "/", "→", "…"):
        return False
    if src_type == "prose" and len(s.split()) < 10:
        return False
    if src_type in ("infobox", "table") and len(s.split()) < 3:
        return False
    return True


def load_and_sample_pages(
    index_path: Path,
    n: int,
    batch_index: int = 0,
    total_batches: int = 1,
) -> list:
    MASTER_SEED = 0
    print(f"Loading index from {index_path}...")
    candidates: list = []

    with open(index_path) as f:
        for line in f:
            entry = json.loads(line)
            if is_informative_page(entry):
                candidates.append(entry)

    print(f"Total eligible: {len(candidates):,} pages")

    rng = random.Random(MASTER_SEED)
    rng.shuffle(candidates)

    slice_size = len(candidates) // total_batches
    start = batch_index * slice_size
    end = start + slice_size if batch_index < total_batches - 1 else len(candidates)
    pool = candidates[start:end]

    print(
        f"Batch {batch_index}/{total_batches}: pages [{start}:{end}] ({len(pool):,} in pool)"
    )
    selected = pool[:n] if n <= len(pool) else pool
    return selected


def filter_selected_pages_by_chunk_count(
    pages: list[dict],
    tiles_root: Path,
    min_page_chunks: int | None = None,
    max_page_chunks: int | None = None,
) -> list[dict]:
    if min_page_chunks is None and max_page_chunks is None:
        return pages

    kept = []
    for entry in pages:
        chunk_count = get_page_chunk_count(entry, tiles_root)
        if min_page_chunks is not None and chunk_count < min_page_chunks:
            continue
        if max_page_chunks is not None and chunk_count > max_page_chunks:
            continue
        kept.append(entry)
    return kept


def pick_random_chunk(entry: dict, tiles_root: Path) -> tuple:
    tiles_dir = tiles_root / entry["tiles_dir"]
    chunks_json = tiles_dir / "chunks.json"

    if not chunks_json.exists():
        return None, None

    with open(chunks_json) as f:
        meta = json.load(f)

    chunks = meta.get("chunks", [])
    if not chunks:
        return None, None

    usable = chunks[: max(1, int(len(chunks) * 0.7))]
    chunk = random.choice(usable)
    chunk_path = tiles_dir / chunk["file"]

    if not chunk_path.exists():
        return None, None

    return str(chunk_path), chunk["chunk_index"]


def encode_image(path: str) -> str:
    img = Image.open(path)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=IMG_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


QUERY_PROMPT = """\
You are generating a query–evidence pair for training a visual retrieval model over Wikipedia screenshot chunks.

TASK: Given this screenshot chunk, generate ONE factual question whose answer is explicitly and completely visible in this chunk.

━━━ STYLE — write questions like real search queries, not templates ━━━
Good questions sound like something a curious person would actually search for online.
Vary the phrasing — use "how much", "in what year", "which", "who", "where", "what caused", "how long", etc.

Here are examples of the STYLE we want (from SimpleQA benchmark):
  ✓ "How much money, in euros, was the surgeon held responsible for paying in the Olivia Puls case?"
  ✓ "In what city was the 2010 FIFA World Cup opening ceremony held?"
  ✓ "How many days did the 1906 San Francisco earthquake fire burn?"
  ✓ "Which award did Fullmetal Alchemist win at the American Anime Awards in 2007?"
  ✓ "Who was the first Black female judge appointed to the Cook County Circuit Court?"
  ✓ "What was the name of the ship that sank during the 1994 Estonia ferry disaster?"

━━━ EVIDENCE SOURCE — be diverse ━━━
You may draw from ANY visible content: prose text, infobox fields, table cells, image captions, diagrams, or photographs.
Do not always default to infobox — choose whichever source produces the most natural, interesting question.

━━━ HARD RULES ━━━
1. SELF-CONTAINED: The question must be fully understandable on its own — no page title, no external context needed.
   Every entity in the question must be named explicitly.

   ✗ "Who composed the music for the film?"          → missing film name
   ✗ "What is Rideaux's occupation?"                 → surname only, who is Rideaux?
   ✗ "On what date was Lerew awarded the DFC?"       → surname only + unexplained acronym
   ✗ "Which medication is listed as a SARM in the provided table?"  → depends on "the provided table"
   ✗ "Who played the actress in the 2013 film Horns?"  → "the actress" not identified
   ✗ "When was the SMN founded?"                     → unexplained acronym
   ✗ "Who was the spouse of John Houston?"           → too ambiguous, which John Houston?
   ✗ "Which cyclist placed second in the Tempo race?" → missing event/year context
   ✗ "What is listed in the infobox?"               → references page layout
   ✗ "Which mission is shown in the screenshot?"     → depends on visual layout
   ✗ "Which former Cleveland Indians player did the Seattle Mariners sign on December 20?" → missing year
   ✗ "Who was Kesha Rogers' opponent in the general election?"  → missing year and race context

   ✓ "Who composed the music for Once Upon a Time in Hong Kong?"
   ✓ "What is the occupation of Rabbi Shmuel Kamenetsky?"
   ✓ "In what year did photographer Clarence Rideaux found the agency PicturePerfect?"
   ✓ "On what date was RAF pilot Arthur Lerew awarded the Distinguished Flying Cross in World War II?"
   ✓ "Which former Cleveland Indians pitcher did the Seattle Mariners sign on December 20, 2004?"

2. EVIDENCE COMPLETE: The answer must be fully visible in this chunk — not guessed or inferred.
   The source sentence (S:) must be a complete, untruncated span.

3. DISTINCTIVE: Include enough specifics (names, dates, locations, titles) to distinguish this chunk from similar pages.

━━━ ANSWER ━━━
Prefer a single concise entity: name, date, place, number, title, or short phrase.

━━━ SKIP if any is true ━━━
- Raw vote counts, track listings, census tables, or episode lists
- Answer not fully visible or requires external context
- Cannot write a self-contained question naming all entities
- Source sentence is truncated or a fragment

Write exactly: SKIP

source_type: image | table | infobox | prose
subject: science | medicine | history | geography | technology | education | culture | politics | economics | biology | sports | entertainment | other

Output format (5 lines only):
Q: <natural, self-contained question>
A: <concise answer>
S: <verbatim complete span from the chunk>
T: <source_type>
C: <subject>"""


async def generate_qa(
    client: genai.Client,
    model: str,
    chunk_path: str,
    semaphore: asyncio.Semaphore,
    token_counter: dict,
) -> dict | None:
    async with semaphore:
        b64 = encode_image(chunk_path)

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": QUERY_PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ],
            }
        ]
        config = GenerateContentConfig(temperature=0.7, max_output_tokens=1024)

        for attempt in range(5):
            try:
                t0 = time.time()
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                )
                elapsed = time.time() - t0

                usage = resp.usage_metadata
                if usage:
                    token_counter["input"] += getattr(usage, "prompt_token_count", 0)
                    token_counter["output"] += getattr(
                        usage, "candidates_token_count", 0
                    )
                    token_counter["calls"] += 1
                    token_counter["total_time"] += elapsed

                text = ""
                for p in resp.candidates[0].content.parts:
                    raw = getattr(p, "text", None)
                    if raw and not getattr(p, "thought", False):
                        text = raw.strip()

                if not text or text.strip() == "SKIP":
                    return None

                fields = {}
                for line in text.split("\n"):
                    line = line.strip()
                    for prefix, key in [
                        ("Q:", "query"),
                        ("A:", "answer"),
                        ("S:", "source_sentence"),
                        ("T:", "source_type"),
                        ("C:", "subject"),
                    ]:
                        if line.startswith(prefix):
                            fields[key] = line[len(prefix) :].strip()
                            break

                q = fields.get("query")
                a = fields.get("answer")
                if not q or not a or len(a) < 2:
                    return None

                qa = {
                    "query": q,
                    "answer": a,
                    "source_sentence": fields.get("source_sentence"),
                    "source_type": fields.get("source_type", "prose"),
                    "subject": fields.get("subject", "other"),
                }

                if not is_natural_question(qa):
                    return None

                return qa

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 2**attempt * 15 + random.uniform(1, 5)
                    print(f"  Rate limited, waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)
                elif attempt < 4:
                    await asyncio.sleep(2)
                else:
                    print(f"  Failed after 5 attempts: {e}")
                    return None

    return None


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic query-chunk pairs from Wikipedia screenshot tiles"
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        required=True,
        help="Root directory of kiwix_tiles (containing index.jsonl)",
    )
    parser.add_argument("--num-pages", type=int, default=10)
    parser.add_argument("--output", type=str, default="query_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini model (e.g. gemini-3.1-pro-preview, gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Which non-overlapping batch to process (0-based)",
    )
    parser.add_argument(
        "--total-batches",
        type=int,
        default=1,
        help="Total number of batches the candidate pool is divided into",
    )
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--postfilter-min-page-chunks", type=int, default=None)
    parser.add_argument("--postfilter-max-page-chunks", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tiles_root = args.tiles_dir
    index_path = tiles_root / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"index.jsonl not found at {index_path}")

    model = args.model
    price_input, price_output = MODEL_PRICING.get(model, (1.25, 10.00))

    print(
        f"Model: {model} via Vertex AI (project={os.environ.get('GOOGLE_CLOUD_PROJECT', 'N/A')})"
    )

    pages = load_and_sample_pages(
        index_path, args.num_pages, args.batch_index, args.total_batches
    )
    pages_before_postfilter = len(pages)
    pages = filter_selected_pages_by_chunk_count(
        pages,
        tiles_root,
        min_page_chunks=args.postfilter_min_page_chunks,
        max_page_chunks=args.postfilter_max_page_chunks,
    )
    if (
        args.postfilter_min_page_chunks is not None
        or args.postfilter_max_page_chunks is not None
    ):
        chunk_filter = []
        if args.postfilter_min_page_chunks is not None:
            chunk_filter.append(f"chunks>={args.postfilter_min_page_chunks}")
        if args.postfilter_max_page_chunks is not None:
            chunk_filter.append(f"chunks<={args.postfilter_max_page_chunks}")
        print(
            "Post-slice page filter: "
            + ", ".join(chunk_filter)
            + f" -> kept {len(pages)}/{pages_before_postfilter} pages"
        )
    print(f"Selected {len(pages)} pages\n")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    token_counter = {"input": 0, "output": 0, "calls": 0, "total_time": 0.0}
    results = []

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    async def process_one(page, chunk_path, chunk_idx):
        qa = await generate_qa(client, model, chunk_path, semaphore, token_counter)
        if not qa:
            return None
        rel_path = str(Path(chunk_path).relative_to(tiles_root))
        return {
            **qa,
            "chunk_path": rel_path,
            "url": page["url"],
            "title": page["title"],
            "chunk_index": chunk_idx,
            "tiles_dir": page["tiles_dir"],
        }

    work_items = []
    for page in pages:
        chunk_path, chunk_idx = pick_random_chunk(page, tiles_root)
        if chunk_path is None:
            continue
        work_items.append((page, chunk_path, chunk_idx))

    print(f"Generating Q&A for {len(work_items)} chunks...\n")
    t_start = time.time()

    tasks = [asyncio.ensure_future(process_one(p, cp, ci)) for p, cp, ci in work_items]

    with open(args.output, "w") as outf:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                outf.flush()
                st = result.get("source_type", "?")
                subj = result.get("subject", "?")
                print(f"  [{st:8s}|{subj:14s}] {result['title']}")
                print(f"    Q: {result['query'][:90]}")
                print(f"    A: {result['answer'][:90]}")
                print()

    wall_time = time.time() - t_start

    type_dist = Counter(r["source_type"] for r in results)
    subj_dist = Counter(r["subject"] for r in results)

    in_tok = token_counter["input"]
    out_tok = token_counter["output"]
    calls = token_counter["calls"]
    cost = (in_tok / 1e6 * price_input) + (out_tok / 1e6 * price_output)

    print(f"\n{'=' * 60}")
    print(f"Wrote {len(results)} Q&A pairs to {args.output}")
    print(f"Source types: {dict(type_dist)}")
    print(f"Subjects:     {dict(subj_dist)}")
    print(f"\n--- Throughput & Cost ({model}) ---")
    print(f"  Calls:        {calls}")
    print(f"  Input tokens: {in_tok:,}  ({in_tok / max(calls, 1):.0f} avg/call)")
    print(f"  Output tokens:{out_tok:,}  ({out_tok / max(calls, 1):.0f} avg/call)")
    print(f"  Wall time:    {wall_time:.1f}s  ({wall_time / max(calls, 1):.1f}s/call)")
    print(f"  Est. cost:    ${cost:.4f} for {calls} calls")
    print(f"  Per 10 pairs: ${cost / max(calls, 1) * 10:.4f}")
    print(f"  Per 1K pairs: ${cost / max(calls, 1) * 1000:.2f}")
    print(f"  Per 50K pairs:${cost / max(calls, 1) * 50000:.0f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
