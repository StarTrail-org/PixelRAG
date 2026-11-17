#!/usr/bin/env python3
"""
Generate synthetic query-passage pairs for text-only contrastive learning.

Samples chunks from a local text passage SQLite database, prompts Gemini with
few-shot examples, and writes JSONL records containing the generated query,
answer, supporting span, and source passage metadata.

The output is used as text-warmup data for the visual embedding model.

Prerequisites:
  - SQLite database with text passages (schema: articles + chunks tables)
  - Google Cloud ADC for Vertex AI Gemini access
  - Optionally: OPENAI_API_KEY for fallback when Gemini is rate-limited
  - Optionally: --fewshot-file with seed examples (JSONL with question/answer/text/supporting_span)

Usage:
  python generate_text_query_pairs.py \
      --db-path /path/to/text_baseline.db \
      --num-articles 1000 \
      --output text_query_pairs.jsonl

Ported from Vis-RAG/agent/scripts/contrastive/generate_text_query_pairs.py
"""

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import openai
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "wise-coyote-478119-h0")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

MODEL_PRICING = {
    "gemini-3.1-flash-lite-preview": (1.00, 4.00),
    "gemini-2.0-flash-001": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
MASTER_SEED = 0

SKIP_PATTERNS = [
    r"disambiguation",
    r"^portal:",
    r"^category:",
    r"^template:",
    r"^wikipedia:",
    r"^file:",
    r"^help:",
    r"^talk:",
    r"^module:",
    r"^draft:",
    r"^list of ",
    r"^lists of ",
]
SKIP_CONTENT_PATTERNS = [
    r"\belection\b",
    r"\belections\b",
    r"\breferendum\b",
    r"\bdiscography\b",
    r"\bfilmography\b",
    r"\btrack listing\b",
    r"\bcensus\b",
    r"\bdemographic\b",
    r"^list of .* episodes",
    r"\bepisodes of\b",
]
SKIP_RE = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]
SKIP_CONTENT_RE = [re.compile(p, re.IGNORECASE) for p in SKIP_CONTENT_PATTERNS]

BAD_QUESTION_PATTERNS = [
    r"\baccording to the\b",
    r"\baccording to this\b",
    r"\bvisible\b",
    r"\blisted (here|above|below|in the table)\b",
    r"\bthe table (shows|lists|above|below)\b",
    r"\bin the (following|above) table\b",
    r"\bshown in\b",
    r"\bthis passage\b",
    r"\bthe passage\b",
    r"\bthe article\b",
    r"^what is (the )?listed",
    r"^what (are|is) listed",
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

# Built-in few-shot examples (used when --fewshot-file is not provided)
BUILTIN_FEWSHOT = [
    {
        "question": "In what year was the National Register of Historic Places listing for the Alamo Mission established?",
        "answer": "1966",
        "text": "The Alamo Mission\n\nThe Alamo Mission (Spanish: Misión de Álamo), commonly called the Alamo and originally known as the Misión San Antonio de Valero, is a historic Spanish mission and fortress compound founded in the 18th century by Roman Catholic missionaries in what is now San Antonio, Texas, United States.\n\nIt was the site of the Battle of the Alamo in 1836. The Alamo complex was designated a UNESCO World Heritage Site in 2015. The compound was listed on the National Register of Historic Places in 1966 and designated a National Historic Landmark in 1960.",
        "supporting_span": "The compound was listed on the National Register of Historic Places in 1966 and designated a National Historic Landmark in 1960.",
    },
    {
        "question": "How many species of birds have been recorded in Colombia as of the latest count?",
        "answer": "1,878",
        "text": "Birds of Colombia\n\nThis is a list of the bird species recorded in Colombia. The avifauna of Colombia included a total of 1,878 species as of July 2006, according to Bird Checklists of the World maintained by Avibase. Of these, 67 are endemic, 3 have been introduced by humans, and 80 are rare or accidental.\n\nColombia has more bird species than any other country in the world except for its neighbor, Brazil, and roughly one-fifth of the world's bird species can be found in Colombia.",
        "supporting_span": "The avifauna of Colombia included a total of 1,878 species as of July 2006, according to Bird Checklists of the World maintained by Avibase.",
    },
    {
        "question": "What was the top speed of the Concorde supersonic airliner in Mach number?",
        "answer": "Mach 2.04",
        "text": "Concorde\n\nConcorde is a retired Franco-British supersonic airliner jointly developed and manufactured by Sud Aviation (later Aérospatiale) and the British Aircraft Corporation (BAC).\n\nConcorde had a maximum speed of Mach 2.04 (2,180 km/h at cruise altitude), over twice the speed of sound. It could seat 92 to 128 passengers. First flown in 1969, Concorde entered service in 1976 and operated for 27 years.\n\nConcorde was retired in 2003 following the crash of Air France Flight 4590 and a general downturn in commercial aviation after the September 11 attacks.",
        "supporting_span": "Concorde had a maximum speed of Mach 2.04 (2,180 km/h at cruise altitude), over twice the speed of sound.",
    },
    {
        "question": "Who was the first person to reach the summit of Mount Everest?",
        "answer": "Sir Edmund Hillary and Tenzing Norgay",
        "text": "Mount Everest\n\nMount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Its elevation of 8,848.86 m (29,031 ft 8+1⁄2 in) was most recently established in 2020 by the Chinese and Nepali authorities.\n\nThe first recorded efforts to reach Everest's summit were made by British mountaineers. On 29 May 1953, New Zealander Sir Edmund Hillary and Sherpa Tenzing Norgay became the first climbers confirmed to have reached the summit of Mount Everest.",
        "supporting_span": "On 29 May 1953, New Zealander Sir Edmund Hillary and Sherpa Tenzing Norgay became the first climbers confirmed to have reached the summit of Mount Everest.",
    },
    {
        "question": "In what year did Marie Curie win her second Nobel Prize?",
        "answer": "1911",
        "text": "Marie Curie\n\nMarie Salomea Skłodowska–Curie (7 November 1867 – 4 July 1934) was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.\n\nShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields. She was awarded the Nobel Prize in Physics in 1903 together with her husband Pierre Curie and physicist Henri Becquerel. In 1911, she won the Nobel Prize in Chemistry for her discovery of the elements polonium and radium.",
        "supporting_span": "In 1911, she won the Nobel Prize in Chemistry for her discovery of the elements polonium and radium.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic text query-passage pairs from a text chunk database"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to SQLite database with articles/chunks tables",
    )
    parser.add_argument(
        "--fewshot-file",
        type=Path,
        default=None,
        help="JSONL with few-shot seed examples (question/answer/text/supporting_span). "
        "If not provided, built-in examples are used.",
    )
    parser.add_argument("--output", type=Path, default=Path("text_query_pairs.jsonl"))
    parser.add_argument("--num-articles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--total-batches", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--min-article-chunks", type=int, default=6)
    parser.add_argument("--max-article-chunks", type=int, default=None)
    parser.add_argument("--min-paragraph-words", type=int, default=60)
    parser.add_argument("--openai-fallback-model", type=str, default="gpt-4o-mini")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def infer_title(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def is_bad_title(title: str) -> bool:
    check = (title or "").strip().lower()
    if not check:
        return True
    for pat in SKIP_RE:
        if pat.search(check):
            return True
    for pat in SKIP_CONTENT_RE:
        if pat.search(check):
            return True
    return False


def is_candidate_passage(text: str, n_tokens: int) -> bool:
    if not text or len(text.strip()) < 300:
        return False
    if n_tokens < 80:
        return False
    title = infer_title(text)
    if is_bad_title(title):
        return False
    prefix = normalize_text(text[:400])
    if "may refer to:" in prefix:
        return False
    return True


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def is_list_like_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return True
    bulletish = 0
    pipe_lines = 0
    short_lines = 0
    for line in lines:
        if line.startswith(("-", "*", "|")):
            bulletish += 1
        if "|" in line:
            pipe_lines += 1
        if len(line.split()) <= 6:
            short_lines += 1
    if bulletish >= max(2, len(lines) // 2):
        return True
    if pipe_lines >= max(2, len(lines) // 2):
        return True
    if short_lines >= max(3, len(lines) // 2 + 1):
        return True
    return False


def extract_best_long_paragraph(text: str, min_paragraph_words: int) -> str | None:
    best = None
    best_score = None
    for paragraph in split_paragraphs(text):
        normalized = normalize_text(paragraph)
        word_count = len(paragraph.split())
        sentence_count = len(re.findall(r"[.!?]", paragraph))
        if is_list_like_paragraph(paragraph):
            continue
        if word_count < min_paragraph_words:
            continue
        if sentence_count < 2:
            continue
        if "|" in paragraph:
            continue
        if normalized.startswith(
            ("references", "external links", "see also", "bibliography", "notes")
        ):
            continue
        score = (word_count, sentence_count, len(paragraph))
        if best_score is None or score > best_score:
            best = paragraph
            best_score = score
    return best


def load_fewshot_examples(path: Path | None) -> list[dict]:
    if path is not None:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    return BUILTIN_FEWSHOT


def format_fewshot_block(examples: list[dict]) -> str:
    blocks = []
    for idx, ex in enumerate(examples, 1):
        blocks.append(
            f"""Example {idx}
Passage:
\"\"\"
{ex["text"]}
\"\"\"
Good output:
Q: {ex["question"]}
A: {ex["answer"]}
S: {ex["supporting_span"]}
T: prose"""
        )
    return "\n\n".join(blocks)


def build_prompt(fewshot_block: str, passage: str, focus_paragraph: str) -> str:
    return f"""You are generating a query-evidence pair for training a text retrieval model over Wikipedia passages.

TASK: Given this passage, generate ONE factual question whose answer is explicitly and completely supported by this passage.

STYLE:
- Write natural search-style questions like SimpleQA, not templates.
- Prefer realistic factual questions a user would actually type into a search engine.
- Use diverse question families such as "what is", "who was", "how many", "in which year", "what did", and similar natural forms.
- Prefer facts stated in a long natural prose paragraph, not list items or table cells.

HARD RULES:
1. SELF-CONTAINED: The question must make sense on its own. Name the relevant entities explicitly.
2. EVIDENCE COMPLETE: The answer must be fully supported by the passage alone.
3. DISTINCTIVE: Include enough detail to identify the fact cleanly.
4. VERBATIM SPAN: `S:` must be copied directly from the passage.
5. PROSE-FIRST: Use the long prose paragraph below as the evidence anchor. If it does not support a good question, write `SKIP`.

SKIP if any is true:
- The passage is mostly raw list junk, track listings, or vote counts.
- The long prose paragraph below does not support a clean question-answer pair.
- You cannot write a self-contained question.
- The answer is not fully supported by this passage.
- The supporting span would be truncated or fragmentary.

Write exactly `SKIP` if the passage should be skipped.

source_type: prose | infobox | table | list | other

Few-shot examples:

{fewshot_block}

Now generate for this new passage.

Preferred long prose paragraph:
\"\"\"
{focus_paragraph}
\"\"\"

Passage:
\"\"\"
{passage}
\"\"\"

Output format (4 lines only):
Q: <natural self-contained question>
A: <concise answer>
S: <verbatim supporting span from the passage>
T: <source_type>"""


def parse_model_output(text: str) -> dict | None:
    if not text or text.strip() == "SKIP":
        return None
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        for prefix, key in [
            ("Q:", "query"),
            ("A:", "answer"),
            ("S:", "source_sentence"),
            ("T:", "source_type"),
        ]:
            if line.startswith(prefix):
                fields[key] = line[len(prefix) :].strip()
                break
    if (
        not fields.get("query")
        or not fields.get("answer")
        or not fields.get("source_sentence")
    ):
        return None
    return {
        "query": fields["query"],
        "answer": fields["answer"],
        "source_sentence": fields["source_sentence"],
        "source_type": fields.get("source_type", "prose"),
    }


async def call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    token_counter: dict,
) -> str:
    config = GenerateContentConfig(temperature=0.7, max_output_tokens=512)
    t0 = time.time()
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        ),
    )
    elapsed = time.time() - t0
    usage = resp.usage_metadata
    if usage:
        token_counter["gemini_input"] += getattr(usage, "prompt_token_count", 0)
        token_counter["gemini_output"] += getattr(usage, "candidates_token_count", 0)
        token_counter["gemini_calls"] += 1
        token_counter["gemini_total_time"] += elapsed

    text = ""
    for part in resp.candidates[0].content.parts:
        raw = getattr(part, "text", None)
        if raw and not getattr(part, "thought", False):
            text = raw.strip()
    return text


async def call_openai_fallback(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    token_counter: dict,
) -> str:
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    elapsed = time.time() - t0
    token_counter["openai_input"] += resp.usage.prompt_tokens
    token_counter["openai_output"] += resp.usage.completion_tokens
    token_counter["openai_calls"] += 1
    token_counter["openai_total_time"] += elapsed
    return (resp.choices[0].message.content or "").strip()


def is_natural_question(qa: dict, passage: str) -> bool:
    q = qa.get("query", "")
    a = qa.get("answer", "")
    s = qa.get("source_sentence", "") or ""
    src_type = qa.get("source_type", "prose")

    if src_type != "prose":
        return False
    for pat in BAD_Q_RE:
        if pat.search(q):
            return False
    if a and a[-1] in ("→", "…", "–", "/", "(", ","):
        return False
    if not s:
        return False
    if s.rstrip()[-1:] in ("(", ",", "–", "/", "→", "…"):
        return False
    if len(s.split()) < 10:
        return False
    if normalize_text(s) not in normalize_text(passage):
        return False
    return True


def load_candidate_articles(
    db_path: Path,
    num_articles: int,
    batch_index: int,
    total_batches: int,
    min_article_chunks: int,
    max_article_chunks: int | None,
) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    query = """
        SELECT article_id, n_chunks, text_length
        FROM articles
        WHERE status = 'extracted'
          AND n_chunks >= ?
    """
    params: list[int] = [min_article_chunks]
    if max_article_chunks is not None:
        query += "\n          AND n_chunks <= ?"
        params.append(max_article_chunks)
    rows = list(cur.execute(query, params))
    conn.close()

    candidates = [
        {"article_id": article_id, "n_chunks": n_chunks, "text_length": text_length}
        for article_id, n_chunks, text_length in rows
    ]
    print(f"Total eligible articles: {len(candidates):,}")

    rng = random.Random(MASTER_SEED)
    rng.shuffle(candidates)

    slice_size = len(candidates) // total_batches
    start = batch_index * slice_size
    end = start + slice_size if batch_index < total_batches - 1 else len(candidates)
    pool = candidates[start:end]
    print(
        f"Batch {batch_index}/{total_batches}: articles [{start}:{end}] ({len(pool):,} in pool)"
    )

    selected = pool[:num_articles] if num_articles <= len(pool) else pool
    return selected


def select_chunk_rows(
    db_path: Path,
    articles: list[dict],
    seed: int,
    min_paragraph_words: int,
) -> list[dict]:
    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    work_items = []

    for article in articles:
        usable_chunk_count = max(1, int(article["n_chunks"] * 0.7))
        candidate_indices = list(range(usable_chunk_count))
        rng.shuffle(candidate_indices)

        selected_row = None
        for chunk_index in candidate_indices[: min(len(candidate_indices), 5)]:
            row = cur.execute(
                """
                SELECT text, char_offset, n_tokens
                FROM chunks
                WHERE article_id = ? AND chunk_index = ?
                """,
                (article["article_id"], chunk_index),
            ).fetchone()
            if not row:
                continue
            text, char_offset, n_tokens = row
            if not is_candidate_passage(text, n_tokens):
                continue
            focus_paragraph = extract_best_long_paragraph(
                text, min_paragraph_words=min_paragraph_words
            )
            if not focus_paragraph:
                continue
            title = infer_title(text)
            selected_row = {
                "article_id": article["article_id"],
                "article_n_chunks": article["n_chunks"],
                "article_text_length": article["text_length"],
                "chunk_index": chunk_index,
                "char_offset": char_offset,
                "n_tokens": n_tokens,
                "title_guess": title,
                "focus_paragraph": focus_paragraph,
                "passage": text,
            }
            break

        if selected_row:
            work_items.append(selected_row)

    conn.close()
    return work_items


async def generate_qa(
    client: genai.Client,
    openai_client: openai.AsyncOpenAI | None,
    semaphore: asyncio.Semaphore,
    token_counter: dict,
    model: str,
    openai_fallback_model: str,
    fewshot_block: str,
    work_item: dict,
) -> dict | None:
    prompt = build_prompt(
        fewshot_block, work_item["passage"], work_item["focus_paragraph"]
    )

    async with semaphore:
        for attempt in range(5):
            try:
                text = await call_gemini(client, model, prompt, token_counter)

                qa = parse_model_output(text)
                if not qa:
                    return None
                if not is_natural_question(qa, work_item["passage"]):
                    return None
                return {
                    **qa,
                    "article_id": work_item["article_id"],
                    "article_n_chunks": work_item["article_n_chunks"],
                    "article_text_length": work_item["article_text_length"],
                    "chunk_index": work_item["chunk_index"],
                    "char_offset": work_item["char_offset"],
                    "n_tokens": work_item["n_tokens"],
                    "title_guess": work_item["title_guess"],
                    "focus_paragraph": work_item["focus_paragraph"],
                    "passage": work_item["passage"],
                    "generator_model": model,
                }
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if openai_client is not None:
                        try:
                            print(
                                f"  Gemini rate limited for article {work_item['article_id']}; "
                                f"falling back to {openai_fallback_model}"
                            )
                            text = await call_openai_fallback(
                                openai_client,
                                openai_fallback_model,
                                prompt,
                                token_counter,
                            )
                            qa = parse_model_output(text)
                            if not qa:
                                return None
                            if not is_natural_question(qa, work_item["passage"]):
                                return None
                            return {
                                **qa,
                                "article_id": work_item["article_id"],
                                "article_n_chunks": work_item["article_n_chunks"],
                                "article_text_length": work_item["article_text_length"],
                                "chunk_index": work_item["chunk_index"],
                                "char_offset": work_item["char_offset"],
                                "n_tokens": work_item["n_tokens"],
                                "title_guess": work_item["title_guess"],
                                "focus_paragraph": work_item["focus_paragraph"],
                                "passage": work_item["passage"],
                                "generator_model": openai_fallback_model,
                            }
                        except Exception as openai_error:
                            print(
                                f"  OpenAI fallback failed for article {work_item['article_id']}: "
                                f"{openai_error}"
                            )
                    wait = 2**attempt * 10 + random.uniform(1, 3)
                    print(f"  Rate limited, waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)
                elif attempt < 4:
                    await asyncio.sleep(2)
                else:
                    print(
                        f"  Failed after 5 attempts for article {work_item['article_id']}: {e}"
                    )
                    return None
    return None


async def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fewshot_examples = load_fewshot_examples(args.fewshot_file)
    fewshot_block = format_fewshot_block(fewshot_examples)

    print(
        f"Model: {args.model} via Vertex AI (project={os.environ.get('GOOGLE_CLOUD_PROJECT', 'N/A')})"
    )
    print(
        f"Few-shot examples: {len(fewshot_examples)} ({'from file' if args.fewshot_file else 'built-in'})"
    )
    has_openai_fallback = bool(os.environ.get("OPENAI_API_KEY"))
    print(
        "OpenAI fallback: "
        + (
            f"enabled ({args.openai_fallback_model})"
            if has_openai_fallback
            else "disabled"
        )
    )

    articles = load_candidate_articles(
        db_path=args.db_path,
        num_articles=args.num_articles,
        batch_index=args.batch_index,
        total_batches=args.total_batches,
        min_article_chunks=args.min_article_chunks,
        max_article_chunks=args.max_article_chunks,
    )
    work_items = select_chunk_rows(
        args.db_path, articles, args.seed, args.min_paragraph_words
    )
    print(f"Selected {len(work_items)} candidate chunks from {len(articles)} articles")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    token_counter = {
        "gemini_input": 0,
        "gemini_output": 0,
        "gemini_calls": 0,
        "gemini_total_time": 0.0,
        "openai_input": 0,
        "openai_output": 0,
        "openai_calls": 0,
        "openai_total_time": 0.0,
    }
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    openai_client = (
        openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        if has_openai_fallback
        else None
    )
    tasks = [
        asyncio.ensure_future(
            generate_qa(
                client=client,
                openai_client=openai_client,
                semaphore=semaphore,
                token_counter=token_counter,
                model=args.model,
                openai_fallback_model=args.openai_fallback_model,
                fewshot_block=fewshot_block,
                work_item=item,
            )
        )
        for item in work_items
    ]

    results = []
    t_start = time.time()
    with open(args.output, "w") as outf:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                outf.flush()
                print(
                    f"  [{result['source_type']:7s}] article={result['article_id']} chunk={result['chunk_index']}"
                )
                print(f"    Q: {result['query'][:100]}")
                print(f"    A: {result['answer'][:80]}")

    wall_time = time.time() - t_start
    type_dist = Counter(r["source_type"] for r in results)
    model_dist = Counter(r["generator_model"] for r in results)
    gemini_in_price, gemini_out_price = MODEL_PRICING.get(args.model, (1.0, 4.0))
    openai_in_price, openai_out_price = MODEL_PRICING.get(
        args.openai_fallback_model, (0.15, 0.60)
    )
    gemini_cost = (
        token_counter["gemini_input"] / 1e6 * gemini_in_price
        + token_counter["gemini_output"] / 1e6 * gemini_out_price
    )
    openai_cost = (
        token_counter["openai_input"] / 1e6 * openai_in_price
        + token_counter["openai_output"] / 1e6 * openai_out_price
    )
    cost = gemini_cost + openai_cost

    print(f"\n{'=' * 60}")
    print(f"Wrote {len(results)} text query pairs to {args.output}")
    print(f"Source types: {dict(type_dist)}")
    print(f"Generator models: {dict(model_dist)}")
    print(
        f"Gemini calls:   {token_counter['gemini_calls']} | "
        f"in={token_counter['gemini_input']:,} out={token_counter['gemini_output']:,}"
    )
    print(
        f"OpenAI calls:   {token_counter['openai_calls']} | "
        f"in={token_counter['openai_input']:,} out={token_counter['openai_output']:,}"
    )
    print(f"Wall time:    {wall_time:.1f}s")
    print(f"Est. cost:    ${cost:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
