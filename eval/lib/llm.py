"""LLM client and prompt building for SimpleQA evaluation.

Supports:
- Google Gemini (Vertex AI and standard API)
- OpenAI-compatible APIs (vLLM, etc.)
"""

import asyncio
import base64
import logging
import os

# Try to import Google GenAI for Gemini support
try:
    import google.genai as genai
    from google.genai.types import (
        GenerateContentConfig,
        Part,
        Blob,
        HttpOptions,
        Content,
    )

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GenerateContentConfig = None
    Part = None
    Blob = None
    HttpOptions = None
    Content = None

from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# System Prompts
SYSTEM_PROMPT_NAIVE = """You are a research assistant who answers questions.
Use <think></think> tags to show your reasoning if needed.
Answer the question directly and concisely.
"""

SYSTEM_PROMPT_EVIDENCE_QA = """You are a research assistant who answers questions based on provided evidence.
Use <think></think> tags to show your reasoning if needed.
Answer the question directly and concisely based ONLY on the provided evidence.
"""

SYSTEM_PROMPT_SCREENSHOT = SYSTEM_PROMPT_EVIDENCE_QA

SYSTEM_PROMPT_TEXT_RAG = SYSTEM_PROMPT_EVIDENCE_QA

SYSTEM_PROMPT_VECTOR = SYSTEM_PROMPT_EVIDENCE_QA

SYSTEM_PROMPT_SHORT_ANSWER = """Answer the question with as few words as possible. Give only the answer, no explanation.
"""

SYSTEM_PROMPT_REACT = """You are a research assistant who answers questions using a search tool.
You will be provided with retrieved Wikipedia screenshot tiles as evidence.

IMPORTANT: Try your best to answer with the evidence you have. Only search again if the evidence is clearly about a WRONG topic and does not contain the answer at all.

To search for different evidence, output ONLY: <search>your refined search query</search>
Otherwise, answer the question directly and concisely.

Rules:
- READ the evidence images carefully — the answer is often there even if not obvious.
- If the images show the relevant Wikipedia article, answer from them. Do NOT search again.
- Only use <search> if the retrieved tiles are about a completely unrelated topic.
- Do NOT repeat the same search query — use different keywords.
- Use <think></think> tags to show your reasoning if needed.
"""

SYSTEM_PROMPT_REACT_V2 = """You are a research assistant who answers questions using a search tool.
You will be provided with retrieved Wikipedia screenshot tiles as evidence.

You have two actions:
1. **Answer**: If you can find or infer the answer from the evidence, respond with your answer directly.
2. **Search**: If the evidence does NOT contain the answer, output: <search>new search query</search>

CRITICAL rules:
- ALWAYS try to answer first. Only search if the evidence is about the WRONG topic entirely.
- Each search query MUST use DIFFERENT keywords than all previous queries. Think about synonyms, related entities, or the answer's broader topic.
- If you've already searched 2+ times without finding the answer, make your BEST GUESS based on whatever partial evidence you have. Do not give up.
- Never output an empty answer. If unsure, state your best guess with a caveat.
- Use <think></think> tags for reasoning.
"""

SYSTEM_PROMPT_REACT_MULTIHOP = """You are a research assistant who answers multi-hop questions using a search tool.
You will be provided with retrieved Wikipedia screenshot tiles as evidence.

Multi-hop questions require information from MULTIPLE Wikipedia pages. For example:
- "Where did X's father die?" → First find who X's father is, then search for the father's death place.
- "Which film came out first, A or B?" → Search for film A's release date, then film B's release date.

Strategy:
1. Read the evidence carefully. Extract any INTERMEDIATE facts (names, dates, locations) that help answer the question.
2. If you found an intermediate fact but still need more info, search for the next entity: <search>entity name topic</search>
3. Only give your final answer when you have ALL the pieces needed.

Rules:
- For multi-hop questions, you will usually need 2-3 searches. This is EXPECTED — do not try to answer with just the first search.
- In <think> tags, ALWAYS record: the specific facts you found (names, dates, places) so you don't lose them.
- Extract specific entity names from evidence tiles to use as search queries.
- Each search query MUST use DIFFERENT keywords. Be specific: use full names, dates, or titles you found.
- When you have enough info, give a concise final answer.
"""

SYSTEM_PROMPT_PIXEL_QUERY = """You are a research assistant who answers questions based on retrieved visual evidence.
The first image contains the question you need to answer.
The remaining images are retrieved evidence that may contain the answer.
Read the question from the first image, then use the evidence images to answer it.
Use <think></think> tags to show your reasoning if needed.
Answer the question directly and concisely.
"""

SYSTEM_PROMPT_MULTIMODAL_QUERY = """You are a research assistant who answers questions based on retrieved visual evidence.
You will receive: (1) a text question, (2) a query image, and (3) retrieved Wikipedia evidence images.
Use the query image and evidence images to answer the question.
Use <think></think> tags to show your reasoning if needed.
Answer the question directly and concisely.
"""


def _build_fewshot_turns(demos: list[dict], encode_image_fn) -> list[dict]:
    """Build a list of (user, assistant) message turns for in-context few-shot.

    Each demo becomes: user={Q text + demo image} → assistant={answer}. The
    chat-tuned model treats these as prior conversation turns rather than
    mixing them with the current question's evidence — this is the canonical
    few-shot format for instruction-tuned chat models.
    """
    turns: list[dict] = []
    for demo in demos:
        user_content: list[dict] = [
            {"type": "text", "text": f"Question: {demo['question']}"},
        ]
        img_path = demo.get("image_path")
        if img_path and encode_image_fn and os.path.exists(img_path):
            try:
                b64 = encode_image_fn(img_path)
                if b64:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to encode few-shot image {img_path}: {e}")
        turns.append({"role": "user", "content": user_content})
        turns.append({"role": "assistant", "content": demo["answer"]})
    return turns


def _tile_image_b64(img, encode_image_fn):
    """Base64 PNG for a retrieved tile. The serve returns a tile either as a local
    file path (a tile corpus is mounted) or as inline base64 bytes (the reader has
    no local tiles — public-API / on-demand-render modes). Handle both so the reader
    actually sees the retrieved evidence instead of silently dropping it."""
    if not isinstance(img, str):
        return None
    if os.path.exists(img):
        return encode_image_fn(img) if encode_image_fn else None
    # No file at this path -> the serve returned the tile inline as base64.
    return img if len(img) > 256 else None


def build_messages(
    query: str,
    retrieval_result: RetrievalResult,
    encode_image_fn=None,
    additional_instructions: str | None = None,
    few_shot_demos: list[dict] | None = None,
) -> list[dict]:
    """Build messages for LLM based on retrieval result.

    When ``retrieval_result.pixel_query_path`` is set the query is sent as an
    image. Two modes:
    - **Multimodal** (retrieval_type contains "multimodal"): text question + query image + retrieved tiles.
    - **Pixel query** (rendered question as image): first image = question, then retrieved tiles.
    """
    # ---- Multimodal / pixel-query mode: text + raw species/landmark photo + retrieved tiles ----
    # query_image_path = raw species/landmark photo (for generation, always).
    # pixel_query_path = rendered card or raw photo (for retrieval only; ignored here).
    # Falls back to pixel_query_path if query_image_path is not set (backward compat).
    gen_image_path = (
        retrieval_result.query_image_path or retrieval_result.pixel_query_path
    )
    if gen_image_path and encode_image_fn:
        system_prompt = SYSTEM_PROMPT_MULTIMODAL_QUERY
        # Decide evidence_note based on what retrieval actually returned. Three cases:
        #   (a) retrieved images (screenshot retrieval) — evidence is image tiles after the query
        #   (b) retrieved text (text retrieval) — evidence is rendered as text after the query
        #   (c) no retrieval — query image only
        # Until 2026-04-29 this branch silently dropped retrieval_result.text whenever the
        # query image was set, turning every "EVQA + text retrieval" cell into an effective
        # naive run. Fixed by adding the text-passages block alongside the multimodal preamble.
        if retrieval_result.images:
            evidence_note = "The first image is the query image. The following images are retrieved Wikipedia evidence. Answer the question based on the evidence."
        elif retrieval_result.text:
            evidence_note = "The image is the query image. Below is retrieved Wikipedia evidence (text). Answer the question based on the evidence and the image."
        else:
            evidence_note = "The first image is the query image. Answer the question based on the image (no additional evidence was retrieved)."
        text_parts = [
            f"Question: {query}",
            "",
            evidence_note,
        ]
        if retrieval_result.text:
            # Option 1: no URL header in multimodal branch either. Reader gets the
            # chunks and the query image, no metadata leak.
            text_parts.extend(
                [
                    "",
                    retrieval_result.text,
                ]
            )
        if additional_instructions:
            text_parts.append("")
            text_parts.append(additional_instructions)
        user_content: list[dict] = [
            {"type": "text", "text": "\n".join(text_parts)},
        ]

        # Add raw species/landmark photo
        if os.path.exists(gen_image_path):
            try:
                img_base64 = encode_image_fn(gen_image_path)
                if img_base64:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to encode query image {gen_image_path}: {e}")
                user_content.append(
                    {"type": "text", "text": f"(Image unavailable) Query: {query}"}
                )
        else:
            logger.warning(f"Query image not found: {gen_image_path}")
            user_content.append({"type": "text", "text": f"Query: {query}"})

        # Add retrieved tiles
        if retrieval_result.images:
            for img_path, score in retrieval_result.images:
                try:
                    img_base64 = _tile_image_b64(img_path, encode_image_fn)
                    if img_base64:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to encode tile: {e}")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ---- Original modes --------------------------------------------------
    # Select system prompt based on retrieval type
    if retrieval_result.base64_image:
        system_prompt = SYSTEM_PROMPT_SCREENSHOT
        user_content = [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{retrieval_result.base64_image}"
                },
            },
        ]
    elif (
        retrieval_result.retrieval_type == "text_api+rendered"
        and retrieval_result.images
        and encode_image_fn
    ):
        # Text retrieval rendered as images. Mirror the text-RAG framing so
        # evidence comes first and the reader sees an explicit "Question:"
        # suffix — same structure as the text→text branch below, only the
        # evidence modality differs.
        system_prompt = SYSTEM_PROMPT_TEXT_RAG
        user_content = []
        for img_path, score in retrieval_result.images:
            try:
                img_base64 = _tile_image_b64(img_path, encode_image_fn)
                if img_base64:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to encode tile: {e}")
        user_content.append({"type": "text", "text": f"Question: {query}"})
    elif retrieval_result.images and encode_image_fn:
        system_prompt = SYSTEM_PROMPT_VECTOR
        user_content = [{"type": "text", "text": query}]
        # Encode and add retrieved images
        for img_path, score in retrieval_result.images:
            try:
                img_base64 = _tile_image_b64(img_path, encode_image_fn)
                if img_base64:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to encode tile: {e}")
    elif retrieval_result.text:
        system_prompt = SYSTEM_PROMPT_TEXT_RAG
        # Option 1 (2026-04-29): no `Context from {urls}:` wrapper. URL leak gave
        # text retrieval an unfair advantage on entity-answering tasks. Reader sees
        # only the retrieved chunks and the question. URL still recorded in the
        # JSONL via retrieval_result.source_url for logging/grading.
        user_content = f"""{retrieval_result.text}

Question: {query}"""
    else:
        # Naive mode
        system_prompt = SYSTEM_PROMPT_NAIVE
        user_content = query

    # Append additional instructions (e.g. short-answer prompt for EM-eval tasks)
    if additional_instructions:
        if isinstance(user_content, str):
            user_content = user_content + "\n\n" + additional_instructions
        else:
            # list of content blocks — append as text
            user_content.append({"type": "text", "text": additional_instructions})

    # Few-shot as prior user/assistant turns (canonical chat few-shot format)
    if few_shot_demos and encode_image_fn:
        fewshot_turns = _build_fewshot_turns(few_shot_demos, encode_image_fn)
    else:
        fewshot_turns = []

    return [
        {"role": "system", "content": system_prompt},
        *fewshot_turns,
        {"role": "user", "content": user_content},
    ]


def _encode_images_to_content(
    images: list[tuple[str, float]], encode_image_fn
) -> list[dict]:
    """Encode image paths to base64 content blocks."""
    content = []
    for img_path, score in images:
        try:
            img_base64 = _tile_image_b64(img_path, encode_image_fn)
            if img_base64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to encode tile: {e}")
    return content


def build_react_messages(
    query: str,
    retrieval_results: list[RetrievalResult],
    assistant_responses: list[str],
    encode_image_fn=None,
    prompt_version: str = "v1",
    is_last_turn: bool = False,
    previous_queries: list[str] | None = None,
) -> list[dict]:
    """Build multi-turn messages for ReAct retrieval loop.

    Args:
        query: Original question text.
        retrieval_results: List of RetrievalResult from each round.
        assistant_responses: List of assistant responses from previous rounds.
        encode_image_fn: Function to encode images to base64.
        prompt_version: "v1" (original) or "v2" (improved).
        is_last_turn: If True, add force-answer instruction.
        previous_queries: List of previous search queries (for v2, to avoid repetition).

    Returns:
        Messages list for the LLM.
    """
    _prompt_map = {
        "v1": SYSTEM_PROMPT_REACT,
        "v2": SYSTEM_PROMPT_REACT_V2,
        "multihop": SYSTEM_PROMPT_REACT_MULTIHOP,
    }
    system_prompt = _prompt_map.get(prompt_version, SYSTEM_PROMPT_REACT_V2)
    messages = [{"role": "system", "content": system_prompt}]

    for turn_idx, retrieval_result in enumerate(retrieval_results):
        # Build user message with evidence images
        if turn_idx == 0:
            user_content: list[dict] = [
                {
                    "type": "text",
                    "text": f"Question: {query}\n\nHere are retrieved Wikipedia evidence tiles:",
                }
            ]
        else:
            text = "Here are new search results for your query:"
            # Remind model of previous queries to avoid repetition (v2 and multihop)
            if prompt_version in ("v2", "multihop") and previous_queries:
                used = previous_queries[:turn_idx]
                if used:
                    text += f"\n⚠️ You already searched: {used}. Do NOT repeat these. Use DIFFERENT keywords."
            user_content = [{"type": "text", "text": text}]

        if retrieval_result.images and encode_image_fn:
            user_content.extend(
                _encode_images_to_content(retrieval_result.images, encode_image_fn)
            )

        if not retrieval_result.has_content:
            user_content.append(
                {"type": "text", "text": "(No results found for this search.)"}
            )

        # On last turn, inject force-answer instruction
        if is_last_turn and turn_idx == len(retrieval_results) - 1:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "\n⚠️ This is your FINAL turn. You MUST provide an answer now — do NOT search again. "
                        "Give your best answer based on ALL evidence seen so far. If uncertain, make your best guess."
                    ),
                }
            )

        messages.append({"role": "user", "content": user_content})

        # Add assistant response if we have one for this turn
        if turn_idx < len(assistant_responses):
            messages.append(
                {"role": "assistant", "content": assistant_responses[turn_idx]}
            )

    return messages


class LLMClient:
    """Simplified async LLM client for Gemini using Vertex AI."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout: float = 120.0,
        max_context_tokens: int | None = None,
        enable_thinking: bool | None = None,
        force_openai_compat: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.enable_thinking = enable_thinking
        print(f"context length model: {max_context_tokens}")

        # Gemini routes to Google GenAI SDK unless forced to OpenAI-compatible
        # (aggregators like OpenRouter / Commonstack expose Gemini via OAI-compat).
        self.is_gemini = ("gemini" in model.lower()) and not force_openai_compat

        if self.is_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai package is required for Gemini models. Install with: pip install google-genai"
                )

            # Use Vertex AI if GEMINI_API_KEY is set and GOOGLE_GENAI_USE_VERTEXAI is true
            vertex_api_key = os.getenv("GEMINI_API_KEY")
            use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
            if vertex_api_key and use_vertex:
                logger.info(f"Using Vertex AI for Gemini model: {model}")
                # Ensure GOOGLE_API_KEY is not set when using Vertex AI (it causes conflicts)
                if "GOOGLE_API_KEY" in os.environ:
                    logger.warning(
                        "GOOGLE_API_KEY is set but using Vertex AI. Unsetting GOOGLE_API_KEY to avoid conflicts."
                    )
                    del os.environ["GOOGLE_API_KEY"]
                os.environ["GEMINI_API_KEY"] = vertex_api_key
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
                self.gemini_client = genai.Client(
                    http_options=HttpOptions(api_version="v1")
                )
            else:
                # Use standard Gemini API
                logger.info(f"Using standard Gemini API for model: {model}")
                api_key = api_key if api_key != "dummy" else os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for Gemini models"
                    )
                self.gemini_client = genai.Client(api_key=api_key)
        else:
            # Use OpenAI-compatible API
            from openai import AsyncOpenAI

            logger.info(f"Using OpenAI-compatible API: {api_base}")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
                max_retries=0,
            )
            self.gemini_client = None

    async def generate(
        self, messages: list[dict], max_retries: int = 3, connection_retries: int = 12
    ) -> tuple[str, dict]:
        """Generate response from messages with retry on timeout/connection errors.

        Args:
            max_retries: Retry count for timeout errors.
            connection_retries: Retry count for connection errors (server restart).
                12 retries × 10s = ~2 min window for server to come back.

        Returns:
            Tuple of (generated_text, usage_dict).
        """
        # Check and truncate if needed
        if hasattr(self, "max_context_tokens") and self.max_context_tokens:
            estimated_tokens = self._estimate_tokens(messages)
            if estimated_tokens > self.max_context_tokens - self.max_tokens:
                logger.warning(
                    f"Estimated {estimated_tokens} tokens exceeds limit, truncating..."
                )
                messages = self._truncate_messages(messages, self.max_context_tokens)

        conn_attempts = 0
        timeout_attempts = 0
        while True:
            try:
                if self.is_gemini:
                    return await self._generate_gemini(messages)
                else:
                    return await self._generate_openai(messages)
            except asyncio.TimeoutError:
                timeout_attempts += 1
                if timeout_attempts >= max_retries:
                    raise
                wait_time = 2**timeout_attempts  # 2, 4, 8 seconds
                logger.warning(
                    f"Timeout on attempt {timeout_attempts}/{max_retries}, retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    timeout_attempts += 1
                    if timeout_attempts >= max_retries:
                        raise
                    wait_time = 2**timeout_attempts
                    logger.warning(
                        f"Timeout on attempt {timeout_attempts}/{max_retries}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                elif "connection" in error_str or "connect" in error_str:
                    conn_attempts += 1
                    if conn_attempts >= connection_retries:
                        raise
                    wait_time = 10  # fixed 10s — server restart takes ~30-60s
                    logger.warning(
                        f"Connection error ({conn_attempts}/{connection_retries}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                elif (
                    "429" in error_str
                    or "rate_limit" in error_str
                    or "rate limit" in error_str
                ):
                    # Provider rate limit — exponential backoff with jitter
                    timeout_attempts += 1
                    if timeout_attempts >= max_retries + 3:  # extra patience for 429
                        raise
                    import random

                    wait_time = min(60, 5 * (2**timeout_attempts)) + random.uniform(
                        0, 3
                    )
                    logger.warning(
                        f"429 rate-limit (attempt {timeout_attempts}), backing off {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def _generate_gemini(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate using Gemini API."""
        # Extract system prompt and user content
        system_prompt = None
        user_content = None

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        # Build parts for Gemini
        parts = []

        # Add system prompt to the beginning of user message if present
        if system_prompt:
            parts.append(Part(text=f"{system_prompt}\n\n"))

        # Process user content
        if isinstance(user_content, str):
            # Simple text
            if parts:
                parts[0] = Part(text=parts[0].text + user_content)
            else:
                parts.append(Part(text=user_content))
        elif isinstance(user_content, list):
            # Multi-modal content
            for item in user_content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if (
                        parts
                        and isinstance(parts[0], Part)
                        and hasattr(parts[0], "text")
                    ):
                        # Append to existing text part
                        parts[0] = Part(text=parts[0].text + text)
                    else:
                        parts.append(Part(text=text))
                elif item.get("type") == "image_url":
                    # Extract base64 image
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        try:
                            header, data = image_url.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]
                            image_bytes = base64.b64decode(data)
                            parts.append(
                                Part(
                                    inline_data=Blob(
                                        mime_type=mime_type, data=image_bytes
                                    )
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to process image: {e}")
                            raise

        # Create content
        content = Content(role="user", parts=parts)

        # Call API in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def _call_api():
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=[content],
                    config=GenerateContentConfig(
                        temperature=self.temperature, max_output_tokens=self.max_tokens
                    ),
                )
                return response
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise

        response = await loop.run_in_executor(None, _call_api)

        # Extract text
        text = response.text if hasattr(response, "text") and response.text else ""

        # Extract usage
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0),
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0),
                "total_tokens": getattr(usage_meta, "total_token_count", 0),
            }

        return text, usage

    async def _generate_openai(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate using OpenAI-compatible API."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        # Some modern reasoning models deprecate `temperature` (Claude Opus 4.7+, some GPT-5 variants).
        # Only send it when we actually want to override the default.
        model_lower = self.model.lower()
        drops_temperature = any(
            x in model_lower for x in ("opus-4-7", "opus-4-8", "gpt-5.4-pro")
        )
        if not drops_temperature:
            kwargs["temperature"] = self.temperature
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        response = await self.client.chat.completions.create(**kwargs)

        generated_text = response.choices[0].message.content

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return generated_text, usage

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count from messages (rough: ~4 chars per token)."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_chars += len(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Rough estimate for image tokens
                            total_chars += 1000 * 4  # ~1000 tokens per image
        return total_chars // 4

    def _truncate_messages(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Truncate text content in messages to fit within token limit."""
        # Reserve tokens for response
        available_tokens = max_tokens - self.max_tokens - 500  # buffer
        max_chars = available_tokens * 4

        truncated = []
        total_chars = 0

        for msg in messages:
            new_msg = msg.copy()
            content = msg.get("content", "")

            if isinstance(content, str):
                if total_chars + len(content) > max_chars:
                    remaining = max(0, max_chars - total_chars)
                    new_msg["content"] = (
                        content[:remaining]
                        + "\n\n[Content truncated due to context limit]"
                    )
                    logger.warning(
                        f"Truncated message content from {len(content)} to {remaining} chars"
                    )
                total_chars += len(new_msg["content"])
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if total_chars + len(text) > max_chars:
                            remaining = max(0, max_chars - total_chars)
                            new_item = item.copy()
                            new_item["text"] = (
                                text[:remaining]
                                + "\n\n[Content truncated due to context limit]"
                            )
                            new_content.append(new_item)
                            logger.warning(
                                f"Truncated text content from {len(text)} to {remaining} chars"
                            )
                            total_chars += remaining
                        else:
                            new_content.append(item)
                            total_chars += len(text)
                    else:
                        new_content.append(item)
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            total_chars += 1000 * 4  # image token estimate
                new_msg["content"] = new_content
            truncated.append(new_msg)

        return truncated
