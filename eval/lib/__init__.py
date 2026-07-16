"""SimpleQA evaluation modules.

Architecture:
- screenshot.py: Screenshot capture utilities (Selenium)
- data.py: Data loading and preparation (screenshots, text fetching)
- retrieval.py: Retrieval strategies (naive, screenshot, text, vector)
- llm.py: LLM client and prompt building
"""

from .screenshot import capture_screenshot, encode_image, encode_image_for_vlm
from .simpleqa_data import (
    load_simpleqa_data,
    load_simpleqa_verified_data,
    load_text_cache,
    extract_url_from_metadata,
    capture_screenshot_for_example,
    capture_screenshot_async,
    encode_screenshot,
    encode_screenshot_async,
    encode_screenshot_for_vlm,
    encode_screenshot_for_vlm_async,
    fetch_webpage_text,
    fetch_text_for_example,
    fetch_text_async,
    make_compressed_encoder,
    load_nq_data,
    load_triviaqa_data,
    load_nq_tables_data,
    load_piqa_data,
    load_hellaswag_data,
    load_commonsenseqa_data,
    load_openbookqa_data,
    load_arc_data,
)
from .retrieval import (
    BaseRetriever,
    EVQANoRetrievalRetriever,
    WorldVQANoRetrievalRetriever,
    NaiveRetriever,
    ScreenshotRetriever,
    TiledScreenshotRetriever,
    TextRetriever,
    JinaReaderRetriever,
    WikipediaAPIRetriever,
    VectorRetriever,
    ColQwenVectorRetriever,
    TiledVectorRetriever,
    TiledColQwenVectorRetriever,
    TiledQwen3VLEmbeddingRetriever,
    TextVectorRetriever,
    DsServeRetriever,
    LocalAPIRetriever,
    TextAPIRetriever,
    OCRWrappedRetriever,
    RenderedTextWrapper,
    HybridRetriever,
    HTMLDOMLookupRetriever,
    RetrievalResult,
)
from .llm import LLMClient, build_messages, build_react_messages
from .pixel_query import PixelQueryRenderer, QueryImageTextRenderer
from .simpleqa_filter import load_simpleqa_wikipedia, load_simpleqa_by_domain

__all__ = [
    # Screenshot utilities
    "capture_screenshot",
    "encode_image",
    "encode_image_for_vlm",
    # Data loading
    "load_simpleqa_data",
    "load_simpleqa_verified_data",
    "load_simpleqa_wikipedia",
    "load_simpleqa_by_domain",
    "load_nq_data",
    "load_triviaqa_data",
    "load_nq_tables_data",
    "load_piqa_data",
    "load_hellaswag_data",
    "load_commonsenseqa_data",
    "load_openbookqa_data",
    "load_arc_data",
    "load_text_cache",
    "extract_url_from_metadata",
    # Data preparation - screenshots
    "capture_screenshot_for_example",
    "capture_screenshot_async",
    "encode_screenshot",
    "encode_screenshot_async",
    "encode_screenshot_for_vlm",
    "encode_screenshot_for_vlm_async",
    # Data preparation - text
    "fetch_webpage_text",
    "fetch_text_for_example",
    "fetch_text_async",
    # Pixel compression
    "make_compressed_encoder",
    # Retrieval
    "BaseRetriever",
    "EVQANoRetrievalRetriever",
    "WorldVQANoRetrievalRetriever",
    "NaiveRetriever",
    "ScreenshotRetriever",
    "TiledScreenshotRetriever",
    "TextRetriever",
    "JinaReaderRetriever",
    "WikipediaAPIRetriever",
    "VectorRetriever",
    "ColQwenVectorRetriever",
    "TiledVectorRetriever",
    "TiledColQwenVectorRetriever",
    "TiledQwen3VLEmbeddingRetriever",
    "TextVectorRetriever",
    "DsServeRetriever",
    "LocalAPIRetriever",
    "TextAPIRetriever",
    "OCRWrappedRetriever",
    "RenderedTextWrapper",
    "HybridRetriever",
    "HTMLDOMLookupRetriever",
    "RetrievalResult",
    # LLM
    "LLMClient",
    "build_messages",
    "build_react_messages",
    # Pixel query
    "PixelQueryRenderer",
    "QueryImageTextRenderer",
]
