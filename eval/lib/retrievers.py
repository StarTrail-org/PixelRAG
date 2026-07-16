"""Retriever factory — maps CLI flags to a (retriever, mode_str) pair.

Extracted from run_naive_simpleqa.py to keep the orchestrator readable.
"""

import logging
import os
import sys

from . import (
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
    EVQANoRetrievalRetriever,
    WorldVQANoRetrievalRetriever,
    TextVectorRetriever,
    DsServeRetriever,
    LocalAPIRetriever,
    TextAPIRetriever,
    OCRWrappedRetriever,
    RenderedTextWrapper,
    HybridRetriever,
    HTMLDOMLookupRetriever,
    load_text_cache,
)
from .retrieval import _get_query_image_path_for_example, _save_task_query_image

logger = logging.getLogger(__name__)

TILE_WIDTH = 1024


def build_retriever(args, examples, model, api_base, api_key):
    """Build a retriever from CLI args.

    Args:
        args: Parsed argparse namespace.
        examples: Loaded dataset examples (some retrievers need them for setup).
        model: Reader model name (for query rewrite fallback).
        api_base: Reader API base (for query rewrite fallback).
        api_key: Reader API key (for query rewrite fallback).

    Returns:
        (retriever, mode_str) tuple.
    """
    tile_size = (TILE_WIDTH, args.tile_height)

    retrieval_mode_count = sum(
        [
            args.url_screenshot,
            args.url_tiled_screenshot,
            args.url_text,
            args.url_jina_reader,
            args.retrieval_augment,
            args.use_tiled_retrieval,
            args.text_vector,
            args.local_api,
            args.text_api,
            args.html_dom_lookup,
            args.hybrid,
        ]
    )

    if args.url_screenshot:
        retriever = ScreenshotRetriever(
            screenshot_dir=args.screenshot_dir, max_pixels=args.max_pixels
        )
        mode = f"Screenshot (Ground Truth, max_pixels={args.max_pixels or 'None'})"

    elif args.url_tiled_screenshot:
        retriever = TiledScreenshotRetriever(
            screenshot_dir=args.screenshot_dir,
            tiles_dir=args.tiles_dir,
            tile_size=tile_size,
            overlap=args.tile_overlap,
            max_tiles=args.max_tiles,
        )
        mode = f"Tiled Screenshot (Ground Truth, max_tiles={args.max_tiles})"

    elif args.url_text:
        text_cache = None
        if args.text_cache and os.path.exists(args.text_cache):
            text_cache = load_text_cache(args.text_cache)
            logger.info(f"Loaded {len(text_cache)} cached items from {args.text_cache}")
        elif args.text_cache:
            logger.info(
                f"Cache file not found: {args.text_cache} (will fetch from source)"
            )
        if args.text_source == "jina":
            retriever = JinaReaderRetriever(
                max_chars=args.max_context_chars,
                api_key=args.jina_api_key,
                text_cache=text_cache,
                cache_path=args.text_cache,
            )
            mode = "Text RAG (Jina)"
        elif args.text_source == "wikipedia":
            retriever = WikipediaAPIRetriever(
                max_chars=args.max_context_chars,
                text_cache=text_cache,
                cache_path=args.text_cache,
            )
            mode = "Text RAG (Wikipedia API)"
        else:
            retriever = TextRetriever(
                max_chars=args.max_context_chars,
                text_cache=text_cache,
                cache_path=args.text_cache,
            )
            mode = "Text RAG (Crawl)"

    elif args.url_jina_reader:
        logger.warning(
            "--url-jina-reader is deprecated, use --url-text --text-source jina instead"
        )
        retriever = JinaReaderRetriever(
            max_chars=args.max_context_chars, api_key=args.jina_api_key
        )
        mode = "Jina Reader"

    elif args.retrieval_augment:
        if args.use_colqwen_retrieval:
            retriever = ColQwenVectorRetriever(
                index_path=args.colqwen_index_path,
                screenshot_dir=args.screenshot_dir,
                model_name=args.colqwen_model,
                search_method=args.colqwen_search_method,
                first_stage_k=args.colqwen_first_stage_k,
                rebuild_index=args.rebuild_colqwen_index,
                recursive=args.colqwen_recursive,
                top_k=args.retrieval_top_k,
                examples=examples,
            )
            mode = "ColQwen Vector Retrieval"
        else:
            retriever = VectorRetriever(
                api_key=args.jina_api_key,
                screenshot_dir=args.screenshot_dir,
                cache_path=args.retrieval_cache,
                use_multivector=not args.single_vector,
                top_k=args.retrieval_top_k,
                examples=examples,
            )
            mode = "Vector Retrieval"

    elif args.use_tiled_retrieval:
        if args.use_colqwen_retrieval:
            tiled_index_path = args.colqwen_index_path.replace(
                ".leann", f"_tiled_{args.tile_height}.leann"
            )
            retriever = TiledColQwenVectorRetriever(
                index_path=tiled_index_path,
                screenshot_dir=args.screenshot_dir,
                tiles_dir=args.tiles_dir,
                tile_size=tile_size,
                overlap=args.tile_overlap,
                model_name=args.colqwen_model,
                search_method=args.colqwen_search_method,
                first_stage_k=args.colqwen_first_stage_k,
                rebuild_index=args.rebuild_colqwen_index,
                top_k=args.retrieval_top_k,
                examples=examples,
            )
            mode = "Tiled ColQwen Vector Retrieval"
        elif args.use_qwen3vl_embedding:
            qwen3vl_cache_path = args.retrieval_cache
            if qwen3vl_cache_path is None:
                task_subset = f"{args.task}_{args.subset}" if args.subset else args.task
                qwen3vl_cache_path = f"qwen3vl_tiles_{task_subset}_{TILE_WIDTH}x{args.tile_height}_{args.num_examples}ex_embeddings.pkl"
            qwen3vl_gpu_ids = [int(x.strip()) for x in args.qwen3vl_gpu_ids.split(",")]

            pixel_query_map = None
            if (
                args.task == "encyclopedic_vqa"
                and not args.evqa_multimodal_query
                and not args.evqa_multi_image_query
            ):
                from .pixel_query import QueryImageTextRenderer

                tiles_dir = args.tiles_dir or "tiles/evqa"
                renderer = QueryImageTextRenderer(
                    output_dir="query_cards/evqa",
                    tiles_dir=tiles_dir,
                )
                pixel_query_map = {}
                for ex in examples:
                    inat_path = _get_query_image_path_for_example(ex, tiles_dir)
                    path = renderer.render(
                        ex["id"], ex["problem"], inat_path, force=args.force
                    )
                    pixel_query_map[ex["id"]] = path
                logger.info(f"EVQA query cards: {len(pixel_query_map)} rendered")
            elif args.pixel_query:
                from .pixel_query import PixelQueryRenderer

                pq_renderer = PixelQueryRenderer(output_dir=args.pixel_query_dir)
                pixel_query_map = pq_renderer.render_all(examples)
                logger.info(
                    f"Pixel query mode: rendered {len(pixel_query_map)} query images"
                )

            retriever = TiledQwen3VLEmbeddingRetriever(
                screenshot_dir=args.screenshot_dir,
                tiles_dir=args.tiles_dir,
                tile_size=tile_size,
                overlap=args.tile_overlap,
                cache_path=qwen3vl_cache_path,
                model_name=args.qwen3vl_model,
                top_k=args.retrieval_top_k,
                examples=examples,
                gpu_ids=qwen3vl_gpu_ids,
                tensor_parallel_size=args.qwen3vl_tp_size,
                pixel_query_map=pixel_query_map,
                multimodal_query_text_only=args.evqa_multimodal_query_text_only,
                multimodal_query_image_only=args.evqa_multimodal_query_image_only,
                multi_image_query=args.evqa_multi_image_query,
                prebuilt_tiles_dir=getattr(args, "prebuilt_tiles_dir", None),
                embedding_backend=getattr(args, "embedding_backend", "vllm"),
                peft_adapter=getattr(args, "peft_adapter", None),
            )
            mode = "Tiled Qwen3-VL-Embedding Retrieval"
            if getattr(args, "prebuilt_tiles_dir", None):
                mode += " (prebuilt hard-mini)"
            if args.task == "encyclopedic_vqa":
                if args.evqa_multi_image_query:
                    mode += " (EVQA multi-image query)"
                elif args.evqa_multimodal_query:
                    if args.evqa_multimodal_query_text_only:
                        mode += " (EVQA multimodal: text-only)"
                    elif args.evqa_multimodal_query_image_only:
                        mode += " (EVQA multimodal: image-only)"
                    else:
                        mode += " (EVQA multimodal: text+image)"
                else:
                    mode += " (EVQA query card)"
            elif args.pixel_query:
                mode += " (Pixel Query)"
        else:
            tile_cache_path = args.retrieval_cache
            if tile_cache_path is None:
                vector_type = "single" if args.single_vector else "multi"
                task_subset = f"{args.task}_{args.subset}" if args.subset else args.task
                tile_cache_path = f"jina_tiles_{task_subset}_{TILE_WIDTH}x{args.tile_height}_{vector_type}_{args.num_examples}ex_embeddings.pkl"
            retriever = TiledVectorRetriever(
                api_key=args.jina_api_key,
                screenshot_dir=args.screenshot_dir,
                tiles_dir=args.tiles_dir,
                tile_size=tile_size,
                overlap=args.tile_overlap,
                cache_path=tile_cache_path,
                use_multivector=not args.single_vector,
                top_k=args.retrieval_top_k,
                examples=examples,
            )
            mode = "Tiled Jina Vector Retrieval"

    elif args.local_api:
        rw_model = args.rewrite_model or model
        rw_api_base = args.rewrite_api_base or api_base
        rw_api_key = args.rewrite_api_key or api_key
        reranker_obj = None
        if args.reranker:
            logger.info(f"Loading reranker on GPU {args.reranker_gpu_id}")
            from .reranker import Qwen3VLReranker

            reranker_obj = Qwen3VLReranker(
                model_name=args.reranker_model,
                gpu_id=args.reranker_gpu_id,
            )
        query_image_fn = None
        if args.no_query_image:
            logger.info(
                "--no-query-image set: retrieval queries will be text-only (reader still sees query image)"
            )
        elif args.task == "encyclopedic_vqa":
            _tiles_dir = args.tiles_dir or "tiles/evqa"

            def query_image_fn(ex, _td=_tiles_dir):
                return _get_query_image_path_for_example(ex, _td, quiet=True)
        elif args.task in (
            "worldvqa",
            "simplevqa",
            "factualvqa",
            "mmsearch",
            "webqa",
            "multimodalqa",
        ):
            _task = args.task

            def query_image_fn(ex, _t=_task):
                return _save_task_query_image(ex, _t, base_dir="tiles")

        retriever = LocalAPIRetriever(
            api_url=args.local_api_url,
            top_k=args.retrieval_top_k,
            query_rewrite=args.query_rewrite,
            rewrite_model=rw_model if args.query_rewrite else None,
            rewrite_api_base=rw_api_base if args.query_rewrite else None,
            rewrite_api_key=rw_api_key if args.query_rewrite else "dummy",
            nprobe=args.nprobe,
            reranker=reranker_obj,
            rerank_top_k=args.rerank_top_k,
            query_image_fn=query_image_fn,
            multi_image_query=args.evqa_multi_image_query,
            tiles_dir=args.tiles_dir or "tiles/evqa",
            query_instruction=args.query_instruction,
        )
        mode = f"Local API Retrieval ({args.local_api_url})"
        if args.query_instruction is not None:
            mode += f" [instr={args.query_instruction!r}]"
        if args.evqa_multi_image_query:
            mode += " (multi-image query)"
        elif query_image_fn:
            mode += " (multimodal query)"
        if args.query_rewrite:
            mode += f" + QueryRewrite({rw_model})"
        if args.reranker:
            mode += f" + Reranker({args.reranker_model}, top{args.rerank_top_k})"
        if args.react:
            mode += f" + ReAct({args.react_prompt}, max_turns={args.react_max_turns})"

    elif args.text_api:
        text_query_image_fn = None
        if not args.no_query_image:
            if args.task == "encyclopedic_vqa":
                _tiles_dir = args.tiles_dir or "tiles/evqa"

                def text_query_image_fn(ex, _td=_tiles_dir):
                    return _get_query_image_path_for_example(ex, _td, quiet=True)
            elif args.task in (
                "worldvqa",
                "simplevqa",
                "factualvqa",
                "mmsearch",
                "webqa",
                "multimodalqa",
            ):
                _task = args.task

                def text_query_image_fn(ex, _t=_task):
                    return _save_task_query_image(ex, _t, base_dir="tiles")

        retriever = TextAPIRetriever(
            api_url=args.text_api_url,
            top_k=args.retrieval_top_k,
            nprobe=args.nprobe,
            query_instruction=args.query_instruction,
            reader_top_k=args.reader_top_k,
            query_image_fn=text_query_image_fn,
        )
        mode = f"Text API Retrieval ({args.text_api_url})"
        if args.query_instruction is not None:
            mode += f" [instr={args.query_instruction!r}]"

    elif args.html_dom_lookup:
        retriever = HTMLDOMLookupRetriever(
            text_api_url=args.text_api_url,
            top_k=args.retrieval_top_k,
            nprobe=args.nprobe,
            query_instruction=args.query_instruction,
            reader_top_k=args.reader_top_k,
            query_image_fn=None,
            context_mode="section",
            llm_verify=getattr(args, "llm_verify", False),
        )
        mode = f"HTML DOM Lookup (text_api={args.text_api_url}, top_k={args.retrieval_top_k})"
        if args.llm_verify:
            mode += " [llm-verify]"

    elif args.hybrid:
        if args.read_as_text_ocr or args.render_as_image:
            print(
                "Error: --hybrid is not compatible with --read-as-text-ocr or --render-as-image."
            )
            sys.exit(1)
        image_base = LocalAPIRetriever(
            api_url=args.local_api_url,
            top_k=args.retrieval_top_k,
            nprobe=args.nprobe,
            tiles_dir=args.tiles_dir or "tiles/evqa",
            query_instruction=args.query_instruction,
        )
        text_base = TextAPIRetriever(
            api_url=args.text_api_url,
            top_k=args.retrieval_top_k,
            nprobe=args.nprobe,
            query_instruction=args.query_instruction,
            reader_top_k=args.reader_top_k,
        )
        retriever = HybridRetriever(
            image_base=image_base,
            text_base=text_base,
            top_k=args.retrieval_top_k,
            reader_top_k=args.reader_top_k,
        )
        mode = f"Hybrid Retrieval (image={args.local_api_url}, text={args.text_api_url}, top_k={args.retrieval_top_k})"

    elif args.text_vector:
        if args.text_source == "ds-serve":
            retriever = DsServeRetriever(
                api_url=args.ds_serve_api_url, top_k=args.retrieval_top_k
            )
            mode = "Text Vector (ds-serve)"
        else:
            text_cache_path = f"text_cache/text_cache_{args.text_source}.jsonl"
            text_cache = load_text_cache(text_cache_path)
            if not text_cache:
                print(f"Error: Text cache not found at {text_cache_path}")
                print(
                    f"Run with --url-text --text-source {args.text_source} first to build the cache."
                )
                sys.exit(1)

            if args.text_embed_preset == "qwen":
                embedding_model = "Qwen/Qwen3-Embedding-0.6B"
                embedding_mode = "sentence-transformers"
                embedding_options = {"batch_size": args.embed_batch_size}
                preset_name = "qwen3-0.6b"
            elif args.text_embed_preset == "jina":
                embedding_model = "jina-embeddings-v4"
                embedding_mode = "openai"
                embedding_options = {
                    "base_url": "https://api.jina.ai/v1",
                    "api_key": args.jina_api_key,
                }
                preset_name = "jina-v4"
            elif args.text_embed_preset == "contriever":
                embedding_model = "facebook/contriever"
                embedding_mode = "sentence-transformers"
                embedding_options = {"batch_size": args.embed_batch_size}
                preset_name = "contriever"
            else:
                embedding_model = "facebook/contriever"
                embedding_mode = "sentence-transformers"
                embedding_options = {"batch_size": args.embed_batch_size}
                preset_name = "contriever"

            index_path = (
                f"indexes/text_{args.text_source}_{preset_name}_c{args.chunk_size}"
            )
            retriever = TextVectorRetriever(
                text_cache=text_cache,
                index_path=index_path,
                embedding_model=embedding_model,
                embedding_mode=embedding_mode,
                embedding_options=embedding_options,
                top_k=args.retrieval_top_k,
                rebuild_index=args.rebuild_text_index,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            mode = f"Text Vector ({args.text_source}, {preset_name})"

    elif args.task == "encyclopedic_vqa" and retrieval_mode_count == 0:
        retriever = EVQANoRetrievalRetriever(tiles_dir=args.tiles_dir or "tiles/evqa")
        mode = "EVQA no retrieval (query + image only)"

    elif args.task == "worldvqa" and retrieval_mode_count == 0:
        retriever = WorldVQANoRetrievalRetriever()
        mode = "WorldVQA no retrieval (query + image only)"

    elif (
        args.task in ("simplevqa", "factualvqa", "mmsearch", "webqa", "multimodalqa")
        and retrieval_mode_count == 0
    ):
        retriever = WorldVQANoRetrievalRetriever()
        mode = f"{args.task} no retrieval (query + image only)"

    else:
        retriever = NaiveRetriever()
        mode = "Naive"

    # Ablation A: wrap image retriever with OCR
    if args.read_as_text_ocr:
        image_modes = (
            args.local_api
            or args.use_tiled_retrieval
            or args.retrieval_augment
            or args.url_screenshot
            or args.url_tiled_screenshot
        )
        if not image_modes:
            print(
                "Error: --read-as-text-ocr requires an image retrieval mode "
                "(--local-api, --use-tiled-retrieval, --retrieval-augment, "
                "--url-screenshot, or --url-tiled-screenshot)."
            )
            sys.exit(1)
        if args.react:
            print(
                "Error: --read-as-text-ocr is not compatible with --react "
                "(react bypasses the retriever wrapper on subsequent turns)."
            )
            sys.exit(1)
        retriever = OCRWrappedRetriever(
            base=retriever,
            ocr_url=args.ocr_url,
            model=args.ocr_model,
            cache_path=args.ocr_cache,
            concurrency=args.ocr_concurrency,
            reader_top_k=args.reader_top_k,
        )
        mode += f" + OCR({args.ocr_url})"
        logger.info(
            f"Ablation A: OCR wrapper enabled ({args.ocr_url}, cache={args.ocr_cache})"
        )

    # Ablation B: wrap text retriever with renderer
    if args.render_as_image:
        if not args.text_api:
            print(
                "Error: --render-as-image requires --text-api (needs a text retriever "
                "exposing get_hits())."
            )
            sys.exit(1)
        retriever = RenderedTextWrapper(
            base=retriever,
            render_dir=args.render_dir,
            reader_top_k=args.reader_top_k,
        )
        mode += f" + Render({args.render_dir})"
        logger.info(f"Ablation B: text->image renderer enabled (dir={args.render_dir})")

    return retriever, mode
