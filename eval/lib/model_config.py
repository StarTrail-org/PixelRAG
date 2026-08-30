"""Model configurations for SimpleQA evaluation.

This module provides model configurations to keep run_naive_simpleqa.py clean.
"""

import os
from typing import Any, Dict

MINIMAX_MODELS = {
    "minimax-m3": "MiniMax-M3",
    "minimax-m2.7": "MiniMax-M2.7",
}

MINIMAX_MODEL_METADATA = {
    "minimax-m3": {
        "context_window": 1_000_000,
        "pricing_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
    },
    "minimax-m2.7": {
        "context_window": 204_800,
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        "input_modalities": ["text"],
        "thinking": ["always_on"],
    },
}

# OrcaRouter is an OpenAI-compatible gateway, the same shape as the OpenRouter
# and Commonstack aggregators wired up in run_bench.py. Defined here so the CLI
# and the model-config branch cannot drift to different endpoints.
ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1"

MINIMAX_ENDPOINTS = {
    "global_en": {
        "api_base": "https://api.minimax.io/v1",
        "anthropic_api_base": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "api_base": "https://api.minimaxi.com/v1",
        "anthropic_api_base": "https://api.minimaxi.com/anthropic",
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configuration based on model name.

    Args:
        model_name: Name of the model (e.g., 'Qwen/Qwen3-VL-4B-Instruct', 'gemini-3-pro-preview')

    Returns:
        Dictionary with endpoint, authentication, model, and capability metadata.
    """
    model_lower = model_name.lower()

    # MiniMax models (OpenAI-compatible API)
    minimax_key = model_lower.rsplit("/", 1)[-1]
    minimax_model = MINIMAX_MODELS.get(minimax_key)
    if minimax_model:
        # The two bases are independent overrides for gateways and proxies, so
        # resolve them first and consult a region only for whichever was left
        # unset. Validating the region up front rejected deployments that had
        # pinned both endpoints and were not using a region at all.
        api_base = os.getenv("MINIMAX_API_BASE")
        anthropic_api_base = os.getenv("MINIMAX_ANTHROPIC_API_BASE")
        if api_base is None or anthropic_api_base is None:
            region = os.getenv("MINIMAX_API_REGION", "global_en").lower()
            if region not in MINIMAX_ENDPOINTS:
                supported = ", ".join(MINIMAX_ENDPOINTS)
                raise ValueError(
                    f"Unsupported MINIMAX_API_REGION {region!r}; "
                    f"expected one of: {supported}"
                )
            endpoints = MINIMAX_ENDPOINTS[region]
            if api_base is None:
                api_base = endpoints["api_base"]
            if anthropic_api_base is None:
                anthropic_api_base = endpoints["anthropic_api_base"]

        return {
            "api_base": api_base,
            "anthropic_api_base": anthropic_api_base,
            "api_key": os.getenv("MINIMAX_API_KEY", os.getenv("API_KEY", "dummy")),
            "model": minimax_model,
            # MINIMAX_MODELS is the registry; this is a side table keyed off it
            # with nothing enforcing that the two agree. A model listed in the
            # registry has to resolve whether or not its capability entry has
            # been filled in — the caller reads api_base, api_key and model.
            **MINIMAX_MODEL_METADATA.get(minimax_key, {}),
        }

    # OrcaRouter models (OpenAI-compatible gateway)
    if "orcarouter" in model_lower:
        return {
            "api_base": os.getenv("ORCAROUTER_API_BASE", ORCAROUTER_API_BASE),
            "api_key": os.getenv("ORCAROUTER_API_KEY", os.getenv("API_KEY", "dummy")),
            "model": model_name,
        }

    # Gemini models
    if "gemini" in model_lower:
        # Check for Vertex AI first
        vertex_api_key = os.getenv("GEMINI_API_KEY")
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"

        if use_vertex and vertex_api_key:
            # Using Vertex AI - don't pass api_key, use environment variable instead
            api_key = None  # Vertex AI uses environment variable, not api_key parameter
        else:
            # Using standard Gemini API
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for Gemini models. "
                    "Set it with: export GOOGLE_API_KEY='your-api-key' or export GEMINI_API_KEY='your-api-key' and GOOGLE_GENAI_USE_VERTEXAI=true"
                )

        # For Gemini models, we use Google's Generative AI SDK directly
        # The api_base is not used for Gemini (SDK handles it internally)
        # But we set a placeholder for compatibility
        api_base = None  # Not used for Gemini SDK

        return {
            "api_base": api_base,
            "api_key": api_key,
            "model": model_name,  # Use the model name as-is
        }

    # Default: assume OpenAI-compatible API (vLLM, etc.)
    return {
        "api_base": os.getenv("API_BASE", "http://localhost:8000/v1"),
        "api_key": os.getenv("API_KEY", "dummy"),
        "model": model_name,
    }


def get_output_filename(
    output_dir: str,
    model_name: str,
    mode: str = "naive",
    num_examples: int = 1000,
    url_screenshot: bool = False,
    task: str = "simpleqa",
) -> str:
    """
    Generate output filename with model name and task included.

    Args:
        output_dir: Base output directory (e.g., 'eval_output/naive_qa')
        model_name: Model name (e.g., 'Qwen/Qwen3-VL-4B-Instruct')
        mode: Evaluation mode ('naive', 'screenshot', 'retrieval')
        num_examples: Number of examples
        url_screenshot: Whether URL screenshot mode is enabled
        task: Task/benchmark name (e.g., 'simpleqa', 'encyclopedic_vqa', 'worldvqa')

    Returns:
        Full output file path
    """
    # Clean model name for filename (replace special chars)
    model_safe = (
        model_name.replace("/", "_").replace(":", "_").replace("-", "_").lower()
    )

    # Build filename components (task first for easy distinction)
    parts = [task]
    if url_screenshot:
        parts.append("urlscreenshot")
    parts.append(mode)
    parts.append(model_safe)
    parts.append(str(num_examples))

    filename = "_".join(parts) + ".jsonl"
    return os.path.join(output_dir, filename)
