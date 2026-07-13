"""Model configurations for SimpleQA evaluation.

This module provides model configurations to keep run_naive_simpleqa.py clean.
"""

import os
from typing import Dict, Optional


MINIMAX_MODELS = {
    "minimax-m3": "MiniMax-M3",
    "minimax-m2.7": "MiniMax-M2.7",
}


def get_model_config(model_name: str) -> Dict[str, Optional[str]]:
    """
    Get model configuration based on model name.

    Args:
        model_name: Name of the model (e.g., 'Qwen/Qwen3-VL-4B-Instruct', 'gemini-3-pro-preview')

    Returns:
        Dictionary with 'api_base', 'api_key', and 'model' keys.
    """
    model_lower = model_name.lower()

    # MiniMax models (OpenAI-compatible API)
    minimax_model = MINIMAX_MODELS.get(model_lower.rsplit("/", 1)[-1])
    if minimax_model:
        return {
            "api_base": os.getenv("MINIMAX_API_BASE", "https://api.minimax.io/v1"),
            "api_key": os.getenv("MINIMAX_API_KEY", os.getenv("API_KEY", "dummy")),
            "model": minimax_model,
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
