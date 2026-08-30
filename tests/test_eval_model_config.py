from pathlib import Path
from runpy import run_path

import pytest

MODEL_CONFIG = run_path(
    str(Path(__file__).parents[1] / "eval" / "lib" / "model_config.py")
)
get_model_config = MODEL_CONFIG["get_model_config"]


@pytest.mark.parametrize(
    ("model_name", "model_id"),
    [
        ("MiniMax-M3", "MiniMax-M3"),
        ("MiniMax/MiniMax-M3", "MiniMax-M3"),
        ("MiniMax-M2.7", "MiniMax-M2.7"),
        ("MiniMax/MiniMax-M2.7", "MiniMax-M2.7"),
    ],
)
def test_minimax_model_config_uses_canonical_model_id(
    monkeypatch, model_name, model_id
):
    monkeypatch.delenv("MINIMAX_API_REGION", raising=False)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "fallback-key")

    config = get_model_config(model_name)

    assert config["api_base"] == "https://api.minimax.io/v1"
    assert config["anthropic_api_base"] == "https://api.minimax.io/anthropic"
    assert config["api_key"] == "fallback-key"
    assert config["model"] == model_id


@pytest.mark.parametrize(
    ("region", "openai_base", "anthropic_base"),
    [
        (
            "global_en",
            "https://api.minimax.io/v1",
            "https://api.minimax.io/anthropic",
        ),
        (
            "cn_zh",
            "https://api.minimaxi.com/v1",
            "https://api.minimaxi.com/anthropic",
        ),
    ],
)
def test_minimax_model_config_selects_regional_endpoints(
    monkeypatch, region, openai_base, anthropic_base
):
    monkeypatch.setenv("MINIMAX_API_REGION", region)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)

    config = get_model_config("MiniMax-M3")

    assert config["api_base"] == openai_base
    assert config["anthropic_api_base"] == anthropic_base


def test_minimax_model_config_supports_endpoint_and_key_overrides(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", "https://gateway.example/openai")
    monkeypatch.setenv(
        "MINIMAX_ANTHROPIC_API_BASE", "https://gateway.example/anthropic"
    )
    monkeypatch.setenv("MINIMAX_API_KEY", "provider-key")

    config = get_model_config("MiniMax-M3")

    assert config["api_base"] == "https://gateway.example/openai"
    assert config["anthropic_api_base"] == "https://gateway.example/anthropic"
    assert config["api_key"] == "provider-key"


@pytest.mark.parametrize(
    ("model_name", "context_window", "modalities", "thinking", "pricing"),
    [
        (
            "MiniMax-M3",
            1_000_000,
            ["text", "image", "video"],
            ["adaptive", "disabled"],
            {"input": 0.6, "output": 2.4, "cache_read": 0.12, "cache_write": None},
        ),
        (
            "MiniMax-M2.7",
            204_800,
            ["text"],
            ["always_on"],
            {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": 0.375},
        ),
    ],
)
def test_minimax_model_config_exposes_current_metadata(
    model_name, context_window, modalities, thinking, pricing
):
    config = get_model_config(model_name)

    assert config["context_window"] == context_window
    assert config["input_modalities"] == modalities
    assert config["thinking"] == thinking
    assert config["pricing_usd_per_million_tokens"] == pricing


def test_minimax_model_config_rejects_unknown_region(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_REGION", "unknown")
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)

    with pytest.raises(ValueError, match="Unsupported MINIMAX_API_REGION"):
        get_model_config("MiniMax-M3")


def test_minimax_model_config_does_not_match_unregistered_models(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://localhost:9000/v1")

    config = get_model_config("MiniMax-M2.7-highspeed")

    assert config["api_base"] == "http://localhost:9000/v1"
    assert config["model"] == "MiniMax-M2.7-highspeed"


def test_minimax_model_config_skips_region_when_both_bases_are_overridden(
    monkeypatch,
):
    """An unsupported region is irrelevant when nothing reads it.

    The README offers the two base variables as independent overrides for
    gateways and proxies. Validating the region before applying them made that
    untrue: a deployment that pinned both endpoints still had to name a region
    from a list it was not using.
    """
    monkeypatch.setenv("MINIMAX_API_REGION", "eu")
    monkeypatch.setenv("MINIMAX_API_BASE", "https://gateway.internal/openai")
    monkeypatch.setenv(
        "MINIMAX_ANTHROPIC_API_BASE", "https://gateway.internal/anthropic"
    )

    config = get_model_config("MiniMax-M3")

    assert config["api_base"] == "https://gateway.internal/openai"
    assert config["anthropic_api_base"] == "https://gateway.internal/anthropic"


def test_minimax_model_config_overrides_one_base_and_keeps_the_region_for_the_other(
    monkeypatch,
):
    """ "Independent" has to mean per-variable, not all-or-nothing."""
    monkeypatch.setenv("MINIMAX_API_REGION", "cn_zh")
    monkeypatch.setenv("MINIMAX_API_BASE", "https://gateway.internal/openai")
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)

    config = get_model_config("MiniMax-M3")

    assert config["api_base"] == "https://gateway.internal/openai"
    assert config["anthropic_api_base"] == "https://api.minimaxi.com/anthropic"


def test_minimax_model_registered_without_metadata_still_resolves(monkeypatch):
    """Registering a model must not require remembering a second dict.

    MINIMAX_MODELS and MINIMAX_MODEL_METADATA are separate mappings with
    nothing keeping them in sync, and the metadata splat indexed the second
    directly -- so adding a model to the first alone raised KeyError for every
    caller of that model. Nothing reads the metadata anyway: run_bench.py, the
    only consumer, uses api_base, api_key and model.
    """
    monkeypatch.delenv("MINIMAX_API_REGION", raising=False)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)
    MODEL_CONFIG["MINIMAX_MODELS"]["minimax-m4"] = "MiniMax-M4"
    try:
        config = get_model_config("MiniMax-M4")
    finally:
        MODEL_CONFIG["MINIMAX_MODELS"].pop("minimax-m4")

    assert config["model"] == "MiniMax-M4"
    assert config["api_base"] == "https://api.minimax.io/v1"
    assert config["anthropic_api_base"] == "https://api.minimax.io/anthropic"
    assert "context_window" not in config
