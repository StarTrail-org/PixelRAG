from pathlib import Path
from runpy import run_path

import pytest

get_model_config = run_path(
    str(Path(__file__).parents[1] / "eval" / "lib" / "model_config.py")
)["get_model_config"]


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

    with pytest.raises(ValueError, match="Unsupported MINIMAX_API_REGION"):
        get_model_config("MiniMax-M3")


def test_minimax_model_config_does_not_match_unregistered_models(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://localhost:9000/v1")

    config = get_model_config("MiniMax-M2.7-highspeed")

    assert config["api_base"] == "http://localhost:9000/v1"
    assert config["model"] == "MiniMax-M2.7-highspeed"
