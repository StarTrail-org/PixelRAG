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
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "fallback-key")

    assert get_model_config(model_name) == {
        "api_base": "https://api.minimax.io/v1",
        "api_key": "fallback-key",
        "model": model_id,
    }


def test_minimax_model_config_supports_endpoint_and_key_overrides(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")
    monkeypatch.setenv("MINIMAX_API_KEY", "provider-key")

    config = get_model_config("MiniMax-M3")

    assert config["api_base"] == "https://api.minimaxi.com/v1"
    assert config["api_key"] == "provider-key"


def test_minimax_model_config_does_not_match_unregistered_models(monkeypatch):
    monkeypatch.setenv("API_BASE", "http://localhost:9000/v1")

    config = get_model_config("MiniMax-M2.7-highspeed")

    assert config["api_base"] == "http://localhost:9000/v1"
    assert config["model"] == "MiniMax-M2.7-highspeed"
