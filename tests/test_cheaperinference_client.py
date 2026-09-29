"""Offline coverage of the optional Cheaper Inference evaluation reader."""

import asyncio
import sys
import types
from pathlib import Path
from runpy import run_path

import pytest

_LIB = Path(__file__).parents[1] / "eval" / "lib"
if "lib" not in sys.modules:
    _pkg = types.ModuleType("lib")
    _pkg.__path__ = [str(_LIB)]
    sys.modules["lib"] = _pkg

from lib.llm import LLMClient

get_config = run_path(str(_LIB / "model_config.py"))["get_cheaperinference_config"]


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gemini-3.1-pro"])
def test_cheaperinference_preserves_model_and_isolates_credentials(monkeypatch, model):
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "ci-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-key")
    monkeypatch.setenv("API_KEY", "generic-key")
    config = get_config(model)
    assert config == {
        "model": model,
        "api_base": "https://api.cheaperinference.com/v1",
        "api_key": "ci-test-key",
    }
    assert get_config(model, "explicit-key")["api_key"] == "explicit-key"


@pytest.mark.parametrize("key", [None, "", " ", "dummy"])
def test_cheaperinference_requires_own_key(monkeypatch, key):
    monkeypatch.delenv("CHEAPER_INFERENCE_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", key)
    monkeypatch.setenv("API_KEY", "generic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="CHEAPER_INFERENCE_API_KEY"):
        get_config("gpt-5.4-mini")


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gemini-3.1-pro"])
def test_cheaperinference_client_forwards_image_messages(monkeypatch, model):
    calls = []
    options = {}

    async def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="4"))],
            usage=types.SimpleNamespace(
                prompt_tokens=3, completion_tokens=1, total_tokens=4
            ),
        )

    def openai_client(**kwargs):
        options.update(kwargs)
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=openai_client)
    )
    client = LLMClient(
        **get_config(model, "ci-test-key"),
        force_openai_compat=True,
        max_tokens=64,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "2+2?"},
            ],
        }
    ]
    text, usage = asyncio.run(client.generate(messages))
    assert text == "4"
    assert usage["total_tokens"] == 4
    assert len(calls) == 1
    assert calls[0]["model"] == model
    assert calls[0]["messages"] == messages
    assert options["base_url"] == "https://api.cheaperinference.com/v1"
    assert options["api_key"] == "ci-test-key"
    assert not client.is_gemini
