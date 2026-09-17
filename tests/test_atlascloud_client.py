"""Offline coverage of the optional Atlas Cloud evaluation reader."""

import asyncio
import sys
import types
from pathlib import Path
from runpy import run_path
from unittest.mock import AsyncMock

import pytest

_LIB = Path(__file__).parents[1] / "eval" / "lib"
if "lib" not in sys.modules:
    _pkg = types.ModuleType("lib")
    _pkg.__path__ = [str(_LIB)]
    sys.modules["lib"] = _pkg

from lib.llm import LLMClient

get_config = run_path(str(_LIB / "model_config.py"))["get_atlascloud_config"]


@pytest.mark.parametrize("model", ["openai/gpt-4.1-mini", "google/gemini-3-pro"])
def test_atlascloud_preserves_model_and_isolates_credentials(monkeypatch, model):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-key")
    monkeypatch.setenv("API_KEY", "generic-key")
    config = get_config(model)
    assert config == {
        "model": model,
        "api_base": "https://api.atlascloud.ai/v1",
        "api_key": "atlas-test-key",
    }
    assert get_config(model, "explicit-key")["api_key"] == "explicit-key"


@pytest.mark.parametrize("key", [None, "", " ", "dummy"])
def test_atlascloud_requires_own_key(monkeypatch, key):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("ATLASCLOUD_API_KEY", key)
    monkeypatch.setenv("API_KEY", "generic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        get_config("deepseek-ai/deepseek-v3.2")


@pytest.mark.parametrize(
    "error",
    [
        None,
        asyncio.TimeoutError(),
        RuntimeError("Connection error"),
        RuntimeError("429"),
    ],
)
def test_atlascloud_client_routes_once(monkeypatch, error):
    calls = []
    options = {}

    async def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
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
        **get_config("google/gemini-3-pro", "atlas-test-key"),
        force_openai_compat=True,
        retry_requests=False,
        max_tokens=64,
    )
    messages = [{"role": "user", "content": "2+2?"}]
    if error is None:
        text, usage = asyncio.run(client.generate(messages))
        assert text == "4"
        assert usage["total_tokens"] == 4
    else:
        with pytest.raises(type(error)):
            asyncio.run(client.generate(messages))
    assert len(calls) == 1
    assert calls[0]["model"] == "google/gemini-3-pro"
    assert calls[0]["messages"] == messages
    assert options["base_url"] == "https://api.atlascloud.ai/v1"
    assert options["api_key"] == "atlas-test-key"
    assert options["max_retries"] == 0
    assert not client.is_gemini


def test_other_readers_keep_existing_retry_behavior(monkeypatch):
    client = LLMClient(model="openai/test-model", use_litellm=True)
    generate = AsyncMock(side_effect=[asyncio.TimeoutError(), ("4", {})])
    monkeypatch.setattr(client, "_generate_litellm", generate)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    assert asyncio.run(client.generate([{"role": "user", "content": "2+2?"}])) == (
        "4",
        {},
    )
    assert generate.await_count == 2
