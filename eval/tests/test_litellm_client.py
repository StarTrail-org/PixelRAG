"""Unit tests for the LiteLLM path in LLMClient (lib/llm.py).

litellm is stubbed in sys.modules so these run without network or real keys.
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stub_litellm(content="4"):
    fake = types.ModuleType("litellm")

    async def acompletion(**kwargs):
        acompletion.calls.append(kwargs)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(
            prompt_tokens=3, completion_tokens=1, total_tokens=4
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)

    acompletion.calls = []
    fake.acompletion = acompletion
    sys.modules["litellm"] = fake
    return fake


def test_litellm_dispatch_forwards_model_creds_and_drop_params():
    fake = _stub_litellm("4")
    from lib.llm import LLMClient

    client = LLMClient(
        model="anthropic/claude-3-5-sonnet",
        api_key="sk-x",
        api_base="",
        use_litellm=True,
        max_tokens=64,
    )
    text, usage = asyncio.run(client.generate([{"role": "user", "content": "2+2?"}]))
    assert text == "4"
    assert usage["total_tokens"] == 4
    call = fake.acompletion.calls[0]
    assert call["model"] == "anthropic/claude-3-5-sonnet"
    assert call["drop_params"] is True
    assert call["api_key"] == "sk-x"
    assert "api_base" not in call  # blank base omitted
    assert call["temperature"] == 0.0


def test_credentials_omitted_when_dummy_or_blank():
    fake = _stub_litellm("ok")
    from lib.llm import LLMClient

    client = LLMClient(
        model="gemini/gemini-1.5-pro",
        api_key="dummy",
        api_base="",
        use_litellm=True,
    )
    asyncio.run(client.generate([{"role": "user", "content": "hi"}]))
    call = fake.acompletion.calls[0]
    assert "api_key" not in call  # "dummy" treated as unset -> provider env fallback
    assert "api_base" not in call


def test_api_base_forwarded_for_proxy():
    fake = _stub_litellm("ok")
    from lib.llm import LLMClient

    client = LLMClient(
        model="openai/gpt-4o-mini",
        api_key="sk-proxy",
        api_base="http://localhost:4000",
        use_litellm=True,
    )
    asyncio.run(client.generate([{"role": "user", "content": "hi"}]))
    assert fake.acompletion.calls[0]["api_base"] == "http://localhost:4000"


def test_use_litellm_takes_precedence_over_gemini_routing():
    _stub_litellm("x")
    from lib.llm import LLMClient

    client = LLMClient(model="gemini/gemini-1.5-pro", use_litellm=True)
    # LiteLLM handles gemini itself, so the native GenAI branch must be disabled.
    assert client.is_gemini is False
    assert client.gemini_client is None


if __name__ == "__main__":
    test_litellm_dispatch_forwards_model_creds_and_drop_params()
    test_credentials_omitted_when_dummy_or_blank()
    test_api_base_forwarded_for_proxy()
    test_use_litellm_takes_precedence_over_gemini_routing()
    print("all litellm client tests passed")
