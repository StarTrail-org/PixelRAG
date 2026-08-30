"""Unit tests for the LiteLLM path in eval/lib/llm.py.

Lives here rather than under ``eval/`` because CI runs ``pytest tests/``: a
test outside this directory is never collected, so it gates nothing.

``eval/lib/__init__.py`` eagerly imports the retrieval and data modules, which
pull trafilatura, pandas and friends -- none of which CI installs, since it
syncs the root project's dev extra. So bind a stub ``lib`` package to the eval
directory and import ``lib.llm`` through it: that resolves the module's
``from .retrieval import ...`` without ever executing the package __init__.
``llm.py`` and ``retrieval.py`` are stdlib-only at module scope, so this needs
nothing installed beyond pytest.

litellm itself is stubbed in sys.modules -- the SDK is imported inside
``_generate_litellm``, so these run with no network, no keys, and no litellm.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

_LIB = Path(__file__).parents[1] / "eval" / "lib"
if "lib" not in sys.modules:
    _pkg = types.ModuleType("lib")
    _pkg.__path__ = [str(_LIB)]
    sys.modules["lib"] = _pkg

from lib.llm import LLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_litellm_stub():
    """Keep the fake out of sys.modules for anything that runs after these."""
    yield
    sys.modules.pop("litellm", None)


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

    client = LLMClient(model="gemini/gemini-1.5-pro", use_litellm=True)

    # LiteLLM handles gemini itself, so the native GenAI branch must be disabled.
    assert client.is_gemini is False
    assert client.gemini_client is None


@pytest.mark.parametrize(
    ("model", "sends_temperature"),
    [
        ("anthropic/claude-opus-4-7", False),
        ("openai/gpt-5.4-pro", False),
        ("anthropic/claude-3-5-sonnet", True),
    ],
)
def test_reasoning_models_drop_temperature(model, sends_temperature):
    """The rule lives in one constant now; both backends must apply it.

    It was duplicated as a literal in the OpenAI and LiteLLM paths, so a newly
    released model could be taught to one and not the other.
    """
    fake = _stub_litellm("ok")

    client = LLMClient(model=model, use_litellm=True)
    asyncio.run(client.generate([{"role": "user", "content": "hi"}]))

    assert ("temperature" in fake.acompletion.calls[0]) is sends_temperature
