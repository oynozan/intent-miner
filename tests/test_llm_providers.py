"""Provider selection and fallback, without touching the network.

The per-provider callers (``_openai_json`` / ``_anthropic_json``) are monkeypatched, so
these tests exercise the ordering and fallback logic -- which is the new, fallible part
-- rather than either vendor's API.
"""

from __future__ import annotations

import pytest

from llm import providers


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """settings() is lru_cached; clear it so each test's monkeypatched env is read fresh."""
    from core.config import settings

    settings.cache_clear()
    yield
    settings.cache_clear()


PARAMS = {
    "openai": {"model": "gpt-5-nano", "effort": "high", "max_tokens": 100},
    "anthropic": {"model": "claude-opus-4-8", "effort": "high", "max_tokens": 100},
}


def _call() -> dict:
    return providers.complete_json(
        system="s", user="u", schema={"type": "object"}, schema_name="t", params=PARAMS
    )


def test_openai_is_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    called: list[str] = []
    monkeypatch.setattr(providers, "_openai_json", lambda **k: called.append("openai") or {"ok": "openai"})
    monkeypatch.setattr(providers, "_anthropic_json", lambda **k: called.append("anthropic") or {"ok": "anthropic"})

    assert _call() == {"ok": "openai"}
    assert called == ["openai"], "anthropic must not be called when openai succeeds"


def test_falls_back_to_anthropic_when_openai_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    def boom(**_):
        raise RuntimeError("openai 500")

    called: list[str] = []
    monkeypatch.setattr(providers, "_openai_json", boom)
    monkeypatch.setattr(providers, "_anthropic_json", lambda **k: called.append("anthropic") or {"ok": "anthropic"})

    assert _call() == {"ok": "anthropic"}
    assert called == ["anthropic"]


def test_skips_unconfigured_primary_and_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OpenAI key at all -> straight to Anthropic, no attempt logged against OpenAI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    def must_not_run(**_):
        raise AssertionError("openai caller must not run without a key")

    monkeypatch.setattr(providers, "_openai_json", must_not_run)
    monkeypatch.setattr(providers, "_anthropic_json", lambda **k: {"ok": "anthropic"})

    assert _call() == {"ok": "anthropic"}


def test_llm_provider_env_flips_the_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    called: list[str] = []
    monkeypatch.setattr(providers, "_openai_json", lambda **k: called.append("openai") or {"ok": "openai"})
    monkeypatch.setattr(providers, "_anthropic_json", lambda **k: called.append("anthropic") or {"ok": "anthropic"})

    assert _call() == {"ok": "anthropic"}
    assert called == ["anthropic"], "LLM_PROVIDER=anthropic must make anthropic primary"


def test_no_provider_configured_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no LLM provider configured"):
        _call()


def test_all_providers_failing_surfaces_the_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    def boom_openai(**_):
        raise RuntimeError("openai down")

    def boom_anthropic(**_):
        raise RuntimeError("anthropic down")

    monkeypatch.setattr(providers, "_openai_json", boom_openai)
    monkeypatch.setattr(providers, "_anthropic_json", boom_anthropic)

    with pytest.raises(RuntimeError, match="all LLM providers failed"):
        _call()


# --- embeddings order (same shape, separate module): Voyage primary, OpenAI fallback ---

def test_embeddings_default_to_voyage_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)  # exercise the default
    from core.config import settings

    settings.cache_clear()
    from llm import embeddings

    assert embeddings._order()[0] == "voyage"


def test_embeddings_fall_back_to_openai_when_voyage_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    from llm import embeddings

    called: list[str] = []
    monkeypatch.setattr(embeddings, "_voyage_embed", lambda *a: (_ for _ in ()).throw(AssertionError("no key")))
    monkeypatch.setattr(embeddings, "_openai_embed", lambda t, i, b: called.append("openai") or [[0.0]] * len(t))

    assert embeddings.embed(["a", "b"]) == [[0.0], [0.0]]
    assert called == ["openai"], "with no Voyage key, embeddings must fall back to OpenAI"


def test_embed_with_provider_reports_which_served(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefilter needs to know the provider so it can lock leaves to the same one."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-x")
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    from llm import embeddings

    monkeypatch.setattr(embeddings, "_voyage_embed", lambda t, i, b: [[1.0]] * len(t))
    vecs, provider = embeddings.embed_with_provider(["a"], input_type="document")
    assert provider == "voyage"
    assert vecs == [[1.0]]


def test_forced_provider_does_not_fall_back_to_a_different_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locking to a provider must RAISE on failure, never silently switch vector spaces --
    that is the whole point of the lock (leaf/candidate must share a space)."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from llm import embeddings

    monkeypatch.setattr(embeddings, "_voyage_embed", lambda *a: (_ for _ in ()).throw(RuntimeError("voyage rate limit")))
    monkeypatch.setattr(embeddings, "_openai_embed", lambda *a: pytest.fail("must not fall back when a provider is forced"))

    with pytest.raises(RuntimeError, match="voyage rate limit"):
        embeddings.embed(["a"], provider="voyage")


def test_forced_but_unconfigured_provider_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    from llm import embeddings

    with pytest.raises(RuntimeError, match="locked for this run but not configured"):
        embeddings.embed(["a"], provider="voyage")
