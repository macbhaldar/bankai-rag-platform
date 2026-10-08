"""LLM provider tests that do not require external model services."""

from app.generation.llm import ExtractiveLLM, OllamaLLM, resolve_llm


def test_ollama_is_top_level_provider():
    provider = OllamaLLM("http://127.0.0.1:11434", "llama3.1", 0.1, 100)
    assert provider.provider == "ollama"
    assert provider.mode == "generative"


def test_extractive_provider_returns_citations(services):
    llm = ExtractiveLLM()
    result = llm.generate(
        "What requires dual approval?",
        [],
        services.retriever.retrieve("dual approval", k=3),
    )
    assert result.mode == "extractive"
    assert result.text


def test_resolve_explicit_extractive(services):
    assert resolve_llm(services.settings).provider == "extractive"
