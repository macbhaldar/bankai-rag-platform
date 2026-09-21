"""LLM providers: OpenAI-compatible, Ollama, and an offline extractive fallback."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.hybrid import RetrievalHit

logger = get_logger(__name__)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9]+")
 
 
class LLMError(RuntimeError):
    pass
 
 
@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    mode: str  # "generative" | "extractive"
    latency_ms: float = 0.0
 
 
class BaseLLM(ABC):
    provider: str = "base"
    model: str = ""
    mode: str = "generative"
 
    @abstractmethod
    def generate(self, question: str, messages: list[dict], hits: list[RetrievalHit]) -> LLMResult: ...
 
 
class OpenAICompatLLM(BaseLLM):
    """Works with OpenAI, Azure OpenAI gateways, LM Studio, llama.cpp server, Groq, ..."""
 
    provider = "openai"
 
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()
 
    def generate(self, question: str, messages: list[dict], hits: list[RetrievalHit]) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=180,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"] or ""
        except requests.RequestException as exc:
            raise LLMError(f"OpenAI-compatible endpoint failed: {exc}") from exc
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            mode="generative",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    class OllamaLLM(BaseLLM):
    provider = "ollama"
 
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()
 
    def generate(self, question: str, messages: list[dict], hits: list[RetrievalHit]) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
                },
                timeout=300,
            )
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "")
        except requests.RequestException as exc:
            raise LLMError(f"Ollama failed: {exc}") from exc
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            mode="generative",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
 
 
class ExtractiveLLM(BaseLLM):
    """Zero-dependency fallback: cite-worthy sentences selected by keyword overlap."""
 
    provider = "extractive"
    model = "keyword-extractive-v1"
    mode = "extractive"
 
    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}
 
    def generate(self, question: str, messages: list[dict], hits: list[RetrievalHit]) -> LLMResult:
        started = time.perf_counter()
        query_tokens = self._content_tokens(question)
        numbered_tokens = {n for n in re.findall(r"\d+(?:\.\d+)?", question)}
        best: list[tuple[float, int, int, str]] = []  # (score, hit_index, sentence_index, sentence)
        for h_idx, hit in enumerate(hits[:8]):
            body = hit.text
            # skip the context header line ("Title — Section")
            body = body.split("\n", 1)[1] if "\n" in body else body
            for s_idx, sentence in enumerate(_SENTENCE_SPLIT.split(body)):
                tokens = self._content_tokens(sentence)
                if not tokens:
                    continue
                overlap = len(query_tokens & tokens) / math_sqrt(len(tokens) or 1)
                digits = set(re.findall(r"\d+(?:\.\d+)?", sentence))
                if numbered_tokens & digits:
                    overlap += 0.35
                if overlap > 0:
                    best.append((overlap, h_idx, s_idx, sentence.strip()))
        best.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
 
        picked: list[tuple[int, int, str]] = []
        seen_hits: set[int] = set()
        for score, h_idx, s_idx, sentence in best:
            if len(picked) >= 4:
                break
            if h_idx in seen_hits and len(picked) >= 2:
                continue  # prefer diversity across documents once we have a base
            picked.append((h_idx, s_idx, sentence))
            seen_hits.add(h_idx)
        # restore document order for readability
        picked.sort(key=lambda item: (item[0], item[1]))
        if not picked:
            text = f"No passage in the indexed documents matches the question: {question}"
        else:
            lines = [f"[{h_idx + 1}] {sentence}" for h_idx, _, sentence in picked]
            text = "Based on the retrieved bank documents:\n\n" + "\n\n".join(lines)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            mode="extractive",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
 
 
def math_sqrt(x: float) -> float:
    return x**0.5
 
 
def probe_ollama(url: str, timeout: float = 1.5) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False
 
 
def resolve_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm.provider.lower()
    if provider == "openai":
        return OpenAICompatLLM(
            settings.llm.base_url, settings.llm.api_key, settings.llm.model,
            settings.llm.temperature, settings.llm.max_tokens,
        )
    if provider == "ollama":
        return OllamaLLM(
            settings.llm.ollama_url, settings.llm.model,
            settings.llm.temperature, settings.llm.max_tokens,
        )
    if provider == "extractive":
        logger.info("LLM mode: extractive (offline fallback, no external services)")
        return ExtractiveLLM()
    if provider == "auto":
        if settings.llm.api_key:
            logger.info("LLM mode: openai-compatible (%s)", settings.llm.model)
            return OpenAICompatLLM(
                settings.llm.base_url, settings.llm.api_key, settings.llm.model,
                settings.llm.temperature, settings.llm.max_tokens,
            )
        if probe_ollama(settings.llm.ollama_url):
            logger.info("LLM mode: ollama (%s)", settings.llm.model)
            return OllamaLLM(
                settings.llm.ollama_url, settings.llm.model,
                settings.llm.temperature, settings.llm.max_tokens,
            )
        logger.info(
            "LLM mode: extractive (no OPENAI/LLM API key and Ollama not reachable at %s)", settings.llm.ollama_url
        )
        return ExtractiveLLM()
    raise LLMError(f"unknown llm provider: {provider!r}")