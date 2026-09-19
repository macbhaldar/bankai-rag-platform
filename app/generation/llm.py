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

    