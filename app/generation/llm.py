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
