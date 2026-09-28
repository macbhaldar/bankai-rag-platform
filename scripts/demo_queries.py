"""End-to-end demo: semantic search, RAG QA with citations, structured text-to-SQL.

Usage: python scripts/demo_queries.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.container import get_services

