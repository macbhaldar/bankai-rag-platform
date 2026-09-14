"""Logging setup: console + rotating file handler under data/logs/."""
from __future__ import annotations
 
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
 
_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
 
 
def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    # Third-party noise reduction
    for noisy in ("chromadb", "urllib3", "httpx", "httpcore", "sentence_transformers", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
 
 
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)