"""Logging estructurado (niveles INFO/DEBUG/ERROR) y medición de latencia."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timer(store: dict, key: str):
    """Guarda en store[key] la duración del bloque en milisegundos."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - t0) * 1000, 1)
