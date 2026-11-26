"""Carga de configuración desde configs/config.yaml con pisado por variables de entorno."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("EPP_CONFIG", ROOT / "configs" / "config.yaml"))

try:  # .env opcional para uso local
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _auto_provider(configured: str) -> str:
    """Si el proveedor elegido no tiene clave, cae a otro disponible o a mock."""
    forced = os.getenv("LLM_PROVIDER")
    if forced:
        return forced
    if configured == "groq" and os.getenv("GROQ_API_KEY"):
        return "groq"
    if configured == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if configured == "openai" and os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"]["provider"] = _auto_provider(cfg["llm"]["provider"])
    if os.getenv("VISION_BACKEND"):
        cfg["vision"]["backend"] = os.environ["VISION_BACKEND"]
    if os.getenv("HITL"):
        cfg["agents"]["human_in_the_loop"] = os.environ["HITL"].lower() in ("1", "true", "si", "sí")
    if os.getenv("PROMPT_GUARD"):
        cfg["security"]["prompt_guard"]["enabled"] = os.environ["PROMPT_GUARD"].lower() in ("1", "true")
    if os.getenv("NOTIFIER"):
        cfg["actions"]["notifier"] = os.environ["NOTIFIER"]
    # rutas relativas a la raíz del repo
    for section, key in [("rag", "corpus_dir"), ("rag", "index_dir"), ("llm", "cache_path"),
                         ("security", "audit_log")]:
        cfg[section][key] = str(ROOT / cfg[section][key])
    return cfg
