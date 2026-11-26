"""Mensajes estructurados entre agentes y utilidades comunes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT


@dataclass
class AgentMessage:
    """Mensaje que viaja entre agentes: emisor, receptor, contenido y metadatos."""
    sender: str
    receiver: str
    content: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"from": self.sender, "to": self.receiver, "content": self.content, "meta": self.meta}


def load_zones() -> dict:
    with open(ROOT / "configs" / "zonas.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["zonas"]
