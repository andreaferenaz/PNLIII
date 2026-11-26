"""Agente 1 – Inspector visual: ejecuta el detector y arma el estado de cada EPP."""
from __future__ import annotations

from pathlib import Path

from ..logging_utils import get_logger, timer
from ..vision import detections_to_state, get_detector
from .base import AgentMessage

log = get_logger("agent.inspector")


class InspectorAgent:
    name = "inspector"

    def run(self, zone: str, image_path: str | Path | None = None,
            detections: list[dict] | None = None) -> AgentMessage:
        t: dict = {}
        backend = "provisto" if detections is not None else get_detector().backend
        with timer(t, "inference_ms"):
            if detections is None:
                if image_path is None:
                    raise ValueError("Se necesita una imagen o una lista de detecciones.")
                detections = get_detector().detect(image_path)
        state = detections_to_state(detections)
        log.info("Zona=%s estado=%s", zone, state)
        return AgentMessage(
            sender=self.name, receiver="normativo",
            content={"zona": zone, "estado": state, "detecciones": detections},
            meta={**t, "backend": backend},
        )
