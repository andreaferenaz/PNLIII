"""Inferencia con un detector de EPP ya entrenado (YOLO de Hugging Face).

El modelo por defecto (keremberke/yolov8n-protective-equipment-detection) detecta
clases como helmet / no_helmet, glove / no_glove, goggles / no_goggles, etc. Acá se
traducen a un estado por elemento: usa | no_usa | no_visible.

Modo mock: si no hay modelo o se configura VISION_BACKEND=mock, las detecciones se
leen de un JSON con el mismo nombre que la imagen (data/samples/*.json) o se reciben
directamente en la API.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import get_config
from ..logging_utils import get_logger

log = get_logger("vision")

EPP_MAP = {
    "helmet": ("casco", "usa"), "no_helmet": ("casco", "no_usa"),
    "glove": ("guantes", "usa"), "no_glove": ("guantes", "no_usa"),
    "goggles": ("anteojos", "usa"), "no_goggles": ("anteojos", "no_usa"),
    "mask": ("barbijo", "usa"), "no_mask": ("barbijo", "no_usa"),
    "shoes": ("calzado", "usa"), "no_shoes": ("calzado", "no_usa"),
    "vest": ("chaleco", "usa"), "no_vest": ("chaleco", "no_usa"),
}
ELEMENTS = ["casco", "chaleco", "anteojos", "guantes", "calzado", "barbijo", "protector_auditivo"]


def detections_to_state(detections: list[dict]) -> dict[str, str]:
    """Agrega detecciones en un estado por elemento. 'no_usa' prevalece sobre 'usa'."""
    state = {e: "no_visible" for e in ELEMENTS}
    for d in detections:
        label = d["label"].lower().replace("-", "_").replace(" ", "_")
        if label in EPP_MAP:
            elem, st = EPP_MAP[label]
        elif label in ELEMENTS or label.startswith("no_"):
            elem = label.removeprefix("no_")
            st = "no_usa" if label.startswith("no_") else "usa"
        else:
            continue
        if state.get(elem) != "no_usa":
            state[elem] = st
    return state


class EPPDetector:
    def __init__(self) -> None:
        self.cfg = get_config()["vision"]
        self.backend = self.cfg["backend"]
        self.model = None
        if self.backend == "yolo":
            try:
                from huggingface_hub import hf_hub_download
                from ultralytics import YOLO

                weights = hf_hub_download(self.cfg["hf_repo"], self.cfg["hf_file"])
                self.model = YOLO(weights)
                log.info("Detector YOLO cargado: %s", self.cfg["hf_repo"])
            except Exception as e:
                log.warning("No se pudo cargar YOLO (%s). Se usa el modo mock.", e)
                self.backend = "mock"

    def detect(self, image_path: str | Path) -> list[dict]:
        image_path = Path(image_path)
        if self.backend == "yolo" and self.model is not None:
            res = self.model.predict(str(image_path), conf=self.cfg["conf_threshold"], verbose=False)[0]
            names = res.names
            return [{"label": names[int(b.cls)], "conf": round(float(b.conf), 3),
                     "box": [round(float(x), 1) for x in b.xyxy[0].tolist()]} for b in res.boxes]
        sidecar = image_path.with_suffix(".json")
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))["detections"]
        log.warning("Modo mock sin JSON para %s: sin detecciones.", image_path.name)
        return []


_detector: EPPDetector | None = None


def get_detector() -> EPPDetector:
    global _detector
    if _detector is None:
        _detector = EPPDetector()
    return _detector
