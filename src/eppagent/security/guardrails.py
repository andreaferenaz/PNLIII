"""Medidas de seguridad: control de acceso, filtrado de entradas, validación de salidas
y auditoría (OWASP Top 10 LLM: LLM01 inyección de prompts, LLM02 divulgación de datos,
LLM05 manejo inseguro de salidas)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..config import get_config

# ---------------------------------------------------------------- acceso
def check_api_key(provided: str | None) -> bool:
    if not get_config()["security"]["require_api_key"]:
        return True
    expected = os.getenv("APP_API_KEY", "demo-key")
    return bool(provided) and secrets.compare_digest(provided, expected)


# ---------------------------------------------------------------- entradas
INJECTION_PATTERNS = [
    r"ignor(a|á|e|ar)\s+(todas?\s+)?(las\s+)?(instrucciones|reglas)",
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(revel|mostr|dec)\w*\s+(el\s+)?(prompt|system\s*prompt|instrucciones\s+del\s+sistema)",
    r"system\s*prompt",
    r"act(ú|u)a\s+como|you\s+are\s+now|jailbreak|DAN\b",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"(api[_ -]?key|contraseña|password|token)\s*(de|del)?\s*(sistema|admin)",
]
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "dni": r"\b\d{2}\.?\d{3}\.?\d{3}\b",
    "telefono": r"\b(?:\+?54\s?)?(?:11|15)[\s-]?\d{4}[\s-]?\d{4}\b",
}


class InputRejected(ValueError):
    pass


class PromptGuard:
    """Clasificador de inyección de prompts de Meta (Llama Prompt Guard 2), como en la Teoría 2.

    Es opcional: si no está habilitado o no se puede cargar (modelo con acceso restringido),
    quedan activas solo las reglas por expresiones regulares.
    """

    def __init__(self) -> None:
        cfg = get_config()["security"]["prompt_guard"]
        self.enabled = cfg["enabled"]
        self.threshold = cfg["threshold"]
        self.pipe = None
        if self.enabled:
            try:
                from transformers import pipeline

                self.pipe = pipeline("text-classification", model=cfg["model"],
                                     token=os.getenv("HF_TOKEN"), truncation=True)
            except Exception as e:  # sin acceso al modelo o sin transformers
                from ..logging_utils import get_logger

                get_logger("security").warning("Prompt Guard no disponible (%s); solo reglas.", e)
                self.enabled = False

    def score(self, text: str) -> float | None:
        """Probabilidad de que el texto sea malicioso (None si el clasificador no está activo)."""
        if not self.enabled or self.pipe is None:
            return None
        out = self.pipe(text)[0]
        label = str(out["label"]).upper()
        malicious = label in ("LABEL_1", "MALICIOUS", "INJECTION", "JAILBREAK")
        return float(out["score"]) if malicious else 1.0 - float(out["score"])


_guard: PromptGuard | None = None


def get_prompt_guard() -> PromptGuard:
    global _guard
    if _guard is None:
        _guard = PromptGuard()
    return _guard


def sanitize_question(text: str) -> tuple[str, list[str]]:
    """Valida longitud, bloquea inyecciones (reglas + Prompt Guard) y enmascara datos personales."""
    cfg = get_config()["security"]
    flags: list[str] = []
    if not text or not text.strip():
        raise InputRejected("La pregunta está vacía.")
    if len(text) > cfg["max_question_chars"]:
        raise InputRejected(f"La pregunta supera {cfg['max_question_chars']} caracteres.")
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            raise InputRejected("Se detectó un posible intento de inyección de prompt.")
    guard = get_prompt_guard()
    score = guard.score(text)
    if score is not None:
        flags.append(f"prompt_guard={score:.2f}")
        if score >= guard.threshold:
            raise InputRejected("Prompt Guard clasificó la entrada como inyección de prompt.")
    clean = text
    for name, pat in PII_PATTERNS.items():
        if re.search(pat, clean):
            flags.append(f"pii_{name}")
            clean = re.sub(pat, f"[{name.upper()}]", clean)
    return clean.strip(), flags


def check_image(data: bytes, filename: str) -> None:
    cfg = get_config()["security"]
    if len(data) > cfg["max_image_mb"] * 1024 * 1024:
        raise InputRejected("La imagen supera el tamaño máximo.")
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise InputRejected("Formato de imagen no admitido (jpg o png).")
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        raise InputRejected("El archivo no es una imagen válida.")


# ---------------------------------------------------------------- salidas
class Dictamen(BaseModel):
    cumple: bool
    faltantes: list[str] = Field(default_factory=list)
    dictamen: str = Field(min_length=5, max_length=1500)
    fuentes: list[str] = Field(default_factory=list)


class Verificacion(BaseModel):
    aprobado: bool
    observaciones: str = ""


class Respuesta(BaseModel):
    respuesta: str = Field(min_length=1, max_length=2000)
    fuentes: list[str] = Field(default_factory=list)


def parse_json_output(text: str, model: type[BaseModel]) -> BaseModel:
    """Extrae el primer objeto JSON de la salida del LLM y lo valida con el esquema."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("La salida del modelo no contiene JSON.")
    try:
        return model.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Salida inválida: {e}") from e


def check_citations(fuentes: list[str], allowed_ids: list[str]) -> list[str]:
    """Descarta fuentes que no estaban en el contexto recuperado (evita citas inventadas)."""
    allowed = set(allowed_ids) | {a.split("#")[0] for a in allowed_ids}
    return [f for f in fuentes if f in allowed]


# ---------------------------------------------------------------- auditoría
def audit(event: str, payload: dict) -> None:
    """Registro append-only. La pregunta se guarda como hash, no en texto plano."""
    path = Path(get_config()["security"]["audit_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
    for k, v in payload.items():
        rec[k] = hashlib.sha256(v.encode()).hexdigest()[:16] if k == "question" else v
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
