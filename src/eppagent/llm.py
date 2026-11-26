"""Cliente LLM con selección dinámica de modelo, caché y conteo de tokens.

Proveedores: groq, anthropic, openai y mock. El modo mock no necesita clave ni internet y
devuelve respuestas deterministas, útil para probar el flujo y correr los tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_config
from .logging_utils import get_logger

log = get_logger("llm")


@dataclass
class LLMUsage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_model": self.by_model,
        }


class LLMClient:
    def __init__(self) -> None:
        cfg = get_config()["llm"]
        self.provider = cfg["provider"]
        self.models = cfg["models"].get(self.provider, {"small": "mock-small", "large": "mock-large"})
        self.max_tokens = cfg["max_tokens"]
        self.temperature = cfg["temperature"]
        self.use_cache = cfg["cache"]
        self.cache_path = Path(cfg["cache_path"])
        self._cache = self._load_cache()
        self._lock = threading.Lock()
        self.usage = LLMUsage()
        self._client = None
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.provider == "groq":   # API compatible con OpenAI, con clave gratuita
            import openai

            self._client = openai.OpenAI(api_key=os.environ["GROQ_API_KEY"],
                                         base_url="https://api.groq.com/openai/v1")
        log.info("Proveedor LLM: %s (modelos %s)", self.provider, self.models)

    # ------------------------------------------------------------------ caché
    def _load_cache(self) -> dict:
        if self.use_cache and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.use_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _key(model: str, system: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()

    # ------------------------------------------------------------------ API
    def complete(self, prompt: str, system: str = "", size: str = "small",
                 max_tokens: int | None = None) -> str:
        """size='small' para clasificar/verificar, 'large' para redactar."""
        model = self.models[size]
        key = self._key(model, system, prompt)
        with self._lock:
            if self.use_cache and key in self._cache:
                self.usage.cache_hits += 1
                return self._cache[key]
        text, tin, tout = self._call(model, system, prompt, max_tokens or self.max_tokens)
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += tin
            self.usage.output_tokens += tout
            m = self.usage.by_model.setdefault(model, {"calls": 0, "tokens": 0})
            m["calls"] += 1
            m["tokens"] += tin + tout
            if self.use_cache:
                self._cache[key] = text
                self._save_cache()
        return text

    def _call(self, model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            r = self._client.messages.create(
                model=model, max_tokens=max_tokens, temperature=self.temperature,
                system=system or "Sos un asistente técnico preciso.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens
        if self.provider in ("openai", "groq"):
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}]
            extra = {"extra_body": {"reasoning_effort": "low"}} if self.provider == "groq" else {}
            r = self._client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens + (800 if extra else 0),
                temperature=self.temperature, **extra)
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
        return self._mock(prompt), len(prompt) // 4, 60

    # ------------------------------------------------------------------ mock
    @staticmethod
    def _mock(prompt: str) -> str:
        """Respuestas deterministas que respetan el formato JSON pedido por cada agente."""
        if "ROL: NORMATIVO" in prompt:
            ctx = prompt.split("CONTEXTO:", 1)[-1]
            fuentes = sorted(set(part.split("]")[0] for part in ctx.split("[F:")[1:]))[:3]
            faltantes = []
            if "ESTADO DETECTADO:" in prompt:
                estado = prompt.split("ESTADO DETECTADO:", 1)[1].split("\n\n")[0]
                faltantes = [l.split(":")[0].strip("- ").strip() for l in estado.splitlines()
                             if "no_usa" in l]
            fb = prompt.split("OBSERVACIONES DEL VERIFICADOR A CORREGIR:", 1)
            if len(fb) > 1 and "no exigidos o no detectados:" in fb[1]:
                quitar = fb[1].split("no exigidos o no detectados:", 1)[1].split("]")[0]
                faltantes = [f for f in faltantes if f"'{f}'" not in quitar]
            cumple = not faltantes
            return json.dumps({
                "cumple": cumple,
                "faltantes": faltantes,
                "dictamen": ("Cumple con el EPP exigido en la zona." if cumple else
                             "No cumple: falta " + ", ".join(faltantes) +
                             ". Según la política de la planta, debe usarse antes de ingresar."),
                "fuentes": fuentes,
            }, ensure_ascii=False)
        if "ROL: VERIFICADOR" in prompt:
            return json.dumps({"aprobado": True, "observaciones": "Dictamen coherente con las detecciones y las fuentes."},
                              ensure_ascii=False)
        if "ROL: RESPUESTA" in prompt:
            ctx = prompt.split("CONTEXTO:", 1)[-1]
            first = ctx.split("[F:", 1)[-1]
            src = first.split("]")[0]
            lines = [l for l in first.split("]", 1)[-1].strip().split("\n")
                     if l.strip() and not l.startswith(("(", "#"))]
            body = (lines[0] if lines else "No tengo esa información.")[:300]
            return json.dumps({"respuesta": body, "fuentes": [src]}, ensure_ascii=False)
        return "{}"


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
