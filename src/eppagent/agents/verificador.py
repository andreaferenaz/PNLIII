"""Agente 3 – Verificador: revisa el dictamen con reglas (zonas.yaml) y con el LLM.
Si encuentra inconsistencias, devuelve observaciones al agente normativo (reflexión)."""
from __future__ import annotations

import json

from ..llm import get_llm
from ..logging_utils import get_logger, timer
from ..security import Verificacion, parse_json_output
from .base import AgentMessage, load_zones

log = get_logger("agent.verificador")

PROMPT = """ROL: VERIFICADOR
Revisá si el dictamen es coherente con el estado detectado y con el EPP exigido.
EPP exigido en la zona: {exigido}
Estado detectado: {estado}
Dictamen: {dictamen}

Respondé SOLO un JSON: {{"aprobado": true|false, "observaciones": "qué corregir"}}
"""


class VerificadorAgent:
    name = "verificador"

    def __init__(self) -> None:
        self.zones = load_zones()

    def rule_check(self, zona: str, estado: dict, dictamen: dict) -> list[str]:
        """Control determinista: faltantes esperados = exigidos que figuran como no_usa."""
        exigido = set(self.zones.get(zona, {}).get("exige", []))
        esperados = {e for e in exigido if estado.get(e) == "no_usa"}
        informados = set(dictamen.get("faltantes", []))
        issues = []
        if esperados - informados:
            issues.append(f"Faltan en el dictamen: {sorted(esperados - informados)}")
        if informados - esperados:
            issues.append(f"Se informaron como faltantes elementos no exigidos o no detectados: "
                          f"{sorted(informados - esperados)}")
        if dictamen.get("cumple") != (not esperados):
            issues.append("El campo 'cumple' no es coherente con los faltantes.")
        if not dictamen.get("fuentes"):
            issues.append("El dictamen no cita fuentes del contexto.")
        return issues

    def run(self, msg: AgentMessage) -> AgentMessage:
        zona, estado, dictamen = msg.content["zona"], msg.content["estado"], msg.content["dictamen"]
        t: dict = {}
        issues = self.rule_check(zona, estado, dictamen)
        with timer(t, "llm_ms"):
            raw = get_llm().complete(PROMPT.format(
                exigido=self.zones.get(zona, {}).get("exige", []),
                estado=json.dumps(estado, ensure_ascii=False),
                dictamen=json.dumps(dictamen, ensure_ascii=False)), size="small")
        llm_v = parse_json_output(raw, Verificacion)
        aprobado = llm_v.aprobado and not issues
        obs = "; ".join(issues + ([llm_v.observaciones] if not llm_v.aprobado else []))
        return AgentMessage(
            sender=self.name, receiver="orquestador" if aprobado else "normativo",
            content={"aprobado": aprobado, "observaciones": obs, "reglas": issues},
            meta=t,
        )
