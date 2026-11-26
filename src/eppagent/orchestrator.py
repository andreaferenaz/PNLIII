"""Orquestación con LangGraph: Inspector → Normativo ⇄ Verificador → revisión humana → acción.

Grafo:

    START → inspector → normativo → verificador ─┬─(no aprobado y quedan rondas)→ normativo
                                                 └─(aprobado o sin rondas)→ decision
    decision ─┬─(cumple o sin notificación)→ END
              └─(no cumple)→ revision_humana (interrupt) → accion → END

- El bucle normativo ⇄ verificador es la autoevaluación (reflexión) con parada temprana.
- revision_humana usa interrupt() de LangGraph: el grafo queda en pausa hasta que una persona
  aprueba o rechaza la alerta (human-in-the-loop). El estado se guarda con un checkpointer.
"""
from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .actions.notifier import notify
from .agents.base import AgentMessage
from .agents.inspector import InspectorAgent
from .agents.normativo import NormativoAgent
from .agents.verificador import VerificadorAgent
from .config import get_config
from .llm import get_llm
from .logging_utils import get_logger

log = get_logger("orchestrator")


class EPPState(TypedDict, total=False):
    run_id: str
    zona: str
    image_path: str | None
    detecciones: list[dict] | None
    notificar: bool
    inspeccion: dict          # mensaje del inspector
    dictamen_msg: dict        # mensaje del normativo
    verificacion: dict        # mensaje del verificador
    feedback: str
    rondas: int
    vistos: list[str]
    decision_humana: dict
    accion: dict
    traza: Annotated[list[dict], operator.add]
    t0: float
    tokens0: int


def _msg(d: dict) -> AgentMessage:
    return AgentMessage(sender=d["from"], receiver=d["to"], content=d["content"], meta=d["meta"])


class Orchestrator:
    def __init__(self) -> None:
        cfg = get_config()
        self.inspector = InspectorAgent()
        self.normativo = NormativoAgent()
        self.verificador = VerificadorAgent()
        self.max_rounds = cfg["agents"]["max_reflection_rounds"]
        self.hitl = cfg["agents"].get("human_in_the_loop", True)
        self.checkpointer = InMemorySaver()
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------- nodos
    def _n_inspector(self, s: EPPState) -> dict:
        m = self.inspector.run(s["zona"], image_path=s.get("image_path"), detections=s.get("detecciones"))
        return {"inspeccion": m.as_dict(), "traza": [m.as_dict()], "rondas": 0, "vistos": [], "feedback": ""}

    def _n_normativo(self, s: EPPState) -> dict:
        m = self.normativo.run(_msg(s["inspeccion"]), feedback=s.get("feedback", ""))
        return {"dictamen_msg": m.as_dict(), "traza": [m.as_dict()]}

    def _n_verificador(self, s: EPPState) -> dict:
        v = self.verificador.run(_msg(s["dictamen_msg"]))
        fp = str(s["dictamen_msg"]["content"]["dictamen"])
        return {"verificacion": v.as_dict(), "traza": [v.as_dict()], "rondas": s.get("rondas", 0) + 1,
                "feedback": v.content["observaciones"], "vistos": s.get("vistos", []) + [fp]}

    def _route_verificador(self, s: EPPState) -> str:
        if s["verificacion"]["content"]["aprobado"]:
            return "decision"
        if s["rondas"] > self.max_rounds:
            return "decision"
        if len(s["vistos"]) != len(set(s["vistos"])):   # el normativo repite la respuesta
            log.warning("Dictamen repetido; se corta el bucle de reflexión.")
            return "decision"
        return "normativo"

    def _n_decision(self, s: EPPState) -> dict:
        """Guardrail final: si después de la reflexión el dictamen sigue sin cumplir las reglas,
        los campos 'cumple' y 'faltantes' se toman del control determinista y se deja constancia."""
        msg = s["dictamen_msg"]
        d = msg["content"]["dictamen"]
        insp = s["inspeccion"]["content"]
        if not self.verificador.rule_check(s["zona"], insp["estado"], d):
            return {}
        exigido = self.verificador.zones.get(s["zona"], {}).get("exige", [])
        esperados = sorted(e for e in exigido if insp["estado"].get(e) == "no_usa")
        if (set(d["faltantes"]) == set(esperados)) and d["cumple"] == (not esperados):
            return {}
        log.warning("Dictamen corregido por reglas: faltantes %s -> %s", d["faltantes"], esperados)
        d2 = {**d, "faltantes": esperados, "cumple": not esperados, "corregido_por_reglas": True,
              "faltantes_llm": d["faltantes"], "cumple_llm": d["cumple"]}
        return {"dictamen_msg": {**msg, "content": {**msg["content"], "dictamen": d2}}}

    def _route_decision(self, s: EPPState) -> str:
        d = s["dictamen_msg"]["content"]["dictamen"]
        if d["cumple"] or not s.get("notificar", True):
            return END
        return "revision_humana" if self.hitl else "accion"

    def _n_revision_humana(self, s: EPPState) -> dict:
        d = s["dictamen_msg"]["content"]["dictamen"]
        decision = interrupt({
            "pregunta": "¿Se envía la alerta de incumplimiento?",
            "zona": s["zona"], "faltantes": d["faltantes"], "dictamen": d["dictamen"],
        })
        return {"decision_humana": decision}

    def _route_revision(self, s: EPPState) -> str:
        return "accion" if s.get("decision_humana", {}).get("aprobar") else END

    def _n_accion(self, s: EPPState) -> dict:
        d = s["dictamen_msg"]["content"]["dictamen"]
        return {"accion": notify({"zona": s["zona"], "faltantes": d["faltantes"], "dictamen": d["dictamen"]})}

    def _build(self) -> StateGraph:
        g = StateGraph(EPPState)
        g.add_node("inspector", self._n_inspector)
        g.add_node("normativo", self._n_normativo)
        g.add_node("verificador", self._n_verificador)
        g.add_node("decision", self._n_decision)
        g.add_node("revision_humana", self._n_revision_humana)
        g.add_node("accion", self._n_accion)
        g.add_edge(START, "inspector")
        g.add_edge("inspector", "normativo")
        g.add_edge("normativo", "verificador")
        g.add_conditional_edges("verificador", self._route_verificador, ["normativo", "decision"])
        g.add_conditional_edges("decision", self._route_decision, ["revision_humana", "accion", END])
        g.add_conditional_edges("revision_humana", self._route_revision, ["accion", END])
        g.add_edge("accion", END)
        return g

    # ------------------------------------------------------------- API pública
    def _result(self, thread_id: str, state: dict, interrupted: dict | None) -> dict:
        d = state["dictamen_msg"]["content"]["dictamen"]
        v = state["verificacion"]["content"]
        usage = get_llm().usage
        res = {"run_id": thread_id, "zona": state["zona"], "estado": state["inspeccion"]["content"]["estado"],
               "cumple": d["cumple"], "faltantes": d["faltantes"], "dictamen": d["dictamen"],
               "fuentes": d["fuentes"], "verificado": v["aprobado"],
               "corregido_por_reglas": d.get("corregido_por_reglas", False),
               "faltantes_llm": d.get("faltantes_llm", d["faltantes"]), "cumple_llm": d.get("cumple_llm", d["cumple"]),
               "observaciones_verificador": v["observaciones"], "rondas_reflexion": state["rondas"],
               "metricas": {"latencia_ms": round((time.perf_counter() - state["t0"]) * 1000, 1),
                            "tokens": usage.input_tokens + usage.output_tokens - state["tokens0"],
                            "retrieved_ids": state["dictamen_msg"]["meta"]["retrieved_ids"]},
               "traza": state["traza"]}
        if interrupted is not None:
            res["estado_alerta"] = "pendiente_aprobacion"
            res["revision_humana"] = interrupted
        elif "accion" in state:
            res["estado_alerta"] = "enviada"
            res["accion"] = state["accion"]
        elif "decision_humana" in state:
            res["estado_alerta"] = "rechazada"
        return res

    def _run(self, payload: Any, thread_id: str) -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        out = self.graph.invoke(payload, cfg)
        intr = out.get("__interrupt__")
        state = self.graph.get_state(cfg).values
        return self._result(thread_id, state, intr[0].value if intr else None)

    def inspect(self, zone: str, image_path=None, detections=None, notify_on_fail: bool = True) -> dict:
        thread_id = uuid.uuid4().hex[:12]
        u = get_llm().usage
        payload: EPPState = {"run_id": thread_id, "zona": zone,
                             "image_path": str(image_path) if image_path else None,
                             "detecciones": detections, "notificar": notify_on_fail, "traza": [],
                             "t0": time.perf_counter(), "tokens0": u.input_tokens + u.output_tokens}
        return self._run(payload, thread_id)

    def resume(self, thread_id: str, aprobar: bool, comentario: str = "") -> dict:
        """Retoma un grafo pausado en la revisión humana."""
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = self.graph.get_state(cfg)
        if not snap.next or "revision_humana" not in snap.next:
            raise KeyError("No hay una revisión pendiente para ese run_id.")
        return self._run(Command(resume={"aprobar": aprobar, "comentario": comentario}), thread_id)

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()


_orch: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch
