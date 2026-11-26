"""API FastAPI que orquesta usuarios, agentes, RAG y modelo de visión.

Ejecutar:  uvicorn eppagent.api.main:app --reload --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from ..agents.base import load_zones
from ..agents.consultor import ConsultorAgent
from ..config import get_config
from ..llm import get_llm
from ..orchestrator import get_orchestrator
from ..rag import get_index
from ..security import InputRejected, audit, check_api_key, check_image, sanitize_question

app = FastAPI(title="Asistente de cumplimiento de EPP",
              description="POC multiagente con RAG – PLN III", version="0.1.0")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not check_api_key(x_api_key):
        audit("acceso_denegado", {})
        raise HTTPException(status_code=401, detail="API key inválida o ausente (header X-API-Key).")


class AskIn(BaseModel):
    pregunta: str


class InspectJSONIn(BaseModel):
    zona: str
    detecciones: list[dict]


class DecisionIn(BaseModel):
    aprobar: bool
    comentario: str = ""


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    return {"status": "ok", "llm": cfg["llm"]["provider"], "vision": cfg["vision"]["backend"],
            "embeddings": get_index().encoder_kind}


@app.get("/zonas", dependencies=[Depends(require_key)])
def zonas() -> dict:
    return load_zones()


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(body: AskIn) -> dict:
    try:
        q, flags = sanitize_question(body.pregunta)
    except InputRejected as e:
        audit("entrada_rechazada", {"motivo": str(e), "question": body.pregunta})
        raise HTTPException(status_code=400, detail=str(e))
    out = ConsultorAgent().run(q)
    audit("consulta", {"question": q, "flags": flags, "fuentes": out["fuentes"]})
    return {**out, "flags": flags}


@app.post("/inspect", dependencies=[Depends(require_key)])
async def inspect(zona: str = Form(...), imagen: UploadFile = File(...)) -> dict:
    if zona not in load_zones():
        raise HTTPException(status_code=400, detail="Zona desconocida.")
    data = await imagen.read()
    try:
        check_image(data, imagen.filename or "")
    except InputRejected as e:
        audit("imagen_rechazada", {"motivo": str(e)})
        raise HTTPException(status_code=400, detail=str(e))
    with tempfile.TemporaryDirectory() as tmp:   # la imagen no se conserva (privacidad)
        path = Path(tmp) / Path(imagen.filename).name
        path.write_bytes(data)
        sidecar = Path(get_config()["rag"]["corpus_dir"]).parents[0] / "samples" / (path.stem + ".json")
        if sidecar.exists():                     # modo mock con imágenes de ejemplo
            (Path(tmp) / sidecar.name).write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")
        result = get_orchestrator().inspect(zona, image_path=path)
    audit("inspeccion", {"zona": zona, "cumple": result["cumple"], "faltantes": result["faltantes"]})
    return result


@app.post("/inspect/json", dependencies=[Depends(require_key)])
def inspect_json(body: InspectJSONIn) -> dict:
    """Variante sin imagen: recibe las detecciones (útil para pruebas y evaluación)."""
    if body.zona not in load_zones():
        raise HTTPException(status_code=400, detail="Zona desconocida.")
    result = get_orchestrator().inspect(body.zona, detections=body.detecciones)
    audit("inspeccion", {"zona": body.zona, "cumple": result["cumple"], "faltantes": result["faltantes"]})
    return result


@app.post("/inspect/{run_id}/decision", dependencies=[Depends(require_key)])
def decision(run_id: str, body: DecisionIn) -> dict:
    """Revisión humana (human-in-the-loop): aprueba o rechaza la alerta pendiente."""
    try:
        result = get_orchestrator().resume(run_id, body.aprobar, body.comentario)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    audit("revision_humana", {"run_id": run_id, "aprobar": body.aprobar,
                              "estado_alerta": result.get("estado_alerta")})
    return result


@app.get("/grafo", dependencies=[Depends(require_key)])
def grafo() -> dict:
    """Diagrama del grafo de LangGraph en formato Mermaid."""
    return {"mermaid": get_orchestrator().mermaid()}


@app.get("/metrics", dependencies=[Depends(require_key)])
def metrics() -> dict:
    usage = get_llm().usage.as_dict()
    rep = Path(get_config()["rag"]["corpus_dir"]).parents[1] / "reports" / "metrics.json"
    return {"uso_llm": usage,
            "evaluacion": json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None}


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")
