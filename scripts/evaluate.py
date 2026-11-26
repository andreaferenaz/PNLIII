"""Evaluación del sistema: RAG, respuesta del modelo y agentes.

Uso:  python scripts/evaluate.py
Genera reports/metrics.json y reports/metrics.md (tablas para el informe).
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eppagent.agents.consultor import ConsultorAgent  # noqa: E402
from eppagent.config import get_config  # noqa: E402
from eppagent.llm import get_llm  # noqa: E402
from eppagent.metrics.evaluation import (decision_accuracy, mrr, ndcg_at_k, precision_at_k,  # noqa: E402
                                         recall_at_k, rouge_l, semantic_similarity, token_f1)
from eppagent.orchestrator import get_orchestrator  # noqa: E402
from eppagent.rag import get_index  # noqa: E402

K = 4


def load(name):
    return [json.loads(l) for l in (ROOT / "data" / "eval" / name).read_text(encoding="utf-8").splitlines() if l]


def eval_rag(qa):
    idx = get_index()
    res = {"sin_rerank": [], "con_rerank": []}
    times = []
    for row in qa:
        rel = set(row["docs_relevantes"])
        for mode, rr in (("sin_rerank", False), ("con_rerank", True)):
            out = idx.search(row["pregunta"], top_k=K, rerank=rr)
            ids = [r["id"] for r in out["results"]]
            res[mode].append({"p@k": precision_at_k(ids, rel, K), "r@k": recall_at_k(ids, rel, K),
                              "mrr": mrr(ids, rel), "ndcg@k": ndcg_at_k(ids, rel, K)})
            if rr:
                times.append(out["timing_ms"]["total"])
    agg = {m: {k: round(statistics.mean(x[k] for x in v), 3) for k in v[0]} for m, v in res.items()}
    agg["tiempo_recuperacion_ms_promedio"] = round(statistics.mean(times), 1)
    agg["reranker_activo"] = idx.reranker not in (None, False)
    agg["embeddings"] = idx.encoder_kind
    return agg


def eval_answers(qa):
    agent = ConsultorAgent()
    rows = []
    for row in qa:
        t0 = time.perf_counter()
        out = agent.run(row["pregunta"])
        lat = (time.perf_counter() - t0) * 1000
        sim = semantic_similarity(out["respuesta"], row["respuesta_ref"])
        cited = {f.split("#")[0] for f in out["fuentes"]}
        rows.append({"rougeL": rouge_l(out["respuesta"], row["respuesta_ref"]),
                     "f1": token_f1(out["respuesta"], row["respuesta_ref"]),
                     "sim": sim, "lat": lat,
                     "cita_relevante": bool(cited & set(row["docs_relevantes"]))})
    sims = [r["sim"] for r in rows if r["sim"] is not None]
    return {"rougeL": round(statistics.mean(r["rougeL"] for r in rows), 3),
            "token_f1": round(statistics.mean(r["f1"] for r in rows), 3),
            "similitud_semantica": round(statistics.mean(sims), 3) if sims else None,
            "citas_relevantes": round(sum(r["cita_relevante"] for r in rows) / len(rows), 3),
            "latencia_ms_p50": round(statistics.median(r["lat"] for r in rows), 1)}


def eval_agents(cases):
    orch = get_orchestrator()
    rows, rows_llm, rounds, lat, tokens, fuentes, reglas = [], [], [], [], [], [], []
    for c in cases:
        out = orch.inspect(c["zona"], detections=c["detecciones"], notify_on_fail=False)
        rows.append({"pred_cumple": out["cumple"], "gold_cumple": c["cumple"],
                     "pred_faltantes": out["faltantes"], "gold_faltantes": c["faltantes"]})
        rows_llm.append({"pred_cumple": out["cumple_llm"], "gold_cumple": c["cumple"],
                         "pred_faltantes": out["faltantes_llm"], "gold_faltantes": c["faltantes"]})
        reglas.append(out["corregido_por_reglas"])
        rounds.append(out["rondas_reflexion"])
        lat.append(out["metricas"]["latencia_ms"])
        tokens.append(out["metricas"]["tokens"])
        fuentes.append(bool(out["fuentes"]))
    llm = decision_accuracy(rows_llm)
    return {**decision_accuracy(rows),
            "exactitud_cumple_solo_llm": llm["exactitud_cumple"],
            "exactitud_faltantes_solo_llm": llm["exactitud_faltantes"],
            "casos_corregidos_por_reglas": sum(reglas),
            "rondas_reflexion_promedio": round(statistics.mean(rounds), 2),
            "casos_corregidos_por_verificador": sum(1 for r in rounds if r > 1),
            "dictamenes_con_fuente": round(sum(fuentes) / len(fuentes), 3),
            "latencia_ms_p50": round(statistics.median(lat), 1),
            "tokens_promedio": round(statistics.mean(tokens), 1)}


if __name__ == "__main__":
    qa, cases = load("qa.jsonl"), load("inspecciones.jsonl")
    report = {"proveedor_llm": get_config()["llm"]["provider"], "k": K,
              "rag": eval_rag(qa), "respuestas": eval_answers(qa), "agentes": eval_agents(cases),
              "uso_llm": get_llm().usage.as_dict()}
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Resultados de la evaluación", "", f"Proveedor LLM: `{report['proveedor_llm']}` · k = {K}", "",
          "## RAG (30 preguntas)", "", "| Variante | P@k | R@k | MRR | nDCG@k |", "|---|---|---|---|---|"]
    for m in ("sin_rerank", "con_rerank"):
        r = report["rag"][m]
        md.append(f"| {m} | {r['p@k']} | {r['r@k']} | {r['mrr']} | {r['ndcg@k']} |")
    md += ["", f"Tiempo de recuperación promedio: {report['rag']['tiempo_recuperacion_ms_promedio']} ms",
           "", "## Respuestas del modelo", ""]
    md += [f"- {k}: {v}" for k, v in report["respuestas"].items()]
    md += ["", "## Agentes (15 inspecciones)", ""] + [f"- {k}: {v}" for k, v in report["agentes"].items()]
    md += ["", "## Uso del LLM", ""] + [f"- {k}: {v}" for k, v in report["uso_llm"].items()]
    (out / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
