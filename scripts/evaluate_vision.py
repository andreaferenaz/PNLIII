"""Evaluación con imágenes reales: detector de EPP y sistema completo.

Requiere haber corrido antes:  python scripts/fetch_eval_images.py --n 100

1. Detector: compara, por elemento de EPP exigido en la zona, el estado que sale del YOLO
   (usa, no usa, no visible) contra el estado esperado de las etiquetas reales.
   Informa la exactitud por elemento y la precisión y el recall de «no usa», que es el
   evento que dispara una alerta.
2. Sistema completo: corre la inspección (Inspector → Normativo → Verificador) con la imagen
   y mide la exactitud de la decisión de cumplimiento y de los faltantes.

Uso:
    python scripts/evaluate_vision.py                 # detector + sistema completo
    python scripts/evaluate_vision.py --solo-detector # sin LLM, no gasta tokens
    python scripts/evaluate_vision.py --max 30        # menos imágenes (límite de la API gratuita)

Genera reports/vision_metrics.json y reports/vision_metrics.md.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eppagent.metrics.evaluation import decision_accuracy  # noqa: E402
from eppagent.vision.detector import detections_to_state, get_detector  # noqa: E402


def load_cases():
    p = ROOT / "data" / "eval" / "vision.jsonl"
    if not p.exists():
        sys.exit("Falta data/eval/vision.jsonl. Corré antes: python scripts/fetch_eval_images.py")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def eval_detector(cases):
    det = get_detector()
    if det.backend != "yolo":
        sys.exit("El detector no está en modo yolo (revisá configs/config.yaml y el acceso a Hugging Face).")
    per_elem = defaultdict(lambda: {"ok": 0, "n": 0})
    tp = fp = fn = 0
    preds = []
    for c in cases:
        pred = detections_to_state(det.detect(ROOT / c["imagen"]))
        preds.append(pred)
        for elem, gold in c["estado"].items():
            p = pred.get(elem, "no_visible")
            per_elem[elem]["n"] += 1
            per_elem[elem]["ok"] += int(p == gold)
            tp += int(p == "no_usa" and gold == "no_usa")
            fp += int(p == "no_usa" and gold != "no_usa")
            fn += int(p != "no_usa" and gold == "no_usa")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "imagenes": len(cases),
        "exactitud_por_elemento": {e: round(v["ok"] / v["n"], 3) for e, v in sorted(per_elem.items())},
        "exactitud_estado_global": round(sum(v["ok"] for v in per_elem.values()) /
                                         max(sum(v["n"] for v in per_elem.values()), 1), 3),
        "no_usa_precision": round(prec, 3), "no_usa_recall": round(rec, 3), "no_usa_f1": round(f1, 3),
    }, preds


def eval_system(cases):
    from eppagent.orchestrator import get_orchestrator
    orch = get_orchestrator()
    rows, lat, tokens = [], [], []
    for c in cases:
        out = orch.inspect(c["zona"], image_path=ROOT / c["imagen"], notify_on_fail=False)
        rows.append({"pred_cumple": out["cumple"], "gold_cumple": c["cumple"],
                     "pred_faltantes": out["faltantes"], "gold_faltantes": c["faltantes"]})
        lat.append(out["metricas"]["latencia_ms"])
        tokens.append(out["metricas"]["tokens"])
    return {**decision_accuracy(rows), "latencia_ms_p50": round(statistics.median(lat), 1),
            "tokens_promedio": round(statistics.mean(tokens), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-detector", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()
    cases = load_cases()[: args.max]

    report = {"detector": eval_detector(cases)[0]}
    if not args.solo_detector:
        report["sistema_completo"] = eval_system(cases)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "vision_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    d = report["detector"]
    md = ["# Evaluación con imágenes reales", "",
          f"{d['imagenes']} imágenes de la partición de prueba de keremberke/protective-equipment-detection.", "",
          "## Detector (estado por elemento exigido)", "", "| Elemento | Exactitud |", "|---|---|"]
    md += [f"| {e} | {v} |" for e, v in d["exactitud_por_elemento"].items()]
    md += ["", f"- Exactitud global del estado: {d['exactitud_estado_global']}",
           f"- «No usa»: precisión {d['no_usa_precision']}, recall {d['no_usa_recall']}, F1 {d['no_usa_f1']}"]
    if "sistema_completo" in report:
        md += ["", "## Sistema completo (imagen → dictamen)", ""]
        md += [f"- {k}: {v}" for k, v in report["sistema_completo"].items()]
    (out / "vision_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
