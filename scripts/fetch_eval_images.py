"""Arma el conjunto de evaluación con imágenes reales de EPP.

Baja la partición de PRUEBA del dataset público keremberke/protective-equipment-detection
(Hugging Face, licencia CC BY 4.0, origen Roboflow Universe). El detector YOLO del proyecto
se entrenó con la partición de entrenamiento de ese dataset, así que estas imágenes no las vio.

Para cada imagen elegida:
- toma las etiquetas reales (helmet, no_glove, goggles, ...);
- asigna la zona donde más elementos exigidos se ven (laboratorio o mantenimiento, que son las
  zonas cuyos EPP cubre el dataset);
- calcula el valor esperado: estado por elemento, faltantes y si cumple.

Uso (en el Codespace, que sí tiene acceso a Hugging Face):
    python scripts/fetch_eval_images.py --n 100

Salida:
    data/eval/vision/images/*.jpg   imágenes elegidas
    data/eval/vision.jsonl          una línea por imagen con zona y valor esperado
"""
import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eppagent.agents.base import load_zones  # noqa: E402
from eppagent.vision.detector import EPP_MAP, detections_to_state  # noqa: E402

REPO = "keremberke/protective-equipment-detection"
ZONAS = ["laboratorio", "mantenimiento"]


def gold_for(labels: list[str], zona: str, zones: dict) -> dict:
    estado = detections_to_state([{"label": l} for l in labels])
    exige = zones[zona]["exige"]
    faltantes = sorted(e for e in exige if estado.get(e) == "no_usa")
    return {"estado": {e: estado[e] for e in exige}, "faltantes": faltantes, "cumple": not faltantes}


def pick_zone(labels: list[str], zones: dict, i: int) -> str:
    estado = detections_to_state([{"label": l} for l in labels])
    vis = {z: sum(estado[e] != "no_visible" for e in zones[z]["exige"]) for z in ZONAS}
    best = max(vis.values())
    cands = [z for z in ZONAS if vis[z] == best]
    return cands[i % len(cands)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="cantidad de imágenes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", help="ruta a un test.zip ya descargado (opcional)")
    args = ap.parse_args()

    if args.zip:
        zpath = Path(args.zip)
    else:
        from huggingface_hub import hf_hub_download
        print(f"Bajando data/test.zip de {REPO} ...")
        zpath = Path(hf_hub_download(REPO, "data/test.zip", repo_type="dataset"))

    raw = ROOT / "data" / "eval" / "vision_raw"
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    ann_file = next(raw.rglob("_annotations.coco.json"))
    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"].lower().replace("-", "_").replace(" ", "_") for c in coco["categories"]}
    by_img: dict[int, list[str]] = {}
    for a in coco["annotations"]:
        name = cats.get(a["category_id"])
        if name in EPP_MAP:
            by_img.setdefault(a["image_id"], []).append(name)

    zones = load_zones()
    rows = []
    for i, img in enumerate(coco["images"]):
        labels = by_img.get(img["id"])
        if not labels:
            continue
        zona = pick_zone(labels, zones, i)
        g = gold_for(labels, zona, zones)
        if all(v == "no_visible" for v in g["estado"].values()):
            continue
        rows.append({"file": img["file_name"], "labels": sorted(set(labels)), "zona": zona, **g})

    # Muestra balanceada entre casos que cumplen y que no cumplen.
    rng = random.Random(args.seed)
    ok = [r for r in rows if r["cumple"]]
    ko = [r for r in rows if not r["cumple"]]
    rng.shuffle(ok)
    rng.shuffle(ko)
    n_ko = min(len(ko), args.n // 2)
    n_ok = min(len(ok), args.n - n_ko)
    n_ko = min(len(ko), args.n - n_ok)
    sel = ko[:n_ko] + ok[:n_ok]
    rng.shuffle(sel)

    out_dir = ROOT / "data" / "eval" / "vision" / "images"
    shutil.rmtree(out_dir.parent, ignore_errors=True)
    out_dir.mkdir(parents=True)
    src_dir = ann_file.parent
    lines = []
    for k, r in enumerate(sel, 1):
        dst = out_dir / f"test_{k:03d}{Path(r['file']).suffix}"
        shutil.copy(src_dir / r["file"], dst)
        lines.append(json.dumps({"imagen": str(dst.relative_to(ROOT)), "origen": r["file"],
                                 "zona": r["zona"], "etiquetas": r["labels"], "estado": r["estado"],
                                 "faltantes": r["faltantes"], "cumple": r["cumple"]}, ensure_ascii=False))
    (ROOT / "data" / "eval" / "vision.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir.parent / "README.md").write_text(
        "# Imágenes de evaluación\n\n"
        f"Partición de prueba de [{REPO}](https://huggingface.co/datasets/{REPO}), "
        "licencia CC BY 4.0, origen Roboflow Universe "
        "(https://universe.roboflow.com/personal-protective-equipment/ppes-kaxsi/dataset/7).\n\n"
        "Las generó `scripts/fetch_eval_images.py`. Las etiquetas esperadas están en `data/eval/vision.jsonl`.\n",
        encoding="utf-8")
    shutil.rmtree(raw, ignore_errors=True)

    c = Counter((r["zona"], r["cumple"]) for r in sel)
    print(f"{len(sel)} imágenes guardadas en {out_dir.relative_to(ROOT)}")
    for (z, cu), n in sorted(c.items()):
        print(f"  {z:14s} {'cumple' if cu else 'no cumple':10s} {n}")


if __name__ == "__main__":
    main()
