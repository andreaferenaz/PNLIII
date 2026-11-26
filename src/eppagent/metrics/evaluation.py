"""Métricas de RAG, de la respuesta del modelo y de los agentes."""
from __future__ import annotations

import math
import re
from collections import Counter


# ---------------------------------------------------------------- RAG
def _doc(i: str) -> str:
    return i.split("#")[0]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = [_doc(x) for x in retrieved[:k]]
    return sum(1 for d in top if d in relevant) / max(k, 1)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = {_doc(x) for x in retrieved[:k]}
    return len(top & relevant) / max(len(relevant), 1)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(retrieved, start=1):
        if _doc(x) in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    seen, dcg = set(), 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        d = _doc(x)
        if d in relevant and d not in seen:
            dcg += 1 / math.log2(i + 1)
            seen.add(d)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------- texto
def _tok(t: str) -> list[str]:
    return re.findall(r"[a-záéíóúñü0-9]+", t.lower())


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L (F1) por subsecuencia común más larga."""
    a, b = _tok(pred), _tok(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def token_f1(pred: str, ref: str) -> float:
    a, b = Counter(_tok(pred)), Counter(_tok(ref))
    common = sum((a & b).values())
    if not common:
        return 0.0
    p, r = common / sum(a.values()), common / sum(b.values())
    return 2 * p * r / (p + r)


_st_model = None


def semantic_similarity(pred: str, ref: str) -> float | None:
    """Coseno entre embeddings (sentence-transformers). None si no está disponible."""
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer, util

        if _st_model is None:
            _st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        e = _st_model.encode([pred, ref], normalize_embeddings=True)
        return float(util.cos_sim(e[0], e[1]))
    except Exception:
        return None


# ---------------------------------------------------------------- agentes
def decision_accuracy(cases: list[dict]) -> dict:
    """Exactitud de la decisión cumple/no cumple y de los faltantes frente al valor esperado."""
    ok_cumple = sum(1 for c in cases if c["pred_cumple"] == c["gold_cumple"])
    ok_falt = sum(1 for c in cases if set(c["pred_faltantes"]) == set(c["gold_faltantes"]))
    n = max(len(cases), 1)
    return {"exactitud_cumple": round(ok_cumple / n, 3), "exactitud_faltantes": round(ok_falt / n, 3)}
