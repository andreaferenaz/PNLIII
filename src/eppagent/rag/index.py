"""RAG híbrido: BM25 + embeddings densos, fusión RRF y re-ranking con cross-encoder.

Si sentence-transformers no está disponible (o no hay acceso a Hugging Face), los
embeddings densos caen a TF-IDF y el re-ranker se desactiva, para que el flujo siga
funcionando en cualquier entorno.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import get_config
from ..logging_utils import get_logger

log = get_logger("rag")


@dataclass
class Chunk:
    id: str          # doc_id#n
    doc_id: str
    title: str
    text: str
    seccion: str = ""   # p. ej. "Artículo 8" o "Anexo — Instructivo..."


# ---------------------------------------------------------------- utilidades
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúñü0-9]+", text.lower())


def load_corpus(corpus_dir: str) -> list[tuple[str, str, str]]:
    """Devuelve (doc_id, título, texto) por cada .md del corpus."""
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        text = p.read_text(encoding="utf-8")
        title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("#")), p.stem)
        docs.append((p.stem, title, text))
    return docs


def chunk_documents(docs, size: int, overlap: int) -> list[Chunk]:
    """Corta por secciones (##) y, si una sección es larga, por ventanas con solapamiento."""
    chunks: list[Chunk] = []
    for doc_id, title, text in docs:
        sections = re.split(r"\n(?=## )", text)
        n = 0
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            first = sec.splitlines()[0]
            seccion = first[3:].strip() if first.startswith("## ") else ""
            start = 0
            while start < len(sec):
                piece = sec[start:start + size]
                if start > 0 and seccion:   # cada ventana conserva el encabezado de su artículo
                    piece = f"## {seccion} (cont.)\n" + piece
                chunks.append(Chunk(f"{doc_id}#{n}", doc_id, title, piece, seccion))
                n += 1
                if start + size >= len(sec):
                    break
                start += size - overlap
    return chunks


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: suma 1/(k + posición) de cada lista."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- índice
class HybridIndex:
    def __init__(self) -> None:
        self.cfg = get_config()["rag"]
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.dense = None            # matriz normalizada
        self.encoder = None          # SentenceTransformer o TfidfVectorizer
        self.encoder_kind = "none"
        self.reranker = None

    # ---------- construcción
    def build(self) -> "HybridIndex":
        docs = load_corpus(self.cfg["corpus_dir"])
        self.chunks = chunk_documents(docs, self.cfg["chunk_size"], self.cfg["chunk_overlap"])
        self.bm25 = BM25Okapi([_tokenize(c.title + " " + c.text) for c in self.chunks])
        texts = [c.title + "\n" + c.text for c in self.chunks]
        self._init_encoder(texts)
        log.info("Índice construido: %d documentos, %d fragmentos, embeddings=%s",
                 len(docs), len(self.chunks), self.encoder_kind)
        return self

    def _init_encoder(self, texts: list[str]) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.cfg["embedding_model"])
            emb = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.dense = np.asarray(emb, dtype=np.float32)
            self.encoder_kind = "sentence-transformers"
        except Exception as e:  # sin red o sin la librería
            log.warning("Embeddings densos no disponibles (%s). Se usa TF-IDF.", e.__class__.__name__)
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.encoder = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            m = self.encoder.fit_transform(texts).toarray().astype(np.float32)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            self.dense = m
            self.encoder_kind = "tfidf"

    def _load_reranker(self) -> None:
        if self.reranker is not None or self.encoder_kind != "sentence-transformers":
            return
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.cfg["reranker_model"])
        except Exception as e:
            log.warning("Re-ranker no disponible (%s).", e.__class__.__name__)
            self.reranker = False

    # ---------- persistencia
    def save(self) -> None:
        d = Path(self.cfg["index_dir"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
                                       encoding="utf-8")
        np.save(d / "dense.npy", self.dense)
        meta = {"encoder_kind": self.encoder_kind}
        if self.encoder_kind == "tfidf":
            with open(d / "tfidf.pkl", "wb") as f:
                pickle.dump(self.encoder, f)
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls) -> "HybridIndex":
        self = cls()
        d = Path(self.cfg["index_dir"])
        if not (d / "chunks.json").exists():
            log.info("No hay índice guardado; se construye.")
            idx = self.build()
            idx.save()
            return idx
        self.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        self.bm25 = BM25Okapi([_tokenize(c.title + " " + c.text) for c in self.chunks])
        self.dense = np.load(d / "dense.npy")
        self.encoder_kind = json.loads((d / "meta.json").read_text())["encoder_kind"]
        if self.encoder_kind == "tfidf":
            with open(d / "tfidf.pkl", "rb") as f:
                self.encoder = pickle.load(f)
        else:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.cfg["embedding_model"])
        return self

    # ---------- búsqueda
    def _embed_query(self, q: str) -> np.ndarray:
        if self.encoder_kind == "sentence-transformers":
            return np.asarray(self.encoder.encode([q], normalize_embeddings=True))[0]
        v = self.encoder.transform([q]).toarray()[0].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, query: str, top_k: int | None = None, rerank: bool = True) -> dict:
        """Devuelve los fragmentos finales, con trazabilidad y tiempos de cada etapa."""
        top_k = top_k or self.cfg["top_k_final"]
        n = self.cfg["top_k_retrieve"]
        t0 = time.perf_counter()
        bm = self.bm25.get_scores(_tokenize(query))
        bm_rank = list(np.argsort(-bm)[:n])
        dn = self.dense @ self._embed_query(query)
        dn_rank = list(np.argsort(-dn)[:n])
        fused = rrf([bm_rank, dn_rank], k=self.cfg["rrf_k"])
        t_retrieve = time.perf_counter() - t0

        candidates = [i for i, _ in fused[: n]]
        used_reranker = False
        if rerank:
            self._load_reranker()
            if self.reranker:
                pairs = [(query, self.chunks[i].text) for i in candidates]
                scores = self.reranker.predict(pairs)
                candidates = [candidates[j] for j in np.argsort(-np.asarray(scores))]
                used_reranker = True
        final = candidates[:top_k]
        return {
            "results": [{**asdict(self.chunks[i]), "rank": r + 1} for r, i in enumerate(final)],
            "pre_rerank_ids": [self.chunks[i].id for i, _ in fused[:top_k]],
            "timing_ms": {"retrieve": round(t_retrieve * 1000, 1),
                          "total": round((time.perf_counter() - t0) * 1000, 1)},
            "reranker": used_reranker,
            "embeddings": self.encoder_kind,
        }


def format_context(results: list[dict]) -> str:
    """Contexto con etiqueta de fuente para que el LLM cite: [F:doc_id#n]."""
    def _src(r: dict) -> str:
        sec = r.get("seccion") or ""
        return f"{r['title']}" + (f" — {sec}" if sec else "")
    return "\n\n".join(f"[F:{r['id']}] ({_src(r)})\n{r['text']}" for r in results)


_index: HybridIndex | None = None


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex.load()
    return _index
