"""Construye y guarda el índice híbrido (BM25 + embeddings) del corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eppagent.rag import HybridIndex  # noqa: E402

if __name__ == "__main__":
    idx = HybridIndex().build()
    idx.save()
    print(f"Índice guardado: {len(idx.chunks)} fragmentos, embeddings={idx.encoder_kind}")
