"""
로봇 스펙 문서 임베딩 저장 및 RAG 검색.

- FAISS 사용 시: vector_db_faiss_dir에 인덱스 저장
- FAISS 미설치 시: pickle 스토어 사용
- 임베딩은 배치 단위로 호출해 rate limit·타임아웃을 완화
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings

_FAISS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 배치 임베딩 (병목 완화)
# ---------------------------------------------------------------------------


class BatchedEmbeddings(Embeddings):
    """embed_documents를 배치로 나눠 호출해 대량 청크 시 타임아웃·rate limit을 줄입니다."""

    def __init__(self, underlying: Embeddings, batch_size: int = 100):
        self.underlying = underlying
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self.underlying.embed_documents(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.underlying.embed_query(text)


def _embeddings(settings: Settings, *, batched: bool = True) -> Embeddings:
    base = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
    if batched and getattr(settings, "embed_batch_size", None):
        return BatchedEmbeddings(base, batch_size=settings.embed_batch_size)
    return base


def _load_store(settings: Settings) -> list[dict[str, Any]]:
    path = Path(settings.vector_db_path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_store(settings: Settings, rows: list[dict[str, Any]]) -> None:
    path = Path(settings.vector_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rows, f)


def _normalize(vec: list[float]) -> list[float]:
    """단위 벡터로 정규화. 검색 시 norm 재계산을 피하기 위해 저장 시 사용."""
    n = math.sqrt(sum(x * x for x in vec)) + 1e-12
    return [x / n for x in vec]


def save_documents_to_store(documents: Iterable[Document], *, settings: Settings) -> dict[str, Any]:
    docs = list(documents)
    if not docs:
        return {"chunks": 0, "backend": "none"}

    # 배치 임베딩 사용 → 대량 청크 시 rate limit·타임아웃 완화
    embedder = _embeddings(settings)

    if _FAISS_AVAILABLE and getattr(settings, "vector_db_faiss_dir", None):
        faiss_dir = Path(settings.vector_db_faiss_dir)
        index_path = faiss_dir / "index.faiss"
        if index_path.exists():
            try:
                store = FAISS.load_local(
                    str(faiss_dir),
                    embedder,
                    allow_dangerous_deserialization=True,
                )
                store.add_documents(docs)
            except Exception:
                store = FAISS.from_documents(docs, embedder)
        else:
            store = FAISS.from_documents(docs, embedder)
        store.save_local(str(faiss_dir))
        return {"chunks": len(docs), "backend": "faiss", "path": str(faiss_dir)}
    else:
        # 배치로 임베딩 후, 검색 속도를 위해 정규화된 벡터 저장
        vectors = embedder.embed_documents([d.page_content for d in docs])
        rows = _load_store(settings)
        for d, v in zip(docs, vectors):
            row_vec = [float(x) for x in v]
            rows.append({
                "text": d.page_content,
                "metadata": d.metadata or {},
                "embedding": _normalize(row_vec),
            })
        _save_store(settings, rows)
        return {"chunks": len(docs), "backend": "pickle"}


def _cosine_top_k(
    query_vec: list[float],
    rows: list[dict[str, Any]],
    k: int,
) -> list[int]:
    """코사인 유사도 기준 상위 k개 인덱스 반환."""
    def norm(a: list[float]) -> float:
        return math.sqrt(sum(x * x for x in a)) + 1e-12

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    qn = norm(query_vec)
    scored: list[tuple[float, int]] = []
    for i, r in enumerate(rows):
        v = r.get("embedding") or []
        if not v:
            continue
        sim = dot(v, query_vec) / (norm(v) * qn)
        scored.append((sim, i))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[: max(1, int(k))]]


def get_robot_context(*, query: str, settings: Settings, k: int = 5) -> str:
    """
    작업 시나리오(또는 질의)를 쿼리로 FAISS/벡터 스토어에서 관련 로봇 스펙 청크를 검색합니다.
    FAISS 사용 시 해당 인덱스 우선, 없으면 pickle 스토어의 코사인 유사도로 검색합니다.
    """
    faiss_dir = getattr(settings, "vector_db_faiss_dir", None)
    if _FAISS_AVAILABLE and faiss_dir:
        faiss_path = Path(faiss_dir)
        if (faiss_path / "index.faiss").exists():
            try:
                store = FAISS.load_local(
                    str(faiss_path),
                    _embeddings(settings),
                    allow_dangerous_deserialization=True,
                )
                found = store.similarity_search_with_relevance_scores(query, k=k)
                return "\n\n".join(doc.page_content for doc, _ in found)
            except Exception:
                pass

    rows = _load_store(settings)
    if not rows:
        return ""

    embedder = _embeddings(settings)
    query_vec = [float(x) for x in embedder.embed_query(query)]
    top_indices = _cosine_top_k(query_vec, rows, k)
    return "\n\n".join(rows[i]["text"] for i in top_indices)