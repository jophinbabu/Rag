# src/retrieve.py
from typing import List, Dict, Any, Optional
import logging
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from src.config import config
from src.embeddings import BaseEmbedder, get_embedder
from src.index import get_qdrant_client, load_bm25, tokenize_for_bm25

logger = logging.getLogger(__name__)

def dense_search(
    query: str,
    k: int = 10,
    collection: str = "rag_docs",
    embedder: Optional[BaseEmbedder] = None,
    qd: Optional[QdrantClient] = None,
) -> List[Dict[str, Any]]:
    """
    Perform dense vector similarity search using Qdrant.
    """
    if embedder is None:
        embedder = get_embedder()
    if qd is None:
        qd = get_qdrant_client()

    if not qd.collection_exists(collection):
        logger.warning(f"Collection '{collection}' does not exist in Qdrant.")
        return []

    qv = embedder.embed_query(query)

    # Use query_points (modern API) with fallback to search
    try:
        response = qd.query_points(
            collection_name=collection,
            query=qv,
            limit=k,
            with_payload=True,
        )
        points = response.points
    except Exception:
        points = qd.search(
            collection_name=collection,
            query_vector=qv,
            limit=k,
            with_payload=True,
        )

    results = []
    for p in points:
        doc = dict(p.payload or {})
        doc["score"] = float(p.score)
        doc["retrieval_type"] = "dense"
        if "chunk_id" not in doc:
            doc["chunk_id"] = str(p.id)
        results.append(doc)
    return results


def sparse_search(
    query: str,
    k: int = 10,
    bm25: Optional[BM25Okapi] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    collection: str = "rag_docs",
) -> List[Dict[str, Any]]:
    """
    Perform sparse keyword search using BM25Okapi.
    """
    if bm25 is None or chunks is None:
        bm25, chunks = load_bm25(collection)

    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    max_score = max(scores) if max(scores) > 0 else 1.0

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        doc = dict(chunks[idx])
        doc["score"] = float(scores[idx] / max_score)  # Normalized BM25 score
        doc["raw_bm25_score"] = float(scores[idx])
        doc["retrieval_type"] = "sparse"
        if "chunk_id" not in doc:
            doc["chunk_id"] = f"chunk_{idx}"
        results.append(doc)
    return results


def rrf(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
    RRF_score(d) = sum(1 / (k + rank + 1))
    """
    fused: Dict[str, Dict[str, Any]] = {}

    for r_list in result_lists:
        for rank, doc in enumerate(r_list):
            chunk_id = doc.get("chunk_id") or doc.get("text", "")[:40]
            if chunk_id not in fused:
                entry = {
                    "doc": dict(doc),
                    "rrf_score": 0.0,
                    "ranks": {},
                }
                fused[chunk_id] = entry
            
            entry = fused[chunk_id]
            retrieval_type = doc.get("retrieval_type", "unknown")
            entry["rrf_score"] += 1.0 / (k + rank + 1)
            entry["ranks"][retrieval_type] = rank + 1

    sorted_fused = sorted(fused.values(), key=lambda x: -x["rrf_score"])
    
    final_docs = []
    for item in sorted_fused:
        doc = item["doc"]
        doc["score"] = item["rrf_score"]
        doc["retrieval_type"] = "hybrid"
        doc["fusion_ranks"] = item["ranks"]
        final_docs.append(doc)

    return final_docs


def hybrid_search(
    query: str,
    k: int = 10,
    collection: str = "rag_docs",
    embedder: Optional[BaseEmbedder] = None,
    qd: Optional[QdrantClient] = None,
    bm25: Optional[BM25Okapi] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining Dense (vector) and Sparse (BM25) with RRF fusion.
    """
    dense_results = dense_search(query, k=k*2, collection=collection, embedder=embedder, qd=qd)
    try:
        sparse_results = sparse_search(query, k=k*2, bm25=bm25, chunks=chunks, collection=collection)
    except Exception as e:
        logger.warning(f"Could not perform sparse search ({e}); returning dense results only.")
        return dense_results[:k]

    fused = rrf([dense_results, sparse_results])
    return fused[:k]


# Backward compatible aliases
def dense(query: str, k: int = 20, collection: str = "rag_docs"):
    return dense_search(query, k=k, collection=collection)

def sparse(query: str, k: int = 20, collection: str = "rag_docs"):
    return sparse_search(query, k=k, collection=collection)

def hybrid(query: str, k: int = 20, collection: str = "rag_docs"):
    return hybrid_search(query, k=k, collection=collection)