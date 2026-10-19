# src/index.py
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi

from src.config import config
from src.embeddings import BaseEmbedder, get_embedder

logger = logging.getLogger(__name__)

def tokenize_for_bm25(text: str) -> List[str]:
    """Clean tokenization for BM25 retrieval without punctuation noise."""
    return re.findall(r"\b\w+\b", text.lower())


import threading

_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}
_CLIENT_LOCK = threading.Lock()

def get_qdrant_client(
    path: Optional[str | Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> QdrantClient:
    """
    Get a singleton Qdrant client. Defaults to embedded on-disk storage in data/qdrant_db.
    Reuses existing client instances to avoid portalocker file lock conflicts.
    """
    global _QDRANT_CLIENTS
    with _CLIENT_LOCK:
        if host and port:
            key = f"remote:{host}:{port}"
            if key in _QDRANT_CLIENTS:
                return _QDRANT_CLIENTS[key]
            try:
                client = QdrantClient(host=host, port=port)
                _QDRANT_CLIENTS[key] = client
                return client
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {host}:{port} ({e}). Falling back to local disk.")

        target_path = str(Path(path or config.QDRANT_DIR).resolve())
        if target_path in _QDRANT_CLIENTS:
            client = _QDRANT_CLIENTS[target_path]
            try:
                client.get_collections()
                return client
            except Exception:
                pass  # Reconnect if previous instance was closed

        target_dir = Path(target_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=target_path)
        except Exception as e:
            if "already accessed by another instance" in str(e):
                if target_path in _QDRANT_CLIENTS:
                    return _QDRANT_CLIENTS[target_path]
            raise
        _QDRANT_CLIENTS[target_path] = client
        return client


def build_index(
    chunks: List[Dict[str, Any]],
    collection: str = "rag_docs",
    embedder: Optional[BaseEmbedder] = None,
    qd: Optional[QdrantClient] = None,
    recreate: bool = True,
) -> Dict[str, Any]:
    """
    Build both Dense (Qdrant) and Sparse (BM25) indexes for the given chunks.
    Automatically handles vector dimension matching for Gemini (768) or Local (384).
    """
    if not chunks:
        logger.warning("No chunks provided to build_index.")
        return {"collection": collection, "points_count": 0}

    # Ensure embedder and client are initialized
    if embedder is None:
        embedder = get_embedder()
    if qd is None:
        qd = get_qdrant_client()

    dim = embedder.dimension
    logger.info(f"Building Qdrant index '{collection}' (vector_dim={dim}, distance=COSINE)...")

    # Create or update collection
    if qd.collection_exists(collection):
        if recreate:
            logger.info(f"Recreating existing collection '{collection}'...")
            qd.delete_collection(collection)
            qd.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
    else:
        qd.create_collection(
            collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    # Generate dense embeddings
    texts = [c["text"] for c in chunks]
    logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
    vecs = embedder.embed_documents(texts)

    # Upload points to Qdrant
    points = [
        PointStruct(id=i, vector=v, payload=c)
        for i, (c, v) in enumerate(zip(chunks, vecs))
    ]
    
    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qd.upload_points(collection, points[i : i + batch_size])
    logger.info(f"Successfully uploaded {len(points)} points to Qdrant.")

    # Build Sparse BM25 Index
    logger.info(f"Building BM25 sparse index for {len(chunks)} chunks...")
    tokenized_corpus = [tokenize_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index and chunks
    bm25_file = config.get_bm25_path(collection)
    with open(bm25_file, "wb") as f:
        pickle.dump((bm25, chunks), f)
    logger.info(f"Saved BM25 index to {bm25_file}")

    return {
        "collection": collection,
        "points_count": len(chunks),
        "vector_dim": dim,
        "bm25_file": str(bm25_file),
    }


def load_bm25(collection: str = "rag_docs") -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Load the persisted BM25 index and chunks."""
    bm25_file = config.get_bm25_path(collection)
    if not bm25_file.exists():
        raise FileNotFoundError(
            f"BM25 index for '{collection}' not found at {bm25_file}. "
            f"Please run index.build_index() first."
        )
    with open(bm25_file, "rb") as f:
        bm25, chunks = pickle.load(f)
    return bm25, chunks


def get_collection_stats(collection: str = "rag_docs", qd: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """Get collection status and point count."""
    client = qd or get_qdrant_client()
    if not client.collection_exists(collection):
        return {"exists": False, "points_count": 0}
    info = client.get_collection(collection)
    return {
        "exists": True,
        "points_count": info.points_count,
        "status": str(info.status),
    }


# Backward-compatible build alias
def build(chunks: List[Dict[str, Any]], collection: str = "section"):
    embedder = get_embedder()
    qd = get_qdrant_client()
    return build_index(chunks, collection=collection, embedder=embedder, qd=qd)