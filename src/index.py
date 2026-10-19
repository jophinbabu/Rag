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


import os
import tempfile
import threading

_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}
_CLIENT_LOCK = threading.Lock()

def get_qdrant_client(
    path: Optional[str | Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Get a singleton Qdrant client with automatic fallback.
    - Reuses existing active client instances across threads/sessions.
    - If local directory lock fails (e.g. NFS / cloud mount Errno 11 AlreadyLocked),
      automatically falls back to /tmp/rag_qdrant_db.
    - If temporary disk also fails, falls back gracefully to in-memory Qdrant (:memory:).
    """
    global _QDRANT_CLIENTS
    with _CLIENT_LOCK:
        # 1. Remote Qdrant (Host/Port or URL / Qdrant Cloud)
        qdrant_url = url or os.getenv("QDRANT_URL")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            key = f"url:{qdrant_url}"
            if key in _QDRANT_CLIENTS:
                return _QDRANT_CLIENTS[key]
            try:
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                _QDRANT_CLIENTS[key] = client
                return client
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {qdrant_url} ({e}). Falling back to local storage.")

        if host and port:
            key = f"remote:{host}:{port}"
            if key in _QDRANT_CLIENTS:
                return _QDRANT_CLIENTS[key]
            try:
                client = QdrantClient(host=host, port=port)
                _QDRANT_CLIENTS[key] = client
                return client
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {host}:{port} ({e}). Falling back to local storage.")

        # 2. Check if an active client already exists for this path
        target_path_str = str(Path(path or config.QDRANT_DIR).resolve())
        if target_path_str in _QDRANT_CLIENTS:
            client = _QDRANT_CLIENTS[target_path_str]
            try:
                client.get_collections()
                return client
            except Exception:
                pass  # Client closed/invalid, recreate

        # If path was not specified, reuse any default healthy client
        if path is None and "default" in _QDRANT_CLIENTS:
            client = _QDRANT_CLIENTS["default"]
            try:
                client.get_collections()
                return client
            except Exception:
                pass

        # 3. Try primary target path on disk
        target_dir = Path(target_path_str)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=target_path_str)
            _QDRANT_CLIENTS[target_path_str] = client
            _QDRANT_CLIENTS["default"] = client
            return client
        except Exception as e:
            logger.warning(f"Primary Qdrant initialization at '{target_path_str}' failed: {e}")

            # Check if any previously registered client is still functional
            for cached_client in list(_QDRANT_CLIENTS.values()):
                try:
                    cached_client.get_collections()
                    _QDRANT_CLIENTS[target_path_str] = cached_client
                    return cached_client
                except Exception:
                    continue

            # Fallback 1: Local /tmp directory (solves NFS / mount portalocker Errno 11)
            temp_path = str((Path(tempfile.gettempdir()) / "rag_qdrant_db").resolve())
            if temp_path != target_path_str:
                try:
                    Path(temp_path).mkdir(parents=True, exist_ok=True)
                    logger.warning(f"Attempting fallback to temporary local storage: '{temp_path}'...")
                    client = QdrantClient(path=temp_path)
                    _QDRANT_CLIENTS[target_path_str] = client
                    _QDRANT_CLIENTS[temp_path] = client
                    _QDRANT_CLIENTS["default"] = client
                    return client
                except Exception as temp_err:
                    logger.warning(f"Temporary disk fallback also failed: {temp_err}")

            # Fallback 2: In-Memory Qdrant (100% crash-proof, no file locks)
            logger.warning("Falling back to in-memory Qdrant client (location=':memory:')...")
            try:
                client = QdrantClient(location=":memory:")
                _QDRANT_CLIENTS[target_path_str] = client
                _QDRANT_CLIENTS["default"] = client
                return client
            except Exception as mem_err:
                logger.error(f"Failed to create in-memory Qdrant client: {mem_err}")
                raise


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