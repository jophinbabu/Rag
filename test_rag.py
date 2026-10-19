# test_rag.py
import pytest
from src.chunk import fixed_chunks, section_chunks, create_chunks
from src.embeddings import get_embedder, LocalEmbedder
from src.index import build_index, get_qdrant_client, load_bm25
from src.retrieve import dense_search, sparse_search, rrf, hybrid_search
from src.generate import format_context, GeminiGenerator
from src.rag import RAGPipeline

def test_fixed_chunks():
    text = "word " * 1500
    chunks = fixed_chunks(text, size=500, overlap=100)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.split()) <= 500

def test_section_chunks_no_regex_leak():
    sample_paper = """
1 Introduction
This is the introduction section discussing retrieval augmented generation. It provides external knowledge.

2 Methodology
2.1 Dense Retrieval
Dense retrieval maps queries and documents into continuous embedding spaces using dual encoders.

2.2 Sparse Retrieval
Sparse retrieval uses exact term frequencies and inverse document frequencies such as BM25.
"""
    chunks = section_chunks(sample_paper, max_words=100)
    # Ensure regex didn't leak numbers like ".1" as separate items
    for c in chunks:
        assert c != ".1"
        assert c != ".2"
        assert len(c.split()) >= 3

def test_create_chunks_metadata():
    doc = {
        "id": "paper_001",
        "title": "Hybrid RAG with Gemini",
        "authors": ["Alice", "Bob"],
        "year": 2026,
        "full_text": "1 Background\nModern RAG combines vector search with lexical matching for optimal accuracy."
    }
    chunks = create_chunks(doc)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c["chunk_id"].startswith("paper_001_c")
    assert c["title"] == "Hybrid RAG with Gemini"
    assert "text" in c
    assert c["metadata"]["authors"] == ["Alice", "Bob"]
    assert c["metadata"]["year"] == 2026

def test_embedder_local():
    embedder = get_embedder(provider="local")
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.dimension == 384
    vec = embedder.embed_query("What is RAG?")
    assert len(vec) == 384

def test_indexing_and_retrieval(tmp_path):
    test_collection = "test_unit_rag"
    test_chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "title": "Quantum Computing",
            "text": "Quantum computers utilize qubits and superposition to solve specific complex algorithms exponentially faster.",
            "page": 1,
            "metadata": {}
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "title": "Neural Networks",
            "text": "Neural networks and deep learning models are trained using backpropagation and gradient descent.",
            "page": 1,
            "metadata": {}
        },
        {
            "chunk_id": "c3",
            "doc_id": "d3",
            "title": "Hybrid Search",
            "text": "Reciprocal Rank Fusion RRF fuses dense vector embeddings with sparse BM25 keyword rankings.",
            "page": 2,
            "metadata": {}
        }
    ]

    embedder = get_embedder(provider="local")
    qd = get_qdrant_client(path=tmp_path / "test_qdrant")
    
    # 1. Build Index
    stats = build_index(test_chunks, collection=test_collection, embedder=embedder, qd=qd, recreate=True)
    assert stats["points_count"] == 3

    # 2. Dense Search
    dense_res = dense_search("qubits and superposition", k=2, collection=test_collection, embedder=embedder, qd=qd)
    assert len(dense_res) >= 1
    assert dense_res[0]["chunk_id"] == "c1"

    # 3. Sparse Search
    bm25, chunks = load_bm25(test_collection)
    sparse_res = sparse_search("gradient descent backpropagation", k=2, bm25=bm25, chunks=chunks)
    assert len(sparse_res) >= 1
    assert sparse_res[0]["chunk_id"] == "c2"

    # 4. Hybrid Search
    hybrid_res = hybrid_search(
        "Reciprocal Rank Fusion BM25",
        k=2,
        collection=test_collection,
        embedder=embedder,
        qd=qd,
        bm25=bm25,
        chunks=chunks
    )
    assert len(hybrid_res) >= 1
    assert hybrid_res[0]["chunk_id"] == "c3"

def test_context_formatting():
    chunks = [
        {"title": "Doc A", "chunk_id": "a1", "page": 1, "score": 0.95, "text": "Content of Doc A"},
        {"title": "Doc B", "chunk_id": "b1", "page": 2, "score": 0.85, "text": "Content of Doc B"}
    ]
    formatted = format_context(chunks)
    assert "Doc A" in formatted
    assert "Chunk ID: a1" in formatted
    assert "Content of Doc A" in formatted
    assert "Doc B" in formatted


def test_qdrant_fallback_on_lock(monkeypatch, tmp_path):
    """Verify that when local disk raises AlreadyLocked / RuntimeError, get_qdrant_client falls back safely."""
    import src.index as idx
    from qdrant_client import QdrantClient as RealQdrantClient

    # Call get_qdrant_client with a unique path
    client1 = idx.get_qdrant_client(path=tmp_path / "lock_test_1")
    assert client1 is not None

    # Re-requesting same path should reuse cached client without re-locking
    client2 = idx.get_qdrant_client(path=tmp_path / "lock_test_1")
    assert client2 is client1

    # Clear cache and simulate disk lock error
    idx._QDRANT_CLIENTS.clear()

    def mock_disk_lock_only(path=None, location=None, **kwargs):
        if path is not None and "disk_locked" in str(path):
            raise RuntimeError("Storage folder /mount/src/rag/data/qdrant_db is already accessed by another instance of Qdrant client.")
        return RealQdrantClient(location=location or ":memory:", **kwargs)

    monkeypatch.setattr(idx, "QdrantClient", mock_disk_lock_only)
    fallback_client = idx.get_qdrant_client(path=tmp_path / "disk_locked")
    assert fallback_client is not None
    assert fallback_client.collection_exists("non_existent") is False


if __name__ == "__main__":
    pytest.main(["-v", "test_rag.py"])
