# src/rag.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union

from src.config import config
from src.embeddings import BaseEmbedder, get_embedder
from src.ingest import download_arxiv, parse_document, ingest_directory
from src.chunk import create_chunks
from src.index import build_index, get_qdrant_client, load_bm25, get_collection_stats
from src.retrieve import dense_search, sparse_search, hybrid_search
from src.generate import GeminiGenerator

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    End-to-End Retrieval-Augmented Generation (RAG) Pipeline powered by Google Gemini.
    Features:
    - Multi-modal ingestion (ArXiv search, PDF, TXT, MD)
    - Structured semantic chunking with metadata tracking
    - Dense vector search (Gemini text-embedding-004 or BGE) + Sparse BM25
    - Reciprocal Rank Fusion (RRF) Hybrid Search
    - Strictly grounded Gemini generation with automatic source citation
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_provider: str = "auto",
        collection: str = "rag_docs",
        generation_model: Optional[str] = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.collection = collection or config.DEFAULT_COLLECTION
        self.generation_model = generation_model or config.GENERATION_MODEL
        
        # Initialize embedder, qdrant client, and generator
        self.embedder: BaseEmbedder = get_embedder(provider=embedding_provider, api_key=self.api_key)
        self.qd = get_qdrant_client()
        self.generator = GeminiGenerator(api_key=self.api_key, model=self.generation_model)
        
        self._bm25 = None
        self._chunks: Optional[List[Dict[str, Any]]] = None

    def ingest_arxiv(
        self,
        query: str = "retrieval augmented generation",
        max_results: int = 5,
        build_now: bool = True,
    ) -> List[Dict[str, Any]]:
        """Download papers from arXiv, parse them, chunk them, and optionally index."""
        logger.info(f"Ingesting {max_results} papers from arXiv query: '{query}'")
        all_chunks: List[Dict[str, Any]] = []

        for paper_meta in download_arxiv(query=query, max_results=max_results):
            try:
                parsed_doc = parse_document(paper_meta["path"], meta=paper_meta)
                chunks = create_chunks(
                    parsed_doc,
                    strategy="section",
                    max_words=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP,
                )
                all_chunks.extend(chunks)
                logger.info(f"Parsed & chunked '{paper_meta['title'][:40]}': {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {paper_meta.get('title')}: {e}")

        if build_now and all_chunks:
            self.build_index(all_chunks, recreate=True)

        return all_chunks

    def ingest_files(
        self,
        files_or_dir: Union[str, Path, List[Union[str, Path]]],
        build_now: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ingest local PDF, TXT, or Markdown documents."""
        all_chunks: List[Dict[str, Any]] = []

        if isinstance(files_or_dir, (str, Path)):
            p = Path(files_or_dir)
            if p.is_dir():
                parsed_docs = ingest_directory(p)
            else:
                parsed_docs = [parse_document(p)]
        else:
            parsed_docs = [parse_document(f) for f in files_or_dir]

        for doc in parsed_docs:
            chunks = create_chunks(
                doc,
                strategy="section",
                max_words=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
            )
            all_chunks.extend(chunks)

        if build_now and all_chunks:
            self.build_index(all_chunks, recreate=True)

        return all_chunks

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        recreate: bool = True,
    ) -> Dict[str, Any]:
        """Build vector and BM25 indexes."""
        res = build_index(
            chunks=chunks,
            collection=self.collection,
            embedder=self.embedder,
            qd=self.qd,
            recreate=recreate,
        )
        self._bm25 = None
        self._chunks = None
        return res

    def _ensure_sparse_index(self) -> bool:
        """Lazy load the BM25 sparse index. Returns True if available, False otherwise."""
        if self._bm25 is not None and self._chunks is not None:
            return True
        try:
            self._bm25, self._chunks = load_bm25(self.collection)
            return True
        except FileNotFoundError:
            return False

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context chunks using 'hybrid' (RRF), 'dense' (vector), or 'sparse' (BM25).
        Safely returns empty list if collection or BM25 index has not been built yet.
        """
        mode = mode.lower()
        has_sparse = self._ensure_sparse_index()

        if mode == "dense":
            return dense_search(
                query=query,
                k=top_k,
                collection=self.collection,
                embedder=self.embedder,
                qd=self.qd,
            )
        elif mode == "sparse":
            if not has_sparse:
                logger.info(f"No BM25 index found for collection '{self.collection}'.")
                return []
            return sparse_search(
                query=query,
                k=top_k,
                bm25=self._bm25,
                chunks=self._chunks,
                collection=self.collection,
            )
        elif mode == "hybrid":
            if not has_sparse:
                # If sparse index does not exist, fall back to dense vector search
                return dense_search(
                    query=query,
                    k=top_k,
                    collection=self.collection,
                    embedder=self.embedder,
                    qd=self.qd,
                )
            return hybrid_search(
                query=query,
                k=top_k,
                collection=self.collection,
                embedder=self.embedder,
                qd=self.qd,
                bm25=self._bm25,
                chunks=self._chunks,
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Use 'hybrid', 'dense', or 'sparse'.")

    def ask(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Execute full RAG: Retrieve context chunks and generate a grounded answer with Gemini.
        """
        retrieved_chunks = self.retrieve(query=query, mode=mode, top_k=top_k)

        if stream:
            return self.generator.generate_stream(query, retrieved_chunks)

        gen_result = self.generator.generate(query, retrieved_chunks)
        
        # Attach retrieved chunks and metadata
        return {
            "query": query,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "retrieved_chunks": retrieved_chunks,
            "mode": mode,
            "model": gen_result["model"],
            "collection": self.collection,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return the pipeline status and index statistics."""
        stats = get_collection_stats(self.collection, self.qd)
        stats["embedding_provider"] = self.embedder.__class__.__name__
        stats["vector_dimension"] = self.embedder.dimension
        stats["generation_model"] = self.generation_model
        stats["has_gemini_key"] = bool(self.api_key)
        return stats
