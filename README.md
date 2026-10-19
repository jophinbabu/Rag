# Hybrid RAG with Google Gemini & Qdrant

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline powered by **Google Gemini**, **Qdrant**, and **BM25**, featuring **Reciprocal Rank Fusion (RRF)** hybrid search.

---

## 🌟 Key Architecture & Features

```
                            [ User Query ]
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
       [ Dense Vector Search ]           [ Sparse Lexical Search ]
         Gemini Embeddings / BGE                  BM25
             (Qdrant DB)                      (rank-bm25)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                    [ Reciprocal Rank Fusion ]
                             (RRF)
                                  │
                                  ▼
                     [ Top-K Context Chunks ]
                                  │
                                  ▼
                   [ Google Gemini LLM Engine ]
                   (gemini-2.5-flash / 1.5-flash)
                                  │
                                  ▼
              [ Grounded Answer with Source Citations ]
```

1. **Google Gemini LLM Engine (`src/generate.py`)**:
   - Strict factual grounding: Prevents hallucinations by instructing the model to rely solely on the provided context passages.
   - Transparent citations: Bracketed references (`[Source 1: Title | Chunk ID]`) mapped directly to retrieved evidence.
   - Streaming support: Real-time token streaming for smooth interactive chat.

2. **Hybrid Search with Reciprocal Rank Fusion (`src/retrieve.py`)**:
   - Combines semantic vector similarity (capturing concepts and intent) with lexical keyword matching (capturing precise terms, acronyms, and codes).
   - Fuses rankings with $RRF\_score(d) = \sum \frac{1}{60 + rank + 1}$.

3. **Dual Embedding Providers (`src/embeddings.py`)**:
   - **Google Gemini**: `text-embedding-004` (768 dimensions).
   - **Local fallback**: `BAAI/bge-small-en-v1.5` (384 dimensions) via SentenceTransformers for offline development.

4. **Reliable Storage & Processing**:
   - **Vector Store (`src/index.py`)**: Embedded on-disk Qdrant database (`data/qdrant_db/`), no Docker setup needed.
   - **Ingestion (`src/ingest.py`)**: ArXiv paper search/download + PyMuPDF (`fitz`) multi-format document parser.
   - **Chunking (`src/chunk.py`)**: Structured section and token chunking with metadata preservation.

---

## 🚀 Quick Start

### 1. Configure Your Gemini API Key
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
copy .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the Interactive Streamlit Web UI
```bash
streamlit run app.py
```
- Upload custom PDFs or download research papers directly from ArXiv.
- Toggle between **Hybrid Search**, **Dense Search**, and **Sparse Search**.
- Ask questions and view live streamed answers with an interactive citation drawer.

### 3. Run via Command Line Interface (CLI)

#### Check status:
```bash
python main.py status
```

#### Run the Automated Demo:
```bash
python main.py demo
```

#### Ingest ArXiv papers:
```bash
python main.py ingest --arxiv "retrieval augmented generation" -n 3
```

#### Ingest local files / directory:
```bash
python main.py ingest --dir data/pdfs
```

#### Ask a question:
```bash
python main.py query "How does Reciprocal Rank Fusion improve retrieval?" --mode hybrid -k 3
```

---

## 🧪 Testing

Run unit and integration tests:
```bash
pytest -v test_rag.py
```

---

## 📂 Project Structure

```
d:/Rag/
├── src/
│   ├── __init__.py
│   ├── config.py         # Central settings and environment variables
│   ├── ingest.py         # ArXiv downloader and PyMuPDF document parser
│   ├── chunk.py          # Section and sliding-window chunker with metadata
│   ├── embeddings.py     # Gemini text-embedding-004 + Local BGE fallback
│   ├── index.py          # Qdrant vector database and BM25 indexer
│   ├── retrieve.py       # Dense, Sparse BM25, and Hybrid RRF search
│   ├── generate.py       # Gemini-2.5-flash grounded generator with citations
│   └── rag.py            # Unified master pipeline orchestrator
├── main.py               # CLI runner
├── app.py                # Streamlit web UI
├── test_rag.py           # Unit and integration tests
├── requirements.txt      # Dependency specifications
├── .env.example          # Environment template
└── data/
    ├── pdfs/             # Downloaded or uploaded documents
    └── qdrant_db/        # Embedded vector database storage
```
