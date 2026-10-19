# app.py
import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any

from src.config import config
from src.rag import RAGPipeline
from src.chunk import create_chunks
from src.ingest import parse_document

# Page Configuration
st.set_page_config(
    page_title="Gemini Hybrid RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a Premium UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        max-width: 1300px;
        margin: 0 auto;
    }
    .chunk-card {
        background: #1e222d;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        border-left: 4px solid #4f8bf9;
    }
    .badge-hybrid {
        background: #6366f1;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-dense {
        background: #10b981;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-sparse {
        background: #f59e0b;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background: #232733;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "custom_api_key" not in st.session_state:
    st.session_state.custom_api_key = ""

has_env_key = bool(config.GEMINI_API_KEY)


@st.cache_resource(show_spinner=False)
def _get_shared_pipeline(embedding_mode: str, collection_name: str, model_name: str) -> RAGPipeline:
    """Shared pipeline resource cached server-wide across all user sessions and reruns."""
    return RAGPipeline(
        embedding_provider=embedding_mode,
        collection=collection_name,
        generation_model=model_name,
    )


def get_or_create_pipeline(api_key: str, embedding_mode: str, collection_name: str, model_name: str) -> RAGPipeline:
    """Singleton pipeline provider to prevent multiple Qdrant client lock contentions."""
    pipe = _get_shared_pipeline(embedding_mode, collection_name, model_name)
    resolved_key = api_key or config.GEMINI_API_KEY
    if resolved_key:
        pipe.api_key = resolved_key
        if hasattr(pipe, "generator") and pipe.generator:
            pipe.generator.api_key = resolved_key
    pipe.collection = collection_name
    pipe.generation_model = model_name
    if hasattr(pipe, "generator") and pipe.generator:
        pipe.generator.model = model_name
    st.session_state.pipeline = pipe
    return pipe


# Sidebar Configuration
with st.sidebar:
    st.title("⚡ Gemini RAG Config")
    
    # Secure API Key Status & Input
    if has_env_key:
        st.markdown(
            """
            <div style="background:#16231d; border:1px solid #059669; border-radius:8px; padding:10px 12px; margin-bottom:12px;">
                <span style="color:#34d399; font-weight:600; font-size:0.88rem;">🔒 Gemini API Key: Active</span>
                <div style="color:#94a3b8; font-size:0.75rem; margin-top:2px;">Loaded securely from backend environment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Override API Key (Optional)"):
            custom_key_input = st.text_input(
                "Custom Gemini Key",
                value=st.session_state.custom_api_key,
                type="password",
                placeholder="Leave blank to use default",
                help="Only enter if you wish to override the backend key.",
            )
            if custom_key_input != st.session_state.custom_api_key:
                st.session_state.custom_api_key = custom_key_input
                st.session_state.pipeline = None
                st.rerun()
    else:
        st.markdown(
            """
            <div style="background:#271d15; border:1px solid #d97706; border-radius:8px; padding:10px 12px; margin-bottom:12px;">
                <span style="color:#fbbf24; font-weight:600; font-size:0.88rem;">⚠️ API Key Required</span>
                <div style="color:#94a3b8; font-size:0.75rem; margin-top:2px;">Enter your key below to enable AI answers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        custom_key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.custom_api_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get a free key from https://aistudio.google.com/",
        )
        if custom_key_input != st.session_state.custom_api_key:
            st.session_state.custom_api_key = custom_key_input
            st.session_state.pipeline = None
            st.rerun()

    active_api_key = st.session_state.custom_api_key or config.GEMINI_API_KEY or ""

    # Model & Retrieval Settings
    st.subheader("Model Settings")
    selected_gen_model = st.selectbox(
        "Gemini Generation Model",
        ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash"],
        index=0
    )

    embedding_mode = st.selectbox(
        "Embedding Provider",
        ["auto", "gemini", "local"],
        help="Auto uses Gemini embedding when API key is set, else local BGE model."
    )

    collection_name = st.text_input("Qdrant Collection", value="rag_docs")

    # Ingestion Controls
    st.divider()
    st.subheader("📥 Document Ingestion")

    # ArXiv Ingestion
    with st.expander("Search & Ingest from ArXiv"):
        arxiv_query = st.text_input("ArXiv Query", value="retrieval augmented generation")
        arxiv_count = st.slider("Number of Papers", min_value=1, max_value=10, value=3)
        if st.button("Download & Index ArXiv Papers", use_container_width=True):
            with st.spinner("Fetching papers, parsing PDFs, and building indexes..."):
                try:
                    pipeline = get_or_create_pipeline(
                        active_api_key, embedding_mode, collection_name, selected_gen_model
                    )
                    chunks = pipeline.ingest_arxiv(query=arxiv_query, max_results=arxiv_count, build_now=True)
                    st.success(f"Indexed {len(chunks)} chunks from {arxiv_count} arXiv papers!")
                except Exception as e:
                    st.error(f"Error during arXiv ingestion: {e}")

    # File Upload Ingestion
    with st.expander("Upload Local Files (PDF / TXT / MD)"):
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Index Uploaded Files", use_container_width=True):
            with st.spinner("Processing uploaded files..."):
                try:
                    pipeline = get_or_create_pipeline(
                        active_api_key, embedding_mode, collection_name, selected_gen_model
                    )
                    all_chunks = []
                    for uf in uploaded_files:
                        save_path = config.PDF_DIR / uf.name
                        with open(save_path, "wb") as f:
                            f.write(uf.getbuffer())
                        doc = parse_document(save_path)
                        chunks = create_chunks(doc)
                        all_chunks.extend(chunks)

                    pipeline.build_index(all_chunks, recreate=True)
                    st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(uploaded_files)} files!")
                except Exception as e:
                    st.error(f"Error indexing files: {e}")

    # Quick Demo Loader
    if st.button("🚀 Load Quick Starter Demo Data", use_container_width=True):
        with st.spinner("Loading demo knowledge base..."):
            try:
                pipeline = get_or_create_pipeline(
                    active_api_key, embedding_mode, collection_name, selected_gen_model
                )
                demo_doc = config.DATA_DIR / "sample_rag_guide.txt"
                if not demo_doc.exists():
                    demo_text = """# Modern Retrieval-Augmented Generation (RAG) Architecture

1 Introduction
Retrieval-Augmented Generation (RAG) combines pre-trained parametric language models with non-parametric external memory. Instead of relying solely on parameters learned during pre-training, RAG dynamically retrieves relevant context passages from an external corpus to inform the language model's generation.

2 Hybrid Search with Reciprocal Rank Fusion
State-of-the-art RAG implementations combine dense semantic embeddings with sparse lexical search (such as BM25). Dense search captures semantic meaning, synonyms, and conceptual intent using vector embeddings. Sparse search excels at exact keyword matching, code symbols, specific IDs, and rare entities.
Reciprocal Rank Fusion (RRF) combines rankings from dense and sparse retrievers using the formula:
RRF_score(d) = sum(1 / (k + rank + 1))
where k is a smoothing constant typically set to 60.

3 Gemini Generation Engine
Google Gemini provides high-speed, cost-effective reasoning with expansive context windows. Grounded prompt engineering instructs Gemini to strictly cite sources and decline speculation when external evidence is insufficient, preventing hallucinations in production environments.
"""
                    with open(demo_doc, "w", encoding="utf-8") as f:
                        f.write(demo_text)

                pipeline.ingest_files(demo_doc, build_now=True)
                st.success("Starter demo knowledge base loaded into Qdrant & BM25!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading demo data: {e}. Please retry in a few moments.")


# Main Application Window
pipeline = get_or_create_pipeline(
    active_api_key, embedding_mode, collection_name, selected_gen_model
)
stats = pipeline.get_stats()
indexed_count = stats.get("points_count", 0)

col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("⚡ Hybrid RAG with Google Gemini")
    st.caption("Dense Vector Search (Qdrant) + Sparse Lexical Search (BM25) with Reciprocal Rank Fusion (RRF)")
with col_stat:
    st.markdown(
        f"<div class='status-pill' style='margin-top: 20px;'>"
        f"📊 Collection <b>{collection_name}</b>: <b>{indexed_count}</b> chunks"
        f"</div>",
        unsafe_allow_html=True
    )

if indexed_count == 0:
    st.info(
        f"💡 **Collection '{collection_name}' is currently empty.** "
        f"Click the button below to load starter knowledge in 2 seconds, or use the sidebar to upload PDFs or search arXiv."
    )
    if st.button("🚀 Populate Collection with Starter Knowledge", type="primary", use_container_width=True):
        with st.spinner("Indexing starter knowledge base into Qdrant & BM25..."):
            demo_doc = config.DATA_DIR / "sample_rag_guide.txt"
            if not demo_doc.exists():
                demo_text = """# Modern Retrieval-Augmented Generation (RAG) Architecture

1 Introduction
Retrieval-Augmented Generation (RAG) combines pre-trained parametric language models with non-parametric external memory. Instead of relying solely on parameters learned during pre-training, RAG dynamically retrieves relevant context passages from an external corpus to inform the language model's generation.

2 Hybrid Search with Reciprocal Rank Fusion
State-of-the-art RAG implementations combine dense semantic embeddings with sparse lexical search (such as BM25). Dense search captures semantic meaning, synonyms, and conceptual intent using vector embeddings. Sparse search excels at exact keyword matching, code symbols, specific IDs, and rare entities.
Reciprocal Rank Fusion (RRF) combines rankings from dense and sparse retrievers using the formula:
RRF_score(d) = sum(1 / (k + rank + 1))
where k is a smoothing constant typically set to 60.

3 Gemini Generation Engine
Google Gemini provides high-speed, cost-effective reasoning with expansive context windows. Grounded prompt engineering instructs Gemini to strictly cite sources and decline speculation when external evidence is insufficient, preventing hallucinations in production environments.
"""
                with open(demo_doc, "w", encoding="utf-8") as f:
                    f.write(demo_text)

            try:
                pipeline.ingest_files(demo_doc, build_now=True)
                st.success("Starter knowledge indexed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error indexing starter data: {e}. Please retry in a few moments.")

# Sample prompt chips when collection has data
sample_prompt = None
if indexed_count > 0 and len(st.session_state.messages) == 0:
    st.markdown("**Suggested Questions:**")
    sq1, sq2, sq3 = st.columns(3)
    with sq1:
        if st.button("🔍 How does Hybrid Search & RRF work?", use_container_width=True):
            sample_prompt = "How does Hybrid Search with Reciprocal Rank Fusion work?"
    with sq2:
        if st.button("⚡ Why use Gemini for generation in RAG?", use_container_width=True):
            sample_prompt = "Why is Google Gemini used for the generation engine in RAG?"
    with sq3:
        if st.button("🧠 What is modern RAG architecture?", use_container_width=True):
            sample_prompt = "What is modern Retrieval-Augmented Generation and how does it work?"

# Search Parameters Bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_mode = st.radio(
        "Search Strategy",
        ["Hybrid (Dense + BM25 RRF)", "Dense (Vector)", "Sparse (BM25)"],
        horizontal=True
    )
    mode_key = "hybrid" if "Hybrid" in search_mode else ("dense" if "Dense" in search_mode else "sparse")

with col2:
    top_k = st.slider("Top K Chunks", min_value=1, max_value=8, value=3)

with col3:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander(f"📚 Retrieved Context Sources ({len(message['chunks'])})"):
                for idx, c in enumerate(message["chunks"], 1):
                    score = c.get("score", 0)
                    r_type = c.get("retrieval_type", "unknown")
                    badge_class = f"badge-{r_type}"
                    st.markdown(
                        f"<div class='chunk-card'>"
                        f"<b>Source {idx}:</b> {c.get('title', 'Unknown')} | "
                        f"<code>{c.get('chunk_id')}</code> "
                        f"<span class='{badge_class}'>{r_type.upper()} ({score:.3f})</span><br><br>"
                        f"{c.get('text')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

# Query Input
prompt = st.chat_input("Ask a question based on your indexed documents...")
query_to_run = prompt or sample_prompt

if query_to_run:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # Process RAG Request
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages..."):
            retrieved_chunks = pipeline.retrieve(query_to_run, mode=mode_key, top_k=top_k)

        if not retrieved_chunks:
            answer = (
                f"No relevant context passages were found in collection '{collection_name}'. "
                "Please make sure documents have been indexed first using the sidebar."
            )
            st.warning(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": []
            })
        else:
            if not active_api_key:
                st.warning("⚠️ No Gemini API key provided. Showing retrieved context passages only. Add your API key in the sidebar to generate AI answers.")
                with st.expander("Retrieved Context Chunks", expanded=True):
                    for idx, c in enumerate(retrieved_chunks, 1):
                        st.markdown(f"**Chunk {idx} [{c.get('chunk_id')}]:**\n\n{c.get('text')}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Context retrieved successfully. (Set your Gemini API key to generate grounded answers)",
                    "chunks": retrieved_chunks
                })
            else:
                try:
                    response_placeholder = st.empty()
                    full_answer = ""
                    # Stream tokens from Gemini
                    for chunk_text in pipeline.ask(query_to_run, mode=mode_key, top_k=top_k, stream=True):
                        full_answer += chunk_text
                        response_placeholder.markdown(full_answer + "▌")
                    response_placeholder.markdown(full_answer)

                    # Show citations
                    with st.expander(f"📚 Grounded Context Citations ({len(retrieved_chunks)})"):
                        for idx, c in enumerate(retrieved_chunks, 1):
                            score = c.get("score", 0)
                            r_type = c.get("retrieval_type", "unknown")
                            badge_class = f"badge-{r_type}"
                            st.markdown(
                                f"<div class='chunk-card'>"
                                f"<b>Source {idx}:</b> {c.get('title', 'Unknown')} | "
                                f"<code>{c.get('chunk_id')}</code> "
                                f"<span class='{badge_class}'>{r_type.upper()} ({score:.3f})</span><br><br>"
                                f"{c.get('text')}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "chunks": retrieved_chunks
                    })

                except Exception as e:
                    st.error(f"Gemini generation error: {e}")
