# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file from project root if it exists
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

import tempfile

class Config:
    def __init__(self):
        self.BASE_DIR = BASE_DIR

        # Detect cloud container environments (Streamlit Cloud, Docker, Linux mounts)
        # On Streamlit Cloud, repo is mounted at /mount/src/rag where portalocker file locking fails
        is_cloud = (
            str(self.BASE_DIR).startswith("/mount")
            or bool(os.getenv("STREAMLIT_SHARING_MODE"))
            or bool(os.getenv("IS_STREAMLIT_CLOUD"))
        )

        custom_data_dir = os.getenv("RAG_DATA_DIR")
        if custom_data_dir:
            self.DATA_DIR = Path(custom_data_dir)
        elif is_cloud:
            self.DATA_DIR = Path(tempfile.gettempdir()) / "rag_data"
        else:
            self.DATA_DIR = self.BASE_DIR / "data"

        self.PDF_DIR = self.DATA_DIR / "pdfs"

        custom_qdrant_dir = os.getenv("QDRANT_DIR")
        if custom_qdrant_dir:
            self.QDRANT_DIR = Path(custom_qdrant_dir)
        elif is_cloud:
            self.QDRANT_DIR = Path(tempfile.gettempdir()) / "rag_qdrant_db"
        else:
            self.QDRANT_DIR = self.DATA_DIR / "qdrant_db"

        # Ensure core directories exist with fallback
        try:
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.PDF_DIR.mkdir(parents=True, exist_ok=True)
            self.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.DATA_DIR = Path(tempfile.gettempdir()) / "rag_data"
            self.PDF_DIR = self.DATA_DIR / "pdfs"
            self.QDRANT_DIR = Path(tempfile.gettempdir()) / "rag_qdrant_db"
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.PDF_DIR.mkdir(parents=True, exist_ok=True)
            self.QDRANT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def GEMINI_API_KEY(self) -> str | None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                pass
        return key

    @property
    def EMBEDDING_PROVIDER(self) -> str:
        # "gemini" or "local" (sentence-transformers)
        return os.getenv("EMBEDDING_PROVIDER", "auto")

    @property
    def GEMINI_EMBEDDING_MODEL(self) -> str:
        return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    @property
    def LOCAL_EMBEDDING_MODEL(self) -> str:
        return os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    @property
    def GENERATION_MODEL(self) -> str:
        return os.getenv("GENERATION_MODEL", "gemini-3.8-flash")

    @property
    def DEFAULT_COLLECTION(self) -> str:
        return os.getenv("DEFAULT_COLLECTION", "rag_docs")

    @property
    def CHUNK_SIZE(self) -> int:
        return int(os.getenv("CHUNK_SIZE", "500"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(os.getenv("CHUNK_OVERLAP", "100"))

    def get_bm25_path(self, collection: str) -> Path:
        return self.DATA_DIR / f"bm25_{collection}.pkl"

config = Config()
