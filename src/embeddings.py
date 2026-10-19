# src/embeddings.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import logging
from src.config import config

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


import time
import random

def _retry_on_transient_error(fn, max_retries: int = 4, initial_delay: float = 1.0):
    """Retry API calls on transient Google API errors (503, 429, timeouts) with exponential backoff."""
    delay = initial_delay
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_transient = any(
                code in err_str
                for code in ["503", "unavailable", "429", "resource_exhausted", "overloaded", "timeout", "deadline"]
            )
            if is_transient and attempt < max_retries - 1:
                sleep_sec = delay + random.uniform(0.1, 0.5)
                logger.warning(
                    f"Gemini API transient error ({e}). Retrying in {sleep_sec:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_sec)
                delay *= 2
            else:
                raise
    raise last_err


class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Gemini Embeddings API."""
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key to GeminiEmbedder."
            )
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        self.model = model or config.GEMINI_EMBEDDING_MODEL
        if "embedding-001" in self.model:
            self._dim = 3072
        elif "embedding-004" in self.model:
            self._dim = 768
        else:
            self._dim = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe dimension with a tiny sample
            res = self.client.models.embed_content(model=self.model, contents="test")
            self._dim = len(res.embeddings[0].values)
        return self._dim

    def _normalize(self, vector: List[float]) -> List[float]:
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = _retry_on_transient_error(
                    lambda: self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                )
            except Exception as e:
                # If configured model fails with 503, attempt gemini-embedding-2 fallback
                if "gemini-embedding-001" in self.model:
                    logger.warning(f"gemini-embedding-001 encountered {e}. Trying gemini-embedding-2...")
                    response = _retry_on_transient_error(
                        lambda: self.client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=batch,
                        )
                    )
                else:
                    raise

            for emb in response.embeddings:
                all_embeddings.append(self._normalize(emb.values))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response = _retry_on_transient_error(
                lambda: self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                )
            )
        except Exception as e:
            if "gemini-embedding-001" in self.model:
                logger.warning(f"gemini-embedding-001 query failed ({e}). Trying gemini-embedding-2...")
                response = _retry_on_transient_error(
                    lambda: self.client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=text,
                    )
                )
            else:
                raise

        if response.embeddings:
            return self._normalize(response.embeddings[0].values)
        raise RuntimeError("Failed to generate embedding from Gemini API.")


class LocalEmbedder(BaseEmbedder):
    """Local embedder using HuggingFace sentence-transformers (BAAI/bge-small-en-v1.5)."""
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or config.LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading local embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()


def get_embedder(
    provider: str = "auto",
    api_key: str | None = None,
    model_name: str | None = None,
) -> BaseEmbedder:
    """
    Factory function to get an embedder instance.
    - provider='gemini': Uses Google Gemini Embeddings (text-embedding-004)
    - provider='local': Uses local SentenceTransformer (bge-small-en-v1.5)
    - provider='auto': Uses Gemini if API key is provided, else falls back to LocalEmbedder
    """
    key = api_key or config.GEMINI_API_KEY
    if provider == "gemini":
        return GeminiEmbedder(api_key=key, model=model_name)
    elif provider == "local":
        return LocalEmbedder(model_name=model_name)
    elif provider == "auto":
        if key:
            try:
                return GeminiEmbedder(api_key=key, model=model_name)
            except Exception as e:
                logger.warning(f"Could not initialize GeminiEmbedder ({e}), falling back to local embedder.")
                return LocalEmbedder(model_name=model_name)
        else:
            logger.info("No Gemini API key detected. Using local SentenceTransformer embedder.")
            return LocalEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
