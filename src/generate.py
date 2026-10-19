# src/generate.py
from typing import List, Dict, Any, Generator, Optional
import logging
from src.config import config

logger = logging.getLogger(__name__)

RAG_SYSTEM_INSTRUCTION = """You are an expert AI research and technical assistant powered by Gemini.
Your job is to answer user questions accurately, thoroughly, and strictly grounded in the provided context passages.

Guidelines:
1. Base your answer ONLY on the information present in the provided context. Do NOT invent, assume, or hallucinate facts not supported by the context.
2. For every key statement or claim in your response, cite the source using the chunk identifier or title in brackets, e.g. [Source 1] or [Chunk: <chunk_id>].
3. If the provided context does not contain sufficient information to answer the question, state honestly:
   "Based on the provided documents, there is insufficient information to answer this question."
4. Format your answer with clean markdown, bullet points, and code blocks where appropriate.
"""

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context string for Gemini."""
    if not chunks:
        return "No relevant context found in documents."

    formatted = []
    for idx, c in enumerate(chunks, 1):
        title = c.get("title", "Untitled Document")
        chunk_id = c.get("chunk_id", f"c{idx}")
        page = c.get("page")
        page_info = f" | Page: {page}" if page else ""
        score = c.get("score")
        score_info = f" | Relevance Score: {score:.3f}" if score is not None else ""
        text = c.get("text", "").strip()

        block = (
            f"--- [Source {idx}: {title} | Chunk ID: {chunk_id}{page_info}{score_info}] ---\n"
            f"{text}\n"
        )
        formatted.append(block)

    return "\n".join(formatted)


class GeminiGenerator:
    """Generator powered by Google Gemini (gemini-2.5-flash / gemini-1.5-flash)."""
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GENERATION_MODEL
        self._client = None

        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        else:
            logger.warning(
                "Gemini API key is not configured. Calls to generate will require setting GEMINI_API_KEY."
            )

    @property
    def client(self):
        if self._client is None:
            self.api_key = config.GEMINI_API_KEY
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set it in your environment, in .env, "
                    "or pass api_key to GeminiGenerator."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_str = format_context(context_chunks)
        prompt = (
            f"Context Information:\n"
            f"====================\n"
            f"{context_str}\n\n"
            f"====================\n"
            f"User Question: {query}\n\n"
            f"Please provide a comprehensive, grounded answer with citations to the sources above:"
        )
        return prompt

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete grounded response from Gemini."""
        prompt = self.build_prompt(query, context_chunks)
        sys_inst = system_instruction or RAG_SYSTEM_INSTRUCTION

        from google.genai import types
        from src.embeddings import _retry_on_transient_error

        response = _retry_on_transient_error(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=0.2,  # Low temperature for strict factual grounding
                )
            )
        )

        answer_text = response.text if response.text else "No response generated."

        # Extract citation references from the retrieved chunks
        citations = [
            {
                "index": i + 1,
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "page": c.get("page"),
                "score": c.get("score"),
                "source": c.get("metadata", {}).get("source"),
            }
            for i, c in enumerate(context_chunks)
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "model": self.model,
            "context_chunks_count": len(context_chunks),
        }

    def generate_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response chunks from Gemini for real-time UI/CLI feedback."""
        prompt = self.build_prompt(query, context_chunks)
        sys_inst = system_instruction or RAG_SYSTEM_INSTRUCTION

        from google.genai import types
        response_stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.2,
            )
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
