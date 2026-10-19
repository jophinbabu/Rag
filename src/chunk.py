# src/chunk.py
import re
from typing import List, Dict, Any, Union

def fixed_chunks(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into fixed-word chunks with sliding window overlap.
    Guarantees non-empty chunks and prevents infinite loops if size <= overlap.
    """
    if not text or not text.strip():
        return []
    
    words = text.split()
    if not words:
        return []
    
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + size]
        if chunk_words:
            chunk = " ".join(chunk_words).strip()
            if chunk:
                chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def section_chunks(text: str, max_words: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text on headings (e.g. '1 Introduction', '2.1 Method', Markdown headers '# Header').
    Fixed bug where capturing groups in regex leaked into re.split output.
    Long sections are further subdivided using fixed_chunks.
    """
    if not text or not text.strip():
        return []

    # Non-capturing group for section numbers and markdown headers
    heading_pattern = r"\n(?=(?:[0-9]+(?:\.[0-9]+)*\s+[A-Z]|#{1,6}\s+[A-Z]|[A-Z\s]{4,}\n))"
    sections = re.split(heading_pattern, text)
    
    out: List[str] = []
    for s in sections:
        s_clean = s.strip()
        if not s_clean:
            continue
        word_count = len(s_clean.split())
        if word_count > max_words:
            out.extend(fixed_chunks(s_clean, size=max_words, overlap=overlap))
        elif word_count >= 10:  # Exclude trivial fragments
            out.append(s_clean)
            
    # If no sections were detected or all were filtered, fallback to fixed chunks
    if not out:
        out = fixed_chunks(text, size=max_words, overlap=overlap)
        
    return out


def create_chunks(
    doc: Dict[str, Any],
    strategy: str = "section",
    max_words: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Create structured chunks with rich metadata from a parsed document.
    Outputs the standardized chunk schema required for indexing and retrieval.
    """
    doc_id = str(doc.get("id", "doc"))
    title = str(doc.get("title", doc_id))
    source_path = str(doc.get("path", ""))
    year = doc.get("year")
    authors = doc.get("authors", [])

    raw_chunks: List[Dict[str, Any]] = []

    # Check if document has page-level breakdown
    pages = doc.get("pages")
    if pages and isinstance(pages, list) and isinstance(pages[0], dict):
        for p in pages:
            p_num = p.get("page_num", 1)
            p_text = p.get("text", "")
            if strategy == "section":
                page_chunks = section_chunks(p_text, max_words=max_words, overlap=overlap)
            else:
                page_chunks = fixed_chunks(p_text, size=max_words, overlap=overlap)
            
            for c_text in page_chunks:
                raw_chunks.append({
                    "text": c_text,
                    "page": p_num,
                })
    else:
        # Fallback to full_text or raw string
        text = doc.get("full_text") or (doc["pages"] if isinstance(doc.get("pages"), str) else "")
        if not text and "text" in doc:
            text = doc["text"]
            
        if strategy == "section":
            text_chunks = section_chunks(str(text), max_words=max_words, overlap=overlap)
        else:
            text_chunks = fixed_chunks(str(text), size=max_words, overlap=overlap)
            
        for c_text in text_chunks:
            raw_chunks.append({
                "text": c_text,
                "page": None,
            })

    # Assemble final standardized chunk dictionaries
    structured_chunks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_chunks):
        c_text = item["text"].strip()
        if len(c_text.split()) < 8:  # Skip noise/too-short chunks
            continue

        chunk_id = f"{doc_id}_c{idx:04d}"
        structured_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "title": title,
            "chunk_index": idx,
            "text": c_text,
            "page": item["page"],
            "metadata": {
                "authors": authors,
                "year": year,
                "source": source_path,
                "strategy": strategy,
                "word_count": len(c_text.split()),
            }
        })

    return structured_chunks