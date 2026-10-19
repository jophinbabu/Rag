# src/ingest.py
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Union
import arxiv
import fitz  # PyMuPDF
from src.config import config

logger = logging.getLogger(__name__)

def download_arxiv(
    query: str = "retrieval augmented generation",
    max_results: int = 5,
    output_dir: Union[str, Path, None] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Search and download research papers from ArXiv.
    Defaults to 5 papers to avoid overwhelming the network.
    """
    out_dir = Path(output_dir or config.PDF_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(
        page_size=min(max_results, 50),
        delay_seconds=3,
        num_retries=3
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    logger.info(f"Querying arXiv for '{query}', max_results={max_results}...")
    for r in client.results(search):
        clean_id = r.get_short_id().replace("/", "_")
        pdf_path = out_dir / f"{clean_id}.pdf"
        
        if not pdf_path.exists():
            logger.info(f"Downloading {r.title[:50]}... to {pdf_path}")
            r.download_pdf(dirpath=str(out_dir), filename=f"{clean_id}.pdf")
        else:
            logger.info(f"Using already downloaded PDF: {pdf_path.name}")

        yield {
            "id": clean_id,
            "title": r.title.strip().replace("\n", " "),
            "year": r.published.year if r.published else None,
            "authors": [a.name for a in r.authors],
            "summary": r.summary.strip().replace("\n", " ") if r.summary else "",
            "path": str(pdf_path),
        }

# Alias for backward compatibility
def download(query: str = "retrieval augmented generation", n: int = 5):
    return download_arxiv(query=query, max_results=n)


def parse_pdf(file_path: Union[str, Path], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Extract text page-by-page from a PDF file using PyMuPDF.
    Ensures file handles are properly closed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    doc = fitz.open(str(path))
    pages = []
    full_text_list = []

    try:
        for idx, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append({"page_num": idx + 1, "text": text.strip()})
                full_text_list.append(text.strip())
    finally:
        doc.close()

    doc_meta = meta.copy() if meta else {}
    doc_meta.setdefault("id", path.stem)
    doc_meta.setdefault("title", path.stem.replace("_", " ").title())
    doc_meta.setdefault("path", str(path))
    doc_meta["pages"] = pages
    doc_meta["full_text"] = "\n\n".join(full_text_list)
    doc_meta["page_count"] = len(pages)
    return doc_meta


def parse_text_file(file_path: Union[str, Path], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse a plain text or Markdown file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()

    doc_meta = meta.copy() if meta else {}
    doc_meta.setdefault("id", path.stem)
    doc_meta.setdefault("title", path.stem.replace("_", " ").title())
    doc_meta.setdefault("path", str(path))
    doc_meta["pages"] = [{"page_num": 1, "text": content}]
    doc_meta["full_text"] = content
    doc_meta["page_count"] = 1
    return doc_meta


def parse_document(file_path: Union[str, Path], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse any supported document (PDF, TXT, MD)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, meta)
    elif ext in [".txt", ".md", ".markdown"]:
        return parse_text_file(path, meta)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# Alias for backward compatibility
def parse(meta: Dict[str, Any]) -> Dict[str, Any]:
    parsed = parse_pdf(meta["path"], meta=meta)
    meta["pages"] = [p["text"] for p in parsed["pages"]]
    meta["full_text"] = parsed["full_text"]
    return meta


def ingest_directory(dir_path: Union[str, Path, None] = None) -> List[Dict[str, Any]]:
    """Scan and parse all documents in a directory."""
    target_dir = Path(dir_path or config.PDF_DIR)
    if not target_dir.exists():
        return []

    parsed_docs = []
    for f in target_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".pdf", ".txt", ".md"]:
            try:
                doc = parse_document(f)
                parsed_docs.append(doc)
                logger.info(f"Parsed {f.name} ({doc['page_count']} pages)")
            except Exception as e:
                logger.warning(f"Failed to parse {f.name}: {e}")
    return parsed_docs