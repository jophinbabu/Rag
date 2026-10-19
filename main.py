# main.py
import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import config
from src.rag import RAGPipeline

console = Console()

def run_status(args):
    pipeline = RAGPipeline(collection=args.collection)
    stats = pipeline.get_stats()
    
    table = Table(title=f"RAG System Status [Collection: {args.collection}]", show_header=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for k, v in stats.items():
        table.add_row(str(k), str(v))

    console.print(table)


def run_ingest(args):
    pipeline = RAGPipeline(collection=args.collection)

    if args.arxiv:
        console.print(f"[bold blue]Searching and downloading {args.count} papers from ArXiv: '{args.arxiv}'...[/bold blue]")
        chunks = pipeline.ingest_arxiv(query=args.arxiv, max_results=args.count, build_now=True)
        console.print(f"[bold green]Ingestion complete! Indexed {len(chunks)} chunks into '{args.collection}'.[/bold green]")
    elif args.dir or args.file:
        target = args.file or args.dir
        console.print(f"[bold blue]Ingesting local documents from: {target}...[/bold blue]")
        chunks = pipeline.ingest_files(target, build_now=True)
        console.print(f"[bold green]Ingestion complete! Indexed {len(chunks)} chunks into '{args.collection}'.[/bold green]")
    else:
        console.print("[yellow]Please specify --arxiv <query> or --dir <path> or --file <path>.[/yellow]")


def run_query(args):
    pipeline = RAGPipeline(collection=args.collection)
    console.print(f"\n[bold cyan]Query:[/bold cyan] {args.prompt}")
    console.print(f"[dim]Mode: {args.mode} | Top-k: {args.k} | Collection: {args.collection}[/dim]\n")

    try:
        result = pipeline.ask(query=args.prompt, mode=args.mode, top_k=args.k)
        
        # Display Retrieved Chunks
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            table = Table(title=f"Retrieved Passages ({len(chunks)})", show_header=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Chunk ID", style="cyan", width=18)
            table.add_column("Score", style="magenta", width=10)
            table.add_column("Source / Title", style="yellow", width=25)
            table.add_column("Snippet", style="white")

            for i, c in enumerate(chunks, 1):
                score_str = f"{c.get('score', 0):.4f}"
                title_str = str(c.get("title", ""))[:22]
                snippet = c.get("text", "").replace("\n", " ")[:90] + "..."
                table.add_row(str(i), c.get("chunk_id", ""), score_str, title_str, snippet)

            console.print(table)
            console.print("")

        # Display Gemini Generated Answer
        answer_panel = Panel(
            Markdown(result["answer"]),
            title=f"[bold green]Gemini Answer ({result['model']})[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        console.print(answer_panel)

    except Exception as e:
        console.print(f"[bold red]Error running query:[/bold red] {e}")


def run_demo(args):
    console.print("[bold magenta]=== Starting RAG with Gemini Automated Demo ===[/bold magenta]\n")
    
    # 1. Create a sample knowledge document if data/pdfs is empty
    sample_doc_path = config.DATA_DIR / "sample_rag_guide.txt"
    if not sample_doc_path.exists():
        console.print("[blue]Creating sample knowledge document for the demo...[/blue]")
        sample_content = """# Modern Retrieval-Augmented Generation (RAG) Architecture

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
        with open(sample_doc_path, "w", encoding="utf-8") as f:
            f.write(sample_content)

    # 2. Ingest & Index
    pipeline = RAGPipeline(collection="demo_rag")
    console.print("[blue]Ingesting sample document into collection 'demo_rag'...[/blue]")
    chunks = pipeline.ingest_files(sample_doc_path, build_now=True)
    console.print(f"[green]Indexed {len(chunks)} structured chunks into Qdrant & BM25.[/green]\n")

    # 3. Retrieve & Ask
    test_query = "Why is hybrid search better than dense search alone, and how does RRF work?"
    console.print(f"[bold cyan]Running Hybrid Query:[/bold cyan] '{test_query}'\n")
    
    retrieved = pipeline.retrieve(test_query, mode="hybrid", top_k=2)
    table = Table(title="Retrieved Chunks via RRF Hybrid Search", show_header=True)
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Chunk ID", style="cyan", width=20)
    table.add_column("RRF Score", style="magenta", width=12)
    table.add_column("Preview", style="white")

    for i, c in enumerate(retrieved, 1):
        table.add_row(str(i), c.get("chunk_id", ""), f"{c.get('score', 0):.4f}", c.get("text", "")[:80] + "...")
    console.print(table)
    console.print("")

    if config.GEMINI_API_KEY:
        console.print("[blue]Generating answer with Gemini...[/blue]")
        res = pipeline.ask(test_query, mode="hybrid", top_k=2)
        console.print(Panel(Markdown(res["answer"]), title="[bold green]Gemini Response[/bold green]", border_style="green"))
    else:
        console.print(
            Panel(
                "[yellow]Retrieval succeeded! To test Gemini generation, set your GEMINI_API_KEY in .env or run with: "
                "set GEMINI_API_KEY=your_key_here[/yellow]",
                title="Gemini API Key Notice",
                border_style="yellow"
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Production RAG with Google Gemini and Hybrid Search")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status
    status_parser = subparsers.add_parser("status", help="Check collection & index status")
    status_parser.add_argument("--collection", default="rag_docs", help="Collection name")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from ArXiv or files")
    ingest_parser.add_argument("--arxiv", type=str, help="ArXiv query string")
    ingest_parser.add_argument("--count", "-n", type=int, default=5, help="Number of papers to download")
    ingest_parser.add_argument("--dir", type=str, help="Directory containing documents")
    ingest_parser.add_argument("--file", type=str, help="Single file path to ingest")
    ingest_parser.add_argument("--collection", default="rag_docs", help="Target collection")

    # Query
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed documents")
    query_parser.add_argument("prompt", type=str, help="Query question")
    query_parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid", help="Retrieval mode")
    query_parser.add_argument("-k", type=int, default=4, help="Top-k documents to retrieve")
    query_parser.add_argument("--collection", default="rag_docs", help="Target collection")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run automated end-to-end demo")

    args = parser.parse_args()

    if args.command == "status":
        run_status(args)
    elif args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
