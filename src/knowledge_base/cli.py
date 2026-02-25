"""CLI entry point for Portable Knowledge Base System."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="knowledge-base", help="Portable Knowledge Base CLI")
console = Console()


@app.command()
def init(
    data_dir: Path = typer.Option("data", "--data-dir", "-d", help="Data directory"),
):
    """Initialize the knowledge base storage."""
    from knowledge_base.storage.portable import (
        SQLiteStore,
        ChromaStore,
        KuzuGraphStore,
        PortableStorageConfig,
    )

    async def _init():
        config = PortableStorageConfig(data_dir=data_dir)
        config.ensure_directories()

        console.print("[bold]Initializing Knowledge Base...[/bold]")

        # Initialize SQLite
        console.print("  [green]SQLite[/green] (documents + FTS5)")
        sqlite = SQLiteStore(config.sqlite)
        await sqlite.initialize()
        await sqlite.close()

        # Initialize ChromaDB
        console.print("  [green]ChromaDB[/green] (vector store)")
        chroma = ChromaStore(config.chroma)
        await chroma.initialize()
        await chroma.close()

        # Initialize Kuzu
        console.print("  [green]Kuzu[/green] (knowledge graph)")
        kuzu = KuzuGraphStore(config.kuzu)
        await kuzu.initialize()
        await kuzu.close()

        console.print(f"\n[bold green]✓ Knowledge Base initialized at {data_dir}[/bold green]")

    asyncio.run(_init())


@app.command()
def ingest(
    file_path: Path = typer.Argument(..., help="File or directory to ingest"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Document domain"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Process directories recursively"),
):
    """Ingest documents into the knowledge base.

    This command:
    1. Processes documents (text extraction)
    2. Chunks content
    3. Stores chunks in SQLite
    4. Generates embeddings and stores in ChromaDB
    """
    from knowledge_base.storage.portable import SQLiteStore, ChromaStore, PortableStorageConfig
    from knowledge_base.ingestion import DocumentProcessor, SemanticChunker, VisionModelClient
    from knowledge_base.clients.embedding import EmbeddingClient

    async def _ingest():
        config = PortableStorageConfig()
        config.ensure_directories()

        # Initialize stores
        sqlite = SQLiteStore(config.sqlite)
        await sqlite.initialize()

        chroma = ChromaStore(config.chroma)
        await chroma.initialize()

        # Initialize embedding client
        embedding_client = EmbeddingClient()
        await embedding_client.initialize()

        # Initialize vision client
        vision = VisionModelClient()
        await vision.initialize()
        processor = DocumentProcessor(vision_client=vision)
        chunker = SemanticChunker()

        try:
            if file_path.is_dir():
                # Process directory
                console.print(f"[bold]Ingesting directory: {file_path}[/bold]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Processing files...", total=None)

                    docs = await processor.process_directory(file_path, recursive=recursive)

                    total_embeddings = 0
                    for doc in docs:
                        progress.update(task, description=f"Processing: {doc.name}")

                        # Save document
                        from knowledge_base.storage.portable.sqlite_store import Document
                        document = Document(
                            name=doc.name,
                            source_uri=doc.source_path,
                            content=doc.content,
                            domain=domain,
                            metadata=doc.metadata,
                            status="processed",
                        )
                        doc_id = await sqlite.add_document(document)

                        # Chunk and save
                        chunks = chunker.chunk(doc.content, doc_id)
                        chunk_ids = await sqlite.add_chunks_batch(chunks)

                        # Generate and store embeddings
                        if chunks:
                            texts = [c.text for c in chunks]
                            embeddings = await embedding_client.embed_batch(texts)
                            metadatas = [
                                {"document_id": doc_id, "chunk_index": c.chunk_index, "domain": domain or "GENERAL"}
                                for c in chunks
                            ]
                            await chroma.add_embeddings(
                                ids=chunk_ids,
                                embeddings=embeddings,
                                metadatas=metadatas,
                                documents=texts,
                            )
                            total_embeddings += len(chunk_ids)

                    progress.update(task, description=f"Completed: {len(docs)} documents, {total_embeddings} embeddings")

                console.print(f"[green]✓ Ingested {len(docs)} documents ({total_embeddings} embeddings)[/green]")

            else:
                # Process single file
                console.print(f"[bold]Ingesting: {file_path}[/bold]")

                doc = await processor.process(file_path)

                from knowledge_base.storage.portable.sqlite_store import Document
                document = Document(
                    name=doc.name,
                    source_uri=doc.source_path,
                    content=doc.content,
                    domain=domain,
                    metadata=doc.metadata,
                    status="processed",
                )
                doc_id = await sqlite.add_document(document)

                chunks = chunker.chunk(doc.content, doc_id)
                chunk_ids = await sqlite.add_chunks_batch(chunks)

                # Generate and store embeddings
                embedding_count = 0
                if chunks:
                    texts = [c.text for c in chunks]
                    embeddings = await embedding_client.embed_batch(texts)
                    metadatas = [
                        {"document_id": doc_id, "chunk_index": c.chunk_index, "domain": domain or "GENERAL"}
                        for c in chunks
                    ]
                    await chroma.add_embeddings(
                        ids=chunk_ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=texts,
                    )
                    embedding_count = len(chunk_ids)

                console.print(f"[green]✓ Ingested: {doc.name} ({len(chunks)} chunks, {embedding_count} embeddings)[/green]")

        finally:
            await embedding_client.close()
            await vision.close()
            await chroma.close()
            await sqlite.close()

    asyncio.run(_ingest())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search the knowledge base using hybrid search (BM25 + vector)."""
    from knowledge_base.storage.portable import (
        SQLiteStore,
        ChromaStore,
        HybridSearchEngine,
        PortableStorageConfig,
    )
    from knowledge_base.clients.embedding import EmbeddingClient

    async def _search():
        config = PortableStorageConfig()

        sqlite = SQLiteStore(config.sqlite)
        await sqlite.initialize()

        chroma = ChromaStore(config.chroma)
        await chroma.initialize()

        embedding_client = EmbeddingClient()
        await embedding_client.initialize()

        engine = HybridSearchEngine(sqlite, chroma)

        # Generate query embedding for vector search
        query_embedding = await embedding_client.embed_query(query)

        # Perform hybrid search
        results = await engine.search(query, query_embedding=query_embedding, limit=limit)

        # Display results
        table = Table(title=f"Search Results: {query}")
        table.add_column("Score", style="green")
        table.add_column("Document", style="blue")
        table.add_column("Text Preview")

        for r in results:
            text_preview = r.text[:100] + "..." if len(r.text) > 100 else r.text
            table.add_row(
                f"{r.score:.3f}",
                r.document_name or r.document_id[:8],
                text_preview,
            )

        console.print(table)

        await embedding_client.close()
        await sqlite.close()
        await chroma.close()

    asyncio.run(_search())


@app.command()
def stats():
    """Show knowledge base statistics."""
    from knowledge_base.storage.portable import (
        SQLiteStore,
        ChromaStore,
        KuzuGraphStore,
        PortableStorageConfig,
    )

    async def _stats():
        config = PortableStorageConfig()

        # Get stats from each store
        sqlite = SQLiteStore(config.sqlite)
        await sqlite.initialize()
        sqlite_stats = await sqlite.get_stats()
        await sqlite.close()

        chroma = ChromaStore(config.chroma)
        await chroma.initialize()
        chroma_stats = await chroma.get_stats()
        await chroma.close()

        kuzu = KuzuGraphStore(config.kuzu)
        await kuzu.initialize()
        kuzu_stats = await kuzu.get_stats()
        await kuzu.close()

        # Display stats
        console.print("\n[bold]Knowledge Base Statistics[/bold]\n")

        table = Table()
        table.add_column("Component", style="blue")
        table.add_column("Metric", style="green")
        table.add_column("Value")

        # SQLite stats
        table.add_row("SQLite", "Documents", str(sqlite_stats.get("documents", 0)))
        table.add_row("SQLite", "Chunks", str(sqlite_stats.get("chunks", 0)))
        table.add_row("SQLite", "Size (MB)", f"{sqlite_stats.get('db_size_mb', 0):.2f}")

        # ChromaDB stats
        table.add_row("ChromaDB", "Embeddings", str(chroma_stats.get("embedding_count", 0)))
        table.add_row("ChromaDB", "Size (MB)", f"{chroma_stats.get('storage_size_mb', 0):.2f}")

        # Kuzu stats
        table.add_row("Kuzu", "Entities", str(kuzu_stats.get("entities", 0)))
        table.add_row("Kuzu", "Relationships", str(kuzu_stats.get("relationships", 0)))
        table.add_row("Kuzu", "Size (MB)", f"{kuzu_stats.get('storage_size_mb', 0):.2f}")

        console.print(table)

    asyncio.run(_stats())


@app.command()
def serve(
    host: str = typer.Option("localhost", "--host", "-h"),
    port: int = typer.Option(8765, "--port", "-p"),
):
    """Start the REST API server."""
    import uvicorn

    console.print(f"[bold]Starting Knowledge Base API at http://{host}:{port}[/bold]")
    uvicorn.run("knowledge_base.main:app", host=host, port=port, reload=True)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
