# Agentic Knowledge Ingestion & Management System

A high-fidelity information extraction engine that transforms unstructured data into a structured, temporally-aware knowledge graph.

## Key Features

- **High-Resolution Adaptive Gleaning**: 2-pass density-aware extraction strategy
- **Verbatim-Grounded Entity Resolution**: Hybrid matching with mandatory grounding quotes
- **Hierarchical Leiden Clustering**: Macro and Micro community detection
- **Map-Reduce Recursive Summarization**: Intelligence reports with edge fidelity
- **Temporal Information Extraction (IE)**: ISO-8601 normalized temporal claims
- **Natural Language Query Interface**: Translate queries to SQL
- **Domain Tagging & Filtering**: Propagate domain context throughout pipeline
- **Human Review Queue**: Flag low-confidence resolutions for manual review
- **LLM-Based Entity Typing**: Advanced entity classification using language models
- **Multi-Domain Knowledge Management**: Unified data model across business domains
- **Hybrid Search (BM25 + Vector)**: Combined keyword and semantic search
- **Cross-Encoder Reranking**: Improved search result quality
- **Auto Domain Detection**: Keyword screening + LLM analysis
- **Multi-Modal Extraction**: Tables, images, figures via modified LLM prompts
- **Guided Extraction**: Fully automated, domain-specific extraction
- **Multi-Level Community Summaries**: Hierarchical entity clustering (macro → meso → micro → nano)
- **Adaptive Type Discovery**: Schema induction from extracted data
- **Unified Search API**: Single endpoint for all search modes

## Quick Start

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run the system
uv run knowledge-base
```

## Note on Frontend

The frontend application has been removed from this repository. The system now provides all functionality via a comprehensive backend API. Users can interact with the system through:
- Interactive API documentation at `http://localhost:8000/docs`
- Direct API calls to the backend endpoints
- WebSocket interface for real-time operations

## Documentation

- **Comprehensive Guide**: [docs/README.md](./docs/README.md)
- **Architecture Details**: [DESIGN_DOC.md](./DESIGN_DOC.md)
- **Setup & Deployment**: [docs/OPERATIONS.md](./docs/OPERATIONS.md)
- **Research & Implementation Plan**: [plan.md](./plan.md)

## Development

- **Lint**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Type check**: `uv run mypy src/`
- **Tests**: `uv run pytest tests/`

## Recent Changes

- Added comprehensive research plan for LLM entity typing and multi-domain management
- Removed temporary test files and cache directories
- Updated documentation with current implementation status
- Improved entity typing with domain-aware classification

## Architecture

The system consists of several key components:

1. **Ingestion Pipeline**: Document parsing, chunking, domain detection
2. **Entity Extraction**: Multi-agent, gleaning, guided extraction
3. **Hybrid Search**: BM25 index, vector store, reranking pipeline
4. **Graph Management**: Hierarchical clustering, community summaries
5. **Query Engine**: Natural language to SQL, hybrid search, reranking

## License

MIT
