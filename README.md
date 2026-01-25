# Agentic Knowledge Ingestion & Management System

A high-fidelity information extraction engine that transforms unstructured data into a structured, temporally-aware knowledge graph.

## Key Features

- **High-Resolution Adaptive Gleaning**: 2-pass density-aware extraction strategy
- **Verbatim-Grounded Entity Resolution**: Hybrid matching with mandatory grounding quotes
- **Hierarchical Leiden Clustering**: Macro and Micro community detection
- **Map-Reduce Recursive Summarization**: Intelligence reports with edge fidelity
- **Temporal Information Extraction (TIE)**: ISO-8601 normalized temporal claims
- **Natural Language Query Interface**: Translate queries to SQL
- **Domain Tagging & Filtering**: Propagate domain context throughout pipeline
- **Human Review Queue**: Flag low-confidence resolutions for manual review

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

## Documentation

- **Comprehensive Guide**: [docs/README.md](./docs/README.md)
- **Architecture Details**: [DESIGN_DOC.md](./DESIGN_DOC.md)
- **Setup & Deployment**: [docs/OPERATIONS.md](./docs/OPERATIONS.md)

## Development

- **Lint**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Type check**: `uv run mypy src/`
