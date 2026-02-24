## Must Follow Rules

- No CI/CD required
- Always use OpenAI-compatible API `http://localhost:8087/v1/` for LLM
- Always use `http://localhost:11434/` for embeddings (bge-m3, 1024 dims)
- Always use `uv` for Python
- Clean slate, production-ready architecture only
- No backwards compatibility needed, no gradual migration

## Configuration

- **`config.yaml`** - All non-secret settings
- **`.env`** - Secrets only (API keys)

## Storage

Portable databases in `data/` directory:
- **SQLite** (`kbv2.db`) - Documents + FTS5
- **ChromaDB** (`chroma/`) - Vector store
- **Kuzu** (`kuzu/`) - Knowledge graph

**No external database servers required.**

---

> **See [CLAUDE.md](CLAUDE.md) for comprehensive development guidelines.**
