## Must follow rules 
- No CI/CD required.
- Always use openai compatible api http://localhost:8087/v1/ for LLM
- Always use http://localhost:11434/ endpoint for embedding. 1024 vector, bge-m3
- Always use uv for python.
- Clean slate, production-ready architecture only. Don't need to maintain compatibility, don't need gradual migration.

# environment variables 
---
DATABASE_URL=postgresql://agentzero@localhost/knowledge_base
LLM_API_BASE=http://localhost:8087/v1
LLM_API_KEY=sk-dummy
EMBEDDING_API_BASE=http://localhost:11434
---

add environment variables here if necessary which are used.