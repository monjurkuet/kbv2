# KBV2 Configuration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Environment Variables (.env)
│
├─ DATABASE_URL
│  └─ postgresql://user:pass@localhost:5432/knowledge_base
│
├─ OLLAMA_URL
│  └─ http://localhost:11434
│
├─ OLLAMA_MODEL
│  └─ nomic-embed-text
│
├─ LLM_GATEWAY_URL
│  └─ http://localhost:8000/v1
│
└─ GOOGLE_API_KEY
   └─ (optional, for Google embeddings)
         │
         ▼
Configuration Classes (pydantic_settings)
│
├─ GleaningConfig
│  ├─ max_density_threshold: 0.8
│  ├─ min_density_threshold: 0.3
│  ├─ max_passes: 2
│  ├─ diminishing_returns_threshold: 0.05
│  └─ stability_threshold: 0.90
│
├─ ResolutionConfig
│  ├─ confidence_threshold: 0.7
│  ├─ similarity_threshold: 0.85
│  └─ max_candidates: 10
│
├─ ClusteringConfig
│  ├─ min_community_size: 3
│  ├─ iterations: 10
│  ├─ macro_resolution: 0.8
│  └─ micro_resolution: 1.2
│
└─ SynthesisConfig
   ├─ max_tokens: 2000
   └─ edge_fidelity: true
         │
         ▼
Service Initialization
│
├─ IngestionOrchestrator.__init__()
│  ├─ Observability()
│  ├─ GatewayClient()
│  ├─ EmbeddingClient()
│  ├─ VectorStore()
│  ├─ PartitioningService()
│  ├─ GleaningService(gateway, config)
│  ├─ ResolutionAgent(gateway, vector_store, config)
│  ├─ ClusteringService()
│  ├─ SynthesisAgent(gateway)
│  └─ TemporalNormalizer()
│
└─ orchestrator.initialize()
   ├─ vector_store.initialize()
   ├─ vector_store.create_entity_embedding_index()
   └─ vector_store.create_chunk_embedding_index()
```

## Environment Variables

### Required Variables

#### DATABASE_URL
PostgreSQL connection string for the knowledge base database.

**Format:**
```
postgresql://[user[:password]@][host][:port][/database]
```

**Example:**
```
DATABASE_URL=postgresql://kb_user:kb_pass@localhost:5432/knowledge_base
```

**Requirements:**
- PostgreSQL 16+
- pgvector extension installed
- Sufficient permissions for table creation and indexing

#### OLLAMA_URL
URL for the Ollama embedding service.

**Example:**
```
OLLAMA_URL=http://localhost:11434
```

**Requirements:**
- Ollama server running
- `nomic-embed-text` model downloaded
- Network access to Ollama server

#### OLLAMA_MODEL
Name of the Ollama embedding model to use.

**Example:**
```
OLLAMA_MODEL=nomic-embed-text
```

**Supported Models:**
- `nomic-embed-text` (default, 768-dim)
- Other Ollama embedding models

#### LLM_GATEWAY_URL
URL for the OpenAI-compatible LLM API gateway.

**Example:**
```
LLM_GATEWAY_URL=http://localhost:8000/v1
```

**Requirements:**
- OpenAI-compatible API server running
- Supports chat completions
- Supports JSON mode responses

### Optional Variables

#### GOOGLE_API_KEY
Google API key for Google Generative AI embeddings.

**Example:**
```
GOOGLE_API_KEY=your-google-api-key
```

**Usage:**
- Alternative to Ollama embeddings
- Uses `gemini-embedding-001` model
- Task type: `RETRIEVAL_DOCUMENT`

## Configuration Classes

### GleaningConfig
Controls the adaptive 2-pass extraction behavior.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_density_threshold` | 0.8 | Maximum information density to continue extraction |
| `min_density_threshold` | 0.3 | Minimum information density to run Pass 2 |
| `max_passes` | 2 | Maximum number of extraction passes |
| `diminishing_returns_threshold` | 0.05 | Minimum new information (5%) to continue |
| `stability_threshold` | 0.90 | Maximum stability (90%) to continue |

**Logic:**
- Pass 1 always runs
- Pass 2 runs if:
  - `information_density >= min_density_threshold` (0.3)
  - `new_info_gain >= diminishing_returns_threshold` (0.05)
  - `stability < stability_threshold` (0.90)
  - `pass_num < max_passes` (2)

### ResolutionConfig
Controls entity deduplication behavior.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.7 | Minimum confidence to auto-merge entities |
| `similarity_threshold` | 0.85 | Minimum vector similarity to consider candidates |
| `max_candidates` | 10 | Maximum number of candidates to evaluate |

**Logic:**
- Vector search finds entities with `similarity >= 0.85`
- LLM evaluates top `max_candidates` entities
- Entities merged if `confidence >= 0.7`
- Otherwise, added to ReviewQueue with `priority` based on confidence

### ClusteringConfig
Controls hierarchical Leiden clustering behavior.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_community_size` | 3 | Minimum entities per community |
| `iterations` | 10 | Number of clustering iterations |
| `macro_resolution` | 0.8 | Resolution parameter for macro communities (level 0) |
| `micro_resolution` | 1.2 | Resolution parameter for micro communities (level 1) |

**Logic:**
- Level 0 clustering at `resolution = 0.8` (broader communities)
- Level 1 clustering at `resolution = 1.2` (tighter communities)
- Micro communities linked to macro parents
- Communities with < `min_community_size` entities are discarded

### SynthesisConfig
Controls report generation behavior.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 2000 | Maximum tokens per report summary |
| `edge_fidelity` | true | Include raw relationships in summaries |

**Logic:**
- Micro reports: Detailed summaries of leaf communities (level 1)
- Macro reports: Strategic synthesis of child reports (level 0)
- Edge fidelity preserves raw relationships to prevent information smoothing
- Reports limited to `max_tokens` to control LLM costs

## Service Initialization

### IngestionOrchestrator
Main orchestrator that coordinates all services.

```python
orchestrator = IngestionOrchestrator()
```

**Initialized Components:**
1. `Observability()` - SRE-Lite tracing and metrics
2. `GatewayClient()` - LLM API client
3. `EmbeddingClient()` - Vector embedding generation
4. `VectorStore()` - Database and vector operations
5. `PartitioningService()` - Document parsing and chunking
6. `GleaningService(gateway, config)` - Adaptive extraction
7. `ResolutionAgent(gateway, vector_store, config)` - Entity deduplication
8. `ClusteringService()` - Community detection
9. `SynthesisAgent(gateway)` - Report generation
10. `TemporalNormalizer()` - Temporal normalization

### Initialization Sequence

```python
await orchestrator.initialize()
```

**Steps:**
1. `vector_store.initialize()` - Creates database tables
2. `vector_store.create_entity_embedding_index()` - Creates IVFFlat index for entities
3. `vector_store.create_chunk_embedding_index()` - Creates IVFFlat index for chunks

### Cleanup

```python
await orchestrator.close()
```

**Steps:**
1. Closes gateway connections
2. Closes database connection pool

## Dependency Injection

All services receive dependencies via constructor injection:

```python
class GleaningService:
    def __init__(
        self,
        gateway: GatewayClient,
        config: GleaningConfig | None = None,
    ) -> None:
        self._gateway = gateway
        self._config = config or GleaningConfig()
```

**Benefits:**
- Easy testing with mock dependencies
- Flexible configuration
- Clear dependency graph

## Configuration Best Practices

### 1. Environment Variables
- Never commit `.env` to version control
- Use `.env.example` as a template
- Document all required and optional variables
- Use sensible defaults where possible

### 2. Configuration Classes
- Use `pydantic_settings` for type-safe configuration
- Provide sensible defaults
- Validate configuration at startup
- Document all parameters

### 3. Service Initialization
- Initialize services in dependency order
- Use async initialization where appropriate
- Handle initialization errors gracefully
- Provide clear error messages

### 4. Connection Management
- Use connection pooling
- Configure appropriate pool sizes
- Handle connection errors with retry logic
- Close connections properly on shutdown

## Configuration Example

### .env Example
```bash
# Database
DATABASE_URL=postgresql://kb_user:kb_pass@localhost:5432/knowledge_base

# Ollama Embeddings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# LLM Gateway
LLM_GATEWAY_URL=http://localhost:8000/v1

# Optional: Google Embeddings
# GOOGLE_API_KEY=your-google-api-key
```

### Custom Configuration
```python
from knowledge_base.ingestion.v1.gleaning_service import GleaningConfig

# Custom gleaning configuration
custom_config = GleaningConfig(
    max_density_threshold=0.9,
    min_density_threshold=0.4,
    max_passes=3,
    diminishing_returns_threshold=0.03,
    stability_threshold=0.95,
)

gleaning_service = GleaningService(gateway, custom_config)
```

## Configuration Validation

### Startup Validation
All configuration classes use Pydantic for validation:

```python
from pydantic import ValidationError

try:
    config = GleaningConfig()
except ValidationError as e:
    print(f"Configuration error: {e}")
    # Handle error
```

### Runtime Validation
Services validate configuration at runtime:

```python
if not 0.0 <= self._config.confidence_threshold <= 1.0:
    raise ValueError("confidence_threshold must be between 0.0 and 1.0")
```

## Configuration Hot Reload

Currently, KBV2 does not support hot reload. Configuration changes require:
1. Update environment variables
2. Restart the application
3. Reinitialize services

## Troubleshooting Configuration

### Common Issues

#### Database Connection Failed
```
Error: Could not connect to database
```
**Solution:**
- Verify `DATABASE_URL` is correct
- Check PostgreSQL is running
- Verify pgvector extension is installed
- Check network connectivity

#### Ollama Not Responding
```
Error: Failed to connect to Ollama
```
**Solution:**
- Verify `OLLAMA_URL` is correct
- Check Ollama server is running: `ollama list`
- Verify model is downloaded: `ollama pull nomic-embed-text`
- Check network connectivity

#### LLM Gateway Timeout
```
Error: LLM gateway timeout
```
**Solution:**
- Verify `LLM_GATEWAY_URL` is correct
- Check gateway server is running
- Increase timeout in GatewayClient
- Check network connectivity

#### Configuration Validation Failed
```
Error: confidence_threshold must be between 0.0 and 1.0
```
**Solution:**
- Review configuration values
- Check against parameter documentation
- Use default values if unsure
- Validate configuration before startup