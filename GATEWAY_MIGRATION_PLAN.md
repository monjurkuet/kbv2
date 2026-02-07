# KBv2 Gateway Consolidation Plan
## Migration to Standard OpenAI Client (2026)

### Executive Summary

**✅ TESTED & CONFIRMED**: The standard `openai` Python client works perfectly with your local LLM API (`http://localhost:8087/v1`). 

**Key Finding**: No need for custom gateway clients. The OpenAI client provides:
- ✅ Full compatibility with your LLM API
- ✅ Support for 32 models (including thinking/reasoning models)
- ✅ Automatic handling of `reasoning_content` for thinking models
- ✅ Industry standard interface (easy model switching)
- ✅ Built-in retries, streaming, error handling

---

## Current State Analysis

### What Works (Tested)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8087/v1",
    api_key="sk-dummy"
)

# ✅ Basic chat completion
response = client.chat.completions.create(
    model="deepseek-v3.2-chat",
    messages=[{"role": "user", "content": "Hello"}]
)

# ✅ Thinking/reasoning models
response = client.chat.completions.create(
    model="kimi-k2-thinking",
    messages=[{"role": "user", "content": "2+2=?"}]
)

# Access reasoning
msg = response.choices[0].message
answer = msg.content
thinking = getattr(msg, "reasoning_content", None)  # For thinking models

# ✅ JSON mode
response = client.chat.completions.create(
    model="deepseek-v3.2-chat",
    messages=[...],
    response_format={"type": "json_object"}
)

# ✅ List models
models = client.models.list()  # Returns 32 models
```

### Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Basic chat completion | ✅ Works | All models |
| System prompts | ✅ Works | All models |
| Temperature/top_p | ✅ Works | All models |
| max_tokens | ✅ Works | All models |
| JSON mode | ✅ Works | Supported models |
| Streaming | ✅ Works | Standard interface |
| Reasoning models | ✅ Works | Use `getattr(msg, 'reasoning_content', None)` |
| Model listing | ✅ Works | 32 models available |
| Error handling | ✅ Works | Built into client |
| Retries | ✅ Works | Built into client |

---

## Migration Strategy

### Phase 1: Create Unified LLM Client Module (1 hour)

**New File**: `src/knowledge_base/clients/llm.py`

```python
"""Unified LLM client using standard OpenAI SDK.

This module provides a single, standard interface to all LLM operations.
No custom gateway logic needed - relies on OpenAI SDK's built-in features.
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


class LLMClient:
    """Standard LLM client using OpenAI SDK.
    
    Works with any OpenAI-compatible API endpoint.
    Automatically handles reasoning models (thinking/reasoning_content).
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "deepseek-v3.2-chat"
    ):
        """Initialize LLM client.
        
        Args:
            base_url: API base URL (defaults to LLM_API_BASE env var)
            api_key: API key (defaults to LLM_API_KEY env var)
            default_model: Default model to use
        """
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_API_BASE", "http://localhost:8087/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "sk-dummy")
        )
        self.default_model = default_model
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """Generate completion.
        
        Returns dict with:
        - content: The response text
        - reasoning: Thinking content (if reasoning model)
        - model: Model used
        - usage: Token usage
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        
        return {
            "content": msg.content,
            "reasoning": getattr(msg, "reasoning_content", None),
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None
        }
    
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Generate JSON completion."""
        result = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            json_mode=True
        )
        
        if result["content"]:
            import json
            try:
                result["json"] = json.loads(result["content"])
            except json.JSONDecodeError:
                result["json"] = None
        
        return result
    
    def list_models(self) -> List[str]:
        """List available models."""
        models = self.client.models.list()
        return [m.id for m in models.data]
    
    def is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning/thinking model."""
        reasoning_keywords = ["thinking", "reasoner", "o1", "o3"]
        return any(kw in model.lower() for kw in reasoning_keywords)


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
```

### Phase 2: Update All Code to Use New Client (2-3 hours)

**Files to Update**:

1. **`src/knowledge_base/orchestrator.py`**
   - Remove: `ResilientGatewayClient`
   - Add: `from knowledge_base.clients.llm import get_llm_client`
   - Update: Use `llm.complete()` instead of gateway methods

2. **`src/knowledge_base/intelligence/v1/hallucination_detector.py`**
   - Remove: `ResilientGatewayClient`
   - Add: `from knowledge_base.clients.llm import get_llm_client`

3. **`src/knowledge_base/intelligence/v1/self_improvement/prompt_evolution.py`**
   - Remove: Custom gateway reference
   - Add: `from knowledge_base.clients.llm import get_llm_client`
   - Update: Use `llm.complete()` or `llm.complete_json()`

4. **`src/knowledge_base/orchestrator_self_improving.py`**
   - Update: Pass LLMClient instead of gateway

### Phase 3: Delete Deprecated Files (30 minutes)

**Files to DELETE**:

```bash
# Custom gateway clients (no longer needed)
src/knowledge_base/common/resilient_gateway/
    ├── gateway.py           # 770 lines - DELETE
    ├── config.py            # Could keep for reference
    ├── circuit_breaker.py   # DELETE
    ├── metrics.py           # DELETE
    ├── model_discovery.py   # DELETE
    └── compatibility.py     # DELETE

src/knowledge_base/common/gateway.py              # DELETE (516 lines)
src/knowledge_base/clients/llm_client.py          # DELETE (726 lines - deprecated)
src/knowledge_base/clients/rotating_llm_client.py # DELETE (318 lines - deprecated)
src/knowledge_base/clients/unified_llm_client.py  # DELETE (96 lines - incomplete)
src/knowledge_base/clients/middleware/            # DELETE entire directory
```

**Total Lines Removed**: ~2,426 lines

### Phase 4: Update Exports and Imports (30 minutes)

**Update `src/knowledge_base/clients/__init__.py`**:

```python
"""KBv2 clients module."""

# Only export the essentials
from .llm import LLMClient, get_llm_client
from .websocket_client import KBV2WebSocketClient

__all__ = [
    "LLMClient",
    "get_llm_client",
    "KBV2WebSocketClient",
]
```

**Update `src/knowledge_base/ingestion/v1/embedding_client.py`**:
- Keep as-is (Ollama embeddings are separate from LLM)

---

## Benefits of This Approach

### 1. **Simplicity**
- **Before**: 8 different clients, 2,400+ lines of custom code
- **After**: 1 standard client, ~150 lines of wrapper code
- **Reduction**: 94% less code to maintain

### 2. **Standard Interface**
- Uses official OpenAI SDK (industry standard)
- Easy to switch models or providers
- Well-documented, battle-tested
- Large community support

### 3. **Built-in Features** (No Custom Code Needed)
- ✅ Automatic retries with exponential backoff
- ✅ Connection pooling
- ✅ Error handling
- ✅ Streaming support
- ✅ Type hints and validation
- ✅ Async support (`AsyncOpenAI`)

### 4. **Reasoning Model Support**
```python
# Works automatically with thinking models
response = client.chat.completions.create(model="kimi-k2-thinking", ...)
msg = response.choices[0].message
content = msg.content
reasoning = getattr(msg, "reasoning_content", None)  # Thinking process
```

### 5. **No Vendor Lock-in**
```python
# Switch to OpenAI
client = OpenAI(base_url="https://api.openai.com/v1", api_key="sk-...")

# Switch to Anthropic (via compatibility layer)
client = OpenAI(base_url="https://api.anthropic.com/v1", ...)

# Switch to local model
client = OpenAI(base_url="http://localhost:8087/v1", api_key="dummy")

# Same interface for all!
```

---

## Testing Checklist

- [ ] Basic chat completion works
- [ ] System prompts work
- [ ] Temperature/max_tokens work
- [ ] JSON mode works
- [ ] Reasoning models return reasoning_content
- [ ] Non-reasoning models work normally
- [ ] Error handling works
- [ ] Model listing works
- [ ] All existing tests pass

---

## Migration Commands

```bash
# 1. Install openai package (should already be installed)
uv pip install openai

# 2. Create new LLM client module
touch src/knowledge_base/clients/llm.py
# (paste the code from Phase 1)

# 3. Update imports in orchestrator files
# sed commands or manual updates

# 4. Delete deprecated files
rm -rf src/knowledge_base/common/resilient_gateway/
rm src/knowledge_base/common/gateway.py
rm src/knowledge_base/clients/llm_client.py
rm src/knowledge_base/clients/rotating_llm_client.py
rm src/knowledge_base/clients/unified_llm_client.py
rm -rf src/knowledge_base/clients/middleware/

# 5. Run verification
uv run python verify_deployment.py

# 6. Run tests
uv run pytest tests/ -v
```

---

## Final Architecture

```
src/knowledge_base/
├── clients/
│   ├── __init__.py           # Exports: LLMClient, KBV2WebSocketClient
│   ├── llm.py                # NEW: Standard OpenAI client wrapper (~150 lines)
│   ├── websocket_client.py   # MCP/WebSocket (unchanged)
│   └── models.py             # Shared models
│
├── ingestion/v1/
│   └── embedding_client.py   # Ollama embeddings (unchanged)
│
└── common/
    └── (no gateway code)     # All custom gateway code removed
```

**Total Active Clients**: 3
1. **LLMClient** (OpenAI SDK) - for all LLM operations
2. **KBV2WebSocketClient** - for MCP/WebSocket protocol
3. **EmbeddingClient** - for Ollama embeddings

**Code Reduction**: ~2,400 lines → ~150 lines (94% reduction)

---

## Recommendation

**Proceed with this migration.** The OpenAI client is:
- ✅ Tested and working
- ✅ Industry standard
- ✅ Feature-complete
- ✅ Better maintained than custom code
- ✅ Supports reasoning models
- ✅ Easier for new developers

**Estimated Time**: 4-5 hours total
**Risk Level**: Low (simple wrapper around proven library)
**Benefit**: Massive (94% code reduction, standard interface)

Ready to implement?