"""KBv2 clients module.

Exports:
- AsyncLLMClient: Async LLM client with random model rotation
- get_llm_client: Singleton getter for AsyncLLMClient
- KBV2WebSocketClient: MCP/WebSocket protocol client
"""

from knowledge_base.clients.llm import AsyncLLMClient, get_llm_client
from knowledge_base.clients.websocket_client import KBV2WebSocketClient

__all__ = [
    # LLM Client with random model rotation
    "AsyncLLMClient",
    "get_llm_client",
    # WebSocket/MCP Client
    "KBV2WebSocketClient",
]
