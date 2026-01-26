"""
WebSocket clients for KBV2 MCP protocol communication.

This module provides interactive clients for communicating with the KBV2
Knowledge Base API via the Model Context Protocol (MCP) over WebSocket.
"""

from knowledge_base.clients.websocket_client import KBV2WebSocketClient
from knowledge_base.clients.progress import ProgressVisualizer
from knowledge_base.clients.cli import IngestionCLI
from knowledge_base.clients.llm_client import LLMClient, create_llm_client

__all__ = [
    "KBV2WebSocketClient",
    "ProgressVisualizer",
    "IngestionCLI",
    "LLMClient",
    "create_llm_client",
]
