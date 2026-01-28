"""MCP Protocol Layer for KBV2
Implements Model Context Protocol for external tool integration.
"""

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.text_to_sql_agent import TextToSQLAgent
from knowledge_base.persistence.v1.vector_store import VectorStore
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """Represents an MCP request from a client."""

    method: str = Field(..., description="The MCP method being called")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the method"
    )
    id: Optional[str | int] = Field(None, description="Unique identifier for the request")


class MCPResponse(BaseModel):
    """Represents an MCP response to a client."""

    result: Optional[Any] = Field(None, description="Result of the operation")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    id: Optional[str | int] = Field(None, description="Identifier matching the request")


class MCPProtocol:
    """Handles MCP protocol communication over WebSocket."""

    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept WebSocket connection and add to clients list."""
        await websocket.accept()
        self.clients.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket from clients list."""
        if websocket in self.clients:
            self.clients.remove(websocket)

    async def handle_message(self, websocket: WebSocket, message: str) -> None:
        """Process incoming MCP message and send response."""
        request_id = None
        try:
            # Parse JSON message
            data = json.loads(message)
            request_id = data.get("id")
            request = MCPRequest(**data)

            # Process the request
            response = await self.process_request(request)

            # Send response back to client
            await websocket.send_text(response.json())

        except Exception as e:
            # Send error response
            error_response = MCPResponse(
                result=None,
                error=f"Failed to process request: {str(e)}",
                id=request_id,
            )
            await websocket.send_text(error_response.json())

    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request based on method."""
        # Default implementation - override in subclasses
        return MCPResponse(
            result={"message": f"Method '{request.method}' not implemented"},
            error=None,
            id=request.id,
        )

    async def broadcast(self, message: str) -> None:
        """Broadcast message to all connected clients."""
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except WebSocketDisconnect:
                disconnected_clients.append(client)

        # Clean up disconnected clients
        for client in disconnected_clients:
            self.disconnect(client)


class KBV2MCPProtocol(MCPProtocol):
    """KBV2-specific MCP protocol with knowledge base methods."""

    def __init__(self):
        super().__init__()
        self.orchestrator = IngestionOrchestrator(
            progress_callback=self._send_progress_update
        )
        self.text_to_sql_agent = None
        self.vector_store = VectorStore()
        self.current_websocket: Optional[WebSocket] = None

    async def _send_progress_update(self, progress_data: Dict[str, Any]) -> None:
        """Send progress update to the current WebSocket client.

        Args:
            progress_data: Dictionary containing progress information.
        """
        if self.current_websocket:
            message = {
                "type": "progress",
                **progress_data,
            }
            try:
                await self.current_websocket.send_text(json.dumps(message))
            except Exception:
                pass

    async def connect(self, websocket: WebSocket) -> None:
        """Accept WebSocket connection and add to clients list."""
        await websocket.accept()
        self.clients.append(websocket)
        self.current_websocket = websocket

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket from clients list."""
        if websocket in self.clients:
            self.clients.remove(websocket)
        if self.current_websocket == websocket:
            self.current_websocket = None

    async def handle_message(self, websocket: WebSocket, message: str) -> None:
        """Process incoming MCP message and send response."""
        self.current_websocket = websocket
        request_id = None
        try:
            # Parse JSON message
            data = json.loads(message)
            request_id = data.get("id")
            request = MCPRequest(**data)

            # Process the request
            response = await self.process_request(request)

            # Send response back to client
            await websocket.send_text(response.json())

        except Exception as e:
            # Send error response
            error_response = MCPResponse(
                result=None,
                error=f"Failed to process request: {str(e)}",
                id=request_id,
            )
            await websocket.send_text(error_response.json())

    async def initialize(self):
        """Initialize the KBV2 MCP protocol components."""
        await self.orchestrator.initialize()
        # Initialize text-to-sql agent with the PostgreSQL database engine
        import os

        database_url = os.getenv(
            "DATABASE_URL", "postgresql://agentzero@localhost:5432/knowledge_base"
        )
        engine = create_engine(database_url)
        self.text_to_sql_agent = TextToSQLAgent(engine=engine)
        await self.vector_store.initialize()

    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request based on method."""
        # Route to appropriate method handler
        method_handlers = {
            "kbv2/ingest_document": self._handle_ingest_document,
            "kbv2/query_text_to_sql": self._handle_query_text_to_sql,
            "kbv2/search_entities": self._handle_search_entities,
            "kbv2/search_chunks": self._handle_search_chunks,
            "kbv2/get_document_status": self._handle_get_document_status,
        }

        if request.method in method_handlers:
            try:
                result = await method_handlers[request.method](request.params)
                return MCPResponse(
                    result=result,
                    error=None,
                    id=request.id,
                )
            except Exception as e:
                return MCPResponse(
                    result=None,
                    error=f"Error processing {request.method}: {str(e)}",
                    id=request.id,
                )
        else:
            return MCPResponse(
                result={"message": f"Method '{request.method}' not implemented"},
                error=None,
                id=request.id,
            )

    async def _handle_ingest_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document ingestion request.

        Args:
            params: Parameters containing 'file_path' and optional 'document_name' and 'domain'.

        Returns:
            Dictionary with ingestion result.
        """
        file_path = params.get("file_path")
        document_name = params.get("document_name")
        domain = params.get("domain")

        if not file_path:
            raise ValueError("file_path parameter is required")

        # Track start time for duration calculation
        start_time = time.time()

        # Send initial progress update
        await self._send_progress_update(
            {
                "stage": 0,
                "status": "started",
                "message": f"Starting ingestion of {document_name or file_path}",
                "timestamp": time.time(),
                "duration": 0.0,
            }
        )

        try:
            # Process the document
            document = await self.orchestrator.process_document(
                file_path=file_path, document_name=document_name, domain=domain
            )

            # Calculate duration
            duration = time.time() - start_time

            # Send completion progress update
            await self._send_progress_update(
                {
                    "stage": 9,
                    "status": "completed",
                    "message": f"Document ingestion completed successfully in {duration:.2f}s",
                    "timestamp": time.time(),
                    "duration": duration,
                }
            )

            return {
                "document_id": str(document.id),
                "document_name": document.name,
                "status": document.status,
                "domain": document.domain,
                "duration": duration,
            }
        except Exception as e:
            # Calculate duration even on failure
            duration = time.time() - start_time

            # Send failure progress update
            await self._send_progress_update(
                {
                    "stage": 9,
                    "status": "failed",
                    "message": f"Document ingestion failed after {duration:.2f}s: {str(e)}",
                    "timestamp": time.time(),
                    "duration": duration,
                    "error": str(e),
                }
            )

            raise

    async def _handle_query_text_to_sql(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text-to-SQL query request.

        Args:
            params: Parameters containing 'query' for natural language query.

        Returns:
            Dictionary with SQL execution results.
        """
        if not self.text_to_sql_agent:
            raise RuntimeError("Text-to-SQL agent not initialized")

        query = params.get("query")
        if not query:
            raise ValueError("query parameter is required")

        result = self.text_to_sql_agent.execute_query(query)
        return result

    async def _handle_search_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity search request.

        Args:
            params: Parameters containing search criteria.

        Returns:
            Dictionary with search results.
        """
        query = params.get("query", "")
        limit = params.get("limit", 10)
        domain = params.get("domain")

        # Use the vector store to search for entities
        if not query:
            raise ValueError("query parameter is required for entity search")

        # Generate embedding for the query using the embedding client
        from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

        embedding_client = EmbeddingClient()
        query_embedding = await embedding_client.embed_text(query)

        # Perform similarity search for entities
        results = await self.vector_store.search_similar_entities(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=0.5,  # Adjust threshold as needed
        )

        entities = []
        for entity_data in results:
            entities.append(
                {
                    "id": str(entity_data["id"]),
                    "name": entity_data["name"],
                    "type": entity_data["entity_type"],
                    "description": entity_data["description"],
                    "properties": entity_data["properties"],
                    "confidence": entity_data["confidence"],
                    "similarity": entity_data["similarity"],
                }
            )

        return {
            "entities": entities,
            "total_count": len(entities),
            "query": query,
            "limit": limit,
            "domain": domain,
        }

    async def _handle_search_chunks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chunk search request.

        Args:
            params: Parameters containing search criteria.

        Returns:
            Dictionary with search results.
        """
        query = params.get("query", "")
        limit = params.get("limit", 10)
        domain = params.get("domain")

        # Use the vector store to search for chunks
        if not query:
            raise ValueError("query parameter is required for chunk search")

        # Generate embedding for the query using the embedding client
        from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

        embedding_client = EmbeddingClient()
        query_embedding = await embedding_client.embed_text(query)

        # Perform similarity search for chunks
        results = await self.vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=0.5,  # Adjust threshold as needed
        )

        chunks = []
        for chunk_data in results:
            chunks.append(
                {
                    "id": str(chunk_data["id"]),
                    "document_id": str(chunk_data["document_id"]),
                    "document_name": chunk_data["document_name"],
                    "text": chunk_data["text"],
                    "chunk_index": chunk_data["chunk_index"],
                    "page_number": chunk_data["page_number"],
                    "similarity": chunk_data["similarity"],
                }
            )

        return {
            "chunks": chunks,
            "total_count": len(chunks),
            "query": query,
            "limit": limit,
            "domain": domain,
        }

    async def _handle_get_document_status(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document status request.

        Args:
            params: Parameters containing 'document_id'.

        Returns:
            Dictionary with document status information.
        """
        document_id = params.get("document_id")
        if not document_id:
            raise ValueError("document_id parameter is required")

        # For now, return a placeholder until we implement the actual lookup
        return {
            "document_id": document_id,
            "status": "not_implemented",
            "progress": 0,
        }


# Create FastAPI app and MCP protocol instance
app = FastAPI(title="KBV2 MCP Server")
kbv2_protocol = KBV2MCPProtocol()


@app.on_event("startup")
async def startup_event():
    """Initialize the MCP server components on startup."""
    await kbv2_protocol.initialize()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for MCP protocol."""
    await kbv2_protocol.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await kbv2_protocol.handle_message(websocket, message)
    except WebSocketDisconnect:
        kbv2_protocol.disconnect(websocket)
