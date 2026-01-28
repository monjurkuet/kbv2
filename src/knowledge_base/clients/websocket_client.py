"""
WebSocket client for KBV2 MCP protocol communication.

This module provides a robust WebSocket client that implements the Model Context
Protocol (MCP) for communicating with the KBV2 Knowledge Base API.
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Any, Callable, Dict, Optional, Union
import websockets
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """Represents an MCP request to the server."""

    method: str = Field(..., description="The MCP method being called")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the method"
    )
    id: Optional[str] = Field(None, description="Unique identifier for the request")


class MCPResponse(BaseModel):
    """Represents an MCP response from the server."""

    result: Optional[Any] = Field(None, description="Result of the operation")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    id: Optional[str] = Field(None, description="Identifier matching the request")


class ProgressUpdate(BaseModel):
    """Represents a progress update from the server."""

    type: str = Field(..., description="Message type (should be 'progress')")
    stage: Union[int, float] = Field(..., description="Current stage number (0-9)")
    status: str = Field(..., description="Status of current stage")
    message: str = Field(..., description="Progress message")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    duration: Optional[float] = Field(None, description="Duration in seconds")


class KBV2WebSocketClient:
    """WebSocket client for KBV2 MCP protocol communication."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        timeout: float = 1800.0,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ):
        """Initialize the WebSocket client.

        Args:
            host: Server hostname
            port: Server port
            timeout: Connection timeout in seconds (default: 1800 = 30 minutes)
            progress_callback: Optional callback for progress updates
        """
        logger.info(
            f"Initializing KBV2WebSocketClient: host={host}, port={port}, timeout={timeout}"
        )
        self.host = host
        self.port = port
        self.timeout = timeout
        self.progress_callback = progress_callback
        self.websocket: Optional[Any] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        self._should_reconnect = False

    async def connect(self) -> None:
        """Establish WebSocket connection to the server."""
        async with self._connection_lock:
            if self._is_connected and self.websocket:
                logger.info("Already connected to WebSocket server")
                return

            uri = f"ws://{self.host}:{self.port}/ws"
            logger.info(f"Connecting to WebSocket server at {uri}")
            try:
                self.websocket = await asyncio.wait_for(
                    websockets.connect(uri, ping_interval=20, ping_timeout=90),
                    timeout=60.0,  # Shorter timeout for connection establishment
                )
                logger.info("WebSocket connection established successfully")
                logger.info("Starting message listener task")
                self._listen_task = asyncio.create_task(
                    self._listen_for_messages(), name="message_listener"
                )
                logger.info("Message listener task started")
                self._is_connected = True
            except asyncio.TimeoutError:
                logger.error(f"Connection timeout after 60s")
                self._is_connected = False
                raise ConnectionError(f"Connection timeout after 60s")
            except Exception as e:
                logger.error(f"Failed to connect to {uri}: {e}", exc_info=True)
                self._is_connected = False
                raise ConnectionError(f"Failed to connect to {uri}: {e}")

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        logger.info("Disconnecting from WebSocket server")

        # Cancel listener task first
        if self._listen_task and not self._listen_task.done():
            logger.info("Cancelling message listener task")
            self._listen_task.cancel()
            try:
                await asyncio.wait_for(self._listen_task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info("Message listener task cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Message listener task cancellation timeout")
            except Exception as e:
                logger.error(f"Error cancelling listener task: {e}", exc_info=True)

        # Close WebSocket connection
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=5.0)
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}", exc_info=True)
            finally:
                self.websocket = None

        self._is_connected = False

        # Clean up pending requests
        if self.pending_requests:
            logger.warning(f"Cleaning up {len(self.pending_requests)} pending requests")
            for request_id, future in self.pending_requests.items():
                if not future.done():
                    future.set_exception(ConnectionError("Connection closed"))
            self.pending_requests.clear()

    async def _listen_for_messages(self) -> None:
        """Listen for incoming messages from the server."""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        logger.info("Message listener task started, waiting for messages...")
        try:
            async for message in self.websocket:
                logger.info(f"Received message from server: {message[:200]}...")
                await self._handle_message(message)
        except asyncio.CancelledError:
            logger.info("Message listener task cancelled")
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed by server")
        except Exception as e:
            logger.error(f"Error listening for messages: {e}", exc_info=True)
            if self._should_reconnect:
                logger.info("Attempting to reconnect...")
                await self._reconnect()
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the server."""
        logger.info("Reconnection attempt initiated")
        await asyncio.sleep(2)  # Brief delay before reconnection
        try:
            await self.disconnect()
            await self.connect()
            logger.info("Reconnection successful")
        except Exception as e:
            logger.error(f"Reconnection failed: {e}", exc_info=True)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming message from server.

        Args:
            message: JSON message string from server
        """
        try:
            data = json.loads(message)
            logger.debug(f"Parsed message data: {data}")
            message_type = data.get("type")
            request_id = data.get("id")

            if message_type == "progress":
                logger.info(
                    f"Received progress update: stage={data.get('stage')}, status={data.get('status')}"
                )
                progress_update = ProgressUpdate(**data)
                if self.progress_callback:
                    self.progress_callback(progress_update)
            else:
                logger.info(f"Received response for request_id={request_id}")
                if request_id and request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    if not future.done():
                        try:
                            response = MCPResponse(**data)
                            logger.info(f"Setting result for request_id={request_id}")
                            future.set_result(response)
                        except Exception as e:
                            logger.error(
                                f"Error setting result for request_id={request_id}: {e}",
                                exc_info=True,
                            )
                            future.set_exception(e)
                else:
                    logger.warning(
                        f"Received response for unknown request_id={request_id}"
                    )

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def send_request(
        self, method: str, params: Dict[str, Any], request_id: Optional[str] = None
    ) -> MCPResponse:
        """Send an MCP request to the server.

        Args:
            method: MCP method name
            params: Method parameters
            request_id: Optional request ID (auto-generated if not provided)

        Returns:
            MCPResponse from the server

        Raises:
            RuntimeError: If not connected to server
            asyncio.TimeoutError: If response timeout
        """
        if not self._is_connected or not self.websocket:
            logger.error("Cannot send request: not connected to server")
            raise RuntimeError("Not connected to server")

        if request_id is None:
            request_id = str(uuid.uuid4())

        logger.info(
            f"Sending request: method={method}, request_id={request_id}, params={params}"
        )
        request = MCPRequest(method=method, params=params, id=request_id)
        request_json = request.json()
        logger.debug(f"Request JSON: {request_json}")

        future: asyncio.Future[MCPResponse] = asyncio.Future()
        self.pending_requests[request_id] = future
        logger.info(
            f"Created future for request_id={request_id}, pending_requests count={len(self.pending_requests)}"
        )

        try:
            await self.websocket.send(request_json)
            logger.info(
                f"Request sent successfully, waiting for response (timeout={self.timeout}s)"
            )
            response = await asyncio.wait_for(future, timeout=self.timeout)
            logger.info(f"Received response for request_id={request_id}")
            return response
        except asyncio.TimeoutError:
            logger.error(
                f"Request timeout for request_id={request_id} after {self.timeout}s"
            )
            self.pending_requests.pop(request_id, None)
            raise
        except Exception as e:
            logger.error(
                f"Error sending request for request_id={request_id}: {e}", exc_info=True
            )
            self.pending_requests.pop(request_id, None)
            raise

    async def ingest_document(
        self,
        file_path: str,
        document_name: Optional[str] = None,
        domain: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> MCPResponse:
        """Ingest a document using the MCP protocol.

        Args:
            file_path: Path to the document file
            document_name: Optional document name
            domain: Optional document domain
            request_id: Optional request ID

        Returns:
            MCPResponse with ingestion result

        Raises:
            RuntimeError: If not connected to server
            ValueError: If file_path is not provided
        """
        if not file_path:
            raise ValueError("file_path is required")

        params = {"file_path": file_path}
        if document_name:
            params["document_name"] = document_name
        if domain:
            params["domain"] = domain

        return await self.send_request("kbv2/ingest_document", params, request_id)

    async def query_text_to_sql(
        self, nl_query: str, request_id: Optional[str] = None
    ) -> MCPResponse:
        """Translate natural language query to SQL.

        Args:
            nl_query: Natural language query string
            request_id: Optional request ID

        Returns:
            MCPResponse with SQL translation result
        """
        params = {"nl_query": nl_query}
        return await self.send_request("kbv2/query_text_to_sql", params, request_id)

    async def search_entities(
        self, query: str, limit: int = 10, request_id: Optional[str] = None
    ) -> MCPResponse:
        """Search for entities.

        Args:
            query: Search query string
            limit: Maximum number of results
            request_id: Optional request ID

        Returns:
            MCPResponse with search results
        """
        params = {"query": query, "limit": limit}
        return await self.send_request("kbv2/search_entities", params, request_id)

    async def search_chunks(
        self, query: str, limit: int = 10, request_id: Optional[str] = None
    ) -> MCPResponse:
        """Search for document chunks.

        Args:
            query: Search query string
            limit: Maximum number of results
            request_id: Optional request ID

        Returns:
            MCPResponse with search results
        """
        params = {"query": query, "limit": limit}
        return await self.send_request("kbv2/search_chunks", params, request_id)

    async def get_document_status(
        self, document_id: str, request_id: Optional[str] = None
    ) -> MCPResponse:
        """Get status of a document.

        Args:
            document_id: Document ID to check
            request_id: Optional request ID

        Returns:
            MCPResponse with document status
        """
        params = {"document_id": document_id}
        return await self.send_request("kbv2/get_document_status", params, request_id)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
