"""Comprehensive logging utilities for multi-agent extraction."""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID

# Create a single, consolidated logger for ALL ingestion activities
# This logger will handle: stage progress, LLM calls, entity extraction, errors
ingestion_logger = logging.getLogger("knowledge_base.ingestion")
ingestion_logger.setLevel(logging.DEBUG)

# Remove any existing handlers to prevent duplication
for handler in ingestion_logger.handlers[:]:
    ingestion_logger.removeHandler(handler)

# ONE file handler for everything
file_handler = logging.FileHandler("/tmp/kbv2_ingestion.log")
file_handler.setLevel(logging.DEBUG)

# Console handler for real-time monitoring
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# ONE formatter for everything - clear and detailed
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)7s | [%(filename)s:%(lineno)3d] | %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

ingestion_logger.addHandler(file_handler)
ingestion_logger.addHandler(console_handler)


class ExtractionLogger:
    """Comprehensive logger for multi-agent extraction process."""

    def __init__(self, document_id: UUID, document_name: str):
        """Initialize logger for a specific document.

        Args:
            document_id: Document UUID
            document_name: Document filename
        """
        self.document_id = str(document_id)
        self.document_name = document_name
        self.start_time = time.time()
        self.llm_call_count = 0
        self.total_tokens = 0
        self.model_usage: Dict[str, int] = {}

    def log_stage_start(self, stage_name: str, total_steps: int = None):
        """Log the start of a processing stage.

        Args:
            stage_name: Name of the stage (e.g., "Perception", "Enhancement")
            total_steps: Total number of steps in this stage
        """
        step_info = f" ({total_steps} steps)" if total_steps else ""
        ingestion_logger.info(
            f"📄 [{self.document_name}] 🔄 STAGE START: {stage_name}{step_info}"
        )
        self._log_websocket(
            "stage_start",
            {
                "stage": stage_name,
                "total_steps": total_steps,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_stage_progress(self, stage_name: str, current_step: int, total_steps: int):
        """Log progress within a stage.

        Args:
            stage_name: Name of the stage
            current_step: Current step number
            total_steps: Total number of steps
        """
        progress = (current_step / total_steps) * 100
        ingestion_logger.info(
            f"📄 [{self.document_name}] ⏳ STAGE PROGRESS: {stage_name} - "
            f"Step {current_step}/{total_steps} ({progress:.1f}%)"
        )
        self._log_websocket(
            "stage_progress",
            {
                "stage": stage_name,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress_percent": progress,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_stage_complete(self, stage_name: str, result_summary: str):
        """Log completion of a stage.

        Args:
            stage_name: Name of the stage
            result_summary: Summary of results
        """
        elapsed = time.time() - self.start_time
        ingestion_logger.info(
            f"📄 [{self.document_name}] ✅ STAGE COMPLETE: {stage_name} - {result_summary} "
            f"(elapsed: {elapsed:.2f}s)"
        )
        self._log_websocket(
            "stage_complete",
            {
                "stage": stage_name,
                "result_summary": result_summary,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_llm_call(
        self,
        agent_name: str,
        model: str,
        prompt_preview: str,
        prompt_tokens: int,
        chunk_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ):
        """Log an LLM call.

        Args:
            agent_name: Name of the agent making the call (e.g., "PerceptionAgent")
            model: Model being used
            prompt_preview: Preview of the prompt (first 100 chars)
            prompt_tokens: Number of tokens in the prompt
            chunk_id: ID of the chunk being processed (if applicable)
            iteration: Iteration number (for iterative processes)
        """
        self.llm_call_count += 1
        self.total_tokens += prompt_tokens
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

        iteration_info = f" (iteration {iteration})" if iteration else ""
        chunk_info = f" [chunk: {chunk_id}]" if chunk_id else ""

        ingestion_logger.info(
            f"📄 [{self.document_name}] 🤖 LLM CALL #{self.llm_call_count}: {agent_name} -> {model}{chunk_info}{iteration_info}\n"
            f"   Prompt: {prompt_preview[:100]}...\n"
            f"   Tokens: {prompt_tokens}"
        )
        self._log_websocket(
            "llm_call",
            {
                "call_number": self.llm_call_count,
                "agent": agent_name,
                "model": model,
                "chunk_id": chunk_id,
                "iteration": iteration,
                "prompt_preview": prompt_preview[:100],
                "prompt_tokens": prompt_tokens,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_llm_response(
        self,
        agent_name: str,
        response_preview: str,
        response_tokens: int,
        model: str,
        chunk_id: Optional[str] = None,
    ):
        """Log an LLM response.

        Args:
            agent_name: Name of the agent receiving the response
            response_preview: Preview of the response (first 100 chars)
            response_tokens: Number of tokens in the response
            model: Model that generated the response
            chunk_id: ID of the chunk being processed
        """
        self.total_tokens += response_tokens
        chunk_info = f" [chunk: {chunk_id}]" if chunk_id else ""

        ingestion_logger.info(
            f"📄 [{self.document_name}] 💬 LLM RESPONSE: {agent_name} <- {model}{chunk_info}\n"
            f"   Response: {response_preview[:100]}...\n"
            f"   Tokens: {response_tokens}"
        )
        self._log_websocket(
            "llm_response",
            {
                "agent": agent_name,
                "model": model,
                "chunk_id": chunk_id,
                "response_preview": response_preview[:100],
                "response_tokens": response_tokens,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_entity_extraction(self, chunk_id: str, entity_count: int, entity_types: List[str]):
        """Log entity extraction results.

        Args:
            chunk_id: ID of the chunk
            entity_count: Number of entities extracted
            entity_types: List of entity types extracted
        """
        unique_types = list(set(entity_types))
        ingestion_logger.info(
            f"📄 [{self.document_name}] 🎯 ENTITIES EXTRACTED: {entity_count} entities "
            f"(types: {', '.join(unique_types)}) [chunk: {chunk_id}]"
        )
        self._log_websocket(
            "entity_extraction",
            {
                "chunk_id": chunk_id,
                "entity_count": entity_count,
                "entity_types": unique_types,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_relationship_extraction(self, chunk_id: str, relationship_count: int):
        """Log relationship extraction results.

        Args:
            chunk_id: ID of the chunk
            relationship_count: Number of relationships extracted
        """
        ingestion_logger.info(
            f"📄 [{self.document_name}] 🔗 RELATIONSHIPS EXTRACTED: {relationship_count} "
            f"[chunk: {chunk_id}]"
        )
        self._log_websocket(
            "relationship_extraction",
            {
                "chunk_id": chunk_id,
                "relationship_count": relationship_count,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_quality_score(self, score: float, level: str, feedback: str):
        """Log quality evaluation score.

        Args:
            score: Quality score (0.0 to 1.0)
            level: Quality level (low, medium, high)
            feedback: Feedback string
        """
        ingestion_logger.info(
            f"📄 [{self.document_name}] 🏆 QUALITY SCORE: {score:.3f} ({level}) - {feedback}"
        )
        self._log_websocket(
            "quality_score",
            {
                "score": score,
                "level": level,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_error(self, error_message: str, agent_name: Optional[str] = None):
        """Log an error.

        Args:
            error_message: Error message
            agent_name: Name of the agent that encountered the error
        """
        agent_info = f" [{agent_name}]" if agent_name else ""
        ingestion_logger.error(
            f"📄 [{self.document_name}] ❌ ERROR{agent_info}: {error_message}"
        )
        self._log_websocket(
            "error",
            {
                "error": error_message,
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def log_summary(self):
        """Log a summary of the entire extraction process."""
        elapsed = time.time() - self.start_time
        model_usage_str = json.dumps(self.model_usage, indent=2)

        ingestion_logger.info(
            f"📄 [{self.document_name}] 📊 EXTRACTION SUMMARY:\n"
            f"   Total time: {elapsed:.2f} seconds\n"
            f"   LLM calls: {self.llm_call_count}\n"
            f"   Total tokens: {self.total_tokens}\n"
            f"   Model usage:\n{model_usage_str}\n"
            f"   Status: {'✅ Complete' if elapsed > 60 else '⏸️ Incomplete'}"
        )
        self._log_websocket(
            "extraction_summary",
            {
                "total_time_seconds": elapsed,
                "llm_calls": self.llm_call_count,
                "total_tokens": self.total_tokens,
                "model_usage": self.model_usage,
                "status": "complete" if elapsed > 60 else "incomplete",
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _log_websocket(self, event_type: str, data: Dict[str, Any]):
        """Log to WebSocket (if available).

        Args:
            event_type: Type of event
            data: Event data
        """
        # Note: This would integrate with the WebSocket client
        # For now, we just prepare the message format
        log_entry = {
            "type": "extraction_log",
            "document_id": self.document_id,
            "document_name": self.document_name,
            "event_type": event_type,
            "data": data,
        }

        # The actual WebSocket sending would be done by the orchestrator
        # This method just standardizes the log format

        # Broadcast via WebSocket if broadcast function is available
        if _webhook_broadcast:
            try:
                asyncio.create_task(_webhook_broadcast(json.dumps(log_entry)))
            except Exception:
                # Silently fail if broadcast doesn't work
                pass

        return log_entry


# Global broadcast function for WebSocket messages
_webhook_broadcast: Optional[Callable[[str], Any]] = None


def set_websocket_broadcast(broadcast_func: Optional[Callable[[str], Any]]) -> None:
    """Set the global WebSocket broadcast function.

    Args:
        broadcast_func: Function that takes a JSON string and broadcasts it to WebSocket clients
    """
    global _webhook_broadcast
    _webhook_broadcast = broadcast_func


# Global logger instances
_loggers: Dict[str, ExtractionLogger] = {}


def get_ingestion_logger(document_id: UUID, document_name: str) -> ExtractionLogger:
    """Get or create an extraction logger for a document.

    Args:
        document_id: Document UUID
        document_name: Document filename

    Returns:
        ExtractionLogger instance
    """
    key = str(document_id)
    if key not in _loggers:
        _loggers[key] = ExtractionLogger(document_id, document_name)
    return _loggers[key]


def remove_ingestion_logger(document_id: UUID):
    """Remove an extraction logger for a document.

    Args:
        document_id: Document UUID
    """
    key = str(document_id)
    if key in _loggers:
        del _loggers[key]
