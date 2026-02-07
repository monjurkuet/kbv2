"""LLM call logging wrapper for comprehensive pipeline logging."""

import functools
import logging
import time
from typing import Any, Callable, Optional

# Import the consolidated logger from extraction_logging
from knowledge_base.intelligence.v1.extraction_logging import ingestion_logger

# Use the consolidated logger for all LLM call logging
llm_call_logger = ingestion_logger


def log_llm_call(agent_name: Optional[str] = None):
    """Decorator to log all LLM calls with detailed information.

    Usage:
        @log_llm_call(agent_name="PerceptionAgent")
        async def my_method(self, ...):
            return await self._gateway.complete(...)

    Args:
        agent_name: Name of the agent making the call (for logging context)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the instance (self) and gateway
            instance = args[0]
            gateway = getattr(instance, "_gateway", None) or getattr(
                instance, "gateway", None
            )

            # Get document_id if available
            document_id = getattr(instance, "_document_id", None) or getattr(
                instance, "document_id", None
            )

            # Build the call signature for logging
            call_args = []

            # Extract messages or prompt
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)
            if messages:
                call_args.append(f"messages={len(messages)} messages")

            prompt = kwargs.get("prompt") or (args[1] if len(args) > 1 else None)
            if prompt:
                call_args.append(f"prompt='{prompt[:50]}...'")

            # Extract other parameters
            model = kwargs.get("model")
            if model:
                call_args.append(f"model={model}")

            temperature = kwargs.get("temperature")
            if temperature:
                call_args.append(f"temperature={temperature}")

            call_signature = ", ".join(call_args)

            # Generate a call ID
            import uuid

            call_id = str(uuid.uuid4())[:8]

            # Log the call
            start_time = time.time()
            llm_call_logger.info(
                f"🤖 LLM CALL [{call_id}] {'[' + agent_name + ']' if agent_name else ''}:\n"
                f"   📞 Function: {func.__name__}\n"
                f"   📄 Document: {document_id or 'N/A'}\n"
                f"   📝 Arguments: {call_signature}\n"
                f"   🔍 Gateway: {gateway and gateway.__class__.__name__ or 'N/A'}"
            )

            try:
                # Make the actual call
                result = await func(*args, **kwargs)

                elapsed = time.time() - start_time

                # Log the response
                if hasattr(result, "choices") and result.choices:
                    response_preview = str(
                        result.choices[0].get("message", {}).get("content", "")
                    )[:100]
                else:
                    response_preview = str(result)[:100]

                llm_call_logger.info(
                    f"💬 LLM RESPONSE [{call_id}] {'[' + agent_name + ']' if agent_name else ''}:\n"
                    f"   ✨ Status: Success\n"
                    f"   ⏱️  Time: {elapsed:.3f}s\n"
                    f"   📝 Preview: {response_preview}..."
                )

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                llm_call_logger.error(
                    f"❌ LLM ERROR [{call_id}] {'[' + agent_name + ']' if agent_name else ''}:\n"
                    f"   ✨ Status: Failed\n"
                    f"   ⏱️  Time: {elapsed:.3f}s\n"
                    f"   💥 Error: {str(e)}"
                )
                raise

        return wrapper

    return decorator


class LLMCallLogger:
    """Context manager for logging LLM calls with additional context."""

    def __init__(
        self,
        agent_name: str,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        step_info: Optional[str] = None,
    ):
        """Initialize LLM call logger.

        Args:
            agent_name: Name of the agent making the call
            document_id: Document ID (optional)
            chunk_id: Chunk ID (optional)
            step_info: Step information (e.g., "Step 2/5")
        """
        self.agent_name = agent_name
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.step_info = step_info
        self.call_id = None
        self.start_time = None

    async def __aenter__(self):
        """Enter context - log the LLM call."""
        import uuid

        self.call_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()

        context_parts = []
        if self.document_id:
            context_parts.append(f"doc: {self.document_id}")
        if self.chunk_id:
            context_parts.append(f"chunk: {self.chunk_id}")
        if self.step_info:
            context_parts.append(f"step: {self.step_info}")

        context_str = f" [{' | '.join(context_parts)}]" if context_parts else ""

        llm_call_logger.info(
            f"🤖 LLM CALL START [{self.call_id}] [{self.agent_name}]{context_str}\n"
            f"   ⏱️  Ready to make LLM call..."
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log the response/error."""
        elapsed = time.time() - self.start_time

        if exc_type is None:
            llm_call_logger.info(
                f"💬 LLM CALL END [{self.call_id}] [{self.agent_name}]:\n"
                f"   ✨ Status: Success\n"
                f"   ⏱️  Duration: {elapsed:.3f}s"
            )
        else:
            llm_call_logger.error(
                f"❌ LLM CALL ERROR [{self.call_id}] [{self.agent_name}]:\n"
                f"   ✨ Status: Failed\n"
                f"   ⏱️  Duration: {elapsed:.3f}s\n"
                f"   💥 Error: {str(exc_val)}"
            )


def log_llm_result(
    agent_name: str,
    result: Any,
    document_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Log an LLM result with detailed information.

    Args:
        agent_name: Name of the agent
        result: The result from the LLM call
        document_id: Document ID (optional)
        metadata: Additional metadata (optional)
    """
    meta_str = f"\n   📊 Metadata: {metadata}" if metadata else ""

    if hasattr(result, "choices") and result.choices:
        content = result.choices[0].get("message", {}).get("content", "")
        model = getattr(result, "model", "unknown")

        if "choices" in result and result.choices:
            choice = result.choices[0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                content = message.get("content", "")
            else:
                # Handle Pydantic model
                content = str(choice)[:200]
        else:
            content = str(result)

        llm_call_logger.info(
            f"📝 LLM RESULT [{agent_name}]:\n"
            f"   🧠 Model: {model}\n"
            f"   📄 Document: {document_id or 'N/A'}\n"
            f"   💬 Response: {str(content)[:200]}...{meta_str}"
        )
    else:
        llm_call_logger.info(
            f"📝 LLM RESULT [{agent_name}]:\n"
            f"   📄 Document: {document_id or 'N/A'}\n"
            f"   📦 Raw: {str(result)[:200]}...{meta_str}"
        )
