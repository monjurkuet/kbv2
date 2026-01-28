"""Batch processing module for efficient LLM and embedding calls.

This module provides batch processing capabilities for handling large volumes
of items through async processing with configurable batch sizes, parallel
execution support, and comprehensive error handling.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import as_completed
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchConfig(BaseModel):
    """Configuration for batch processing.

    Attributes:
        batch_size: Number of items to process per batch.
        max_workers: Maximum number of parallel workers for parallel processing.
        timeout_seconds: Timeout for individual processing operations.
    """

    batch_size: int = Field(default=10, ge=1, description="Items per batch")
    max_workers: int = Field(default=4, ge=1, description="Max parallel workers")
    timeout_seconds: int = Field(default=300, ge=1, description="Operation timeout")


class BatchResult(BaseModel):
    """Result from batch processing operations.

    Attributes:
        successful: List of successfully processed items.
        failed: Dictionary mapping item indices to error information.
        total_processed: Total number of items processed.
        total_time_ms: Total processing time in milliseconds.
    """

    successful: List[Any] = Field(
        default_factory=list, description="Successful results"
    )
    failed: Dict[int, Dict[str, Any]] = Field(
        default_factory=dict, description="Failed items with errors"
    )
    total_processed: int = Field(default=0, description="Total items processed")
    total_time_ms: float = Field(default=0.0, description="Processing time in ms")


class BatchProcessor:
    """Batch processor for efficient LLM and embedding calls.

    This processor handles large volumes of items by processing them in batches
    or in parallel using thread pools. It supports both async and sync processor
    functions and provides comprehensive error tracking.

    Example:
        >>> processor = BatchProcessor(batch_size=20, max_workers=8)
        >>> result = await processor.process_batch(
        ...     items=texts,
        ...     processor=embedding_client.embed_batch
        ... )
    """

    def __init__(self, config: Optional[BatchConfig] = None) -> None:
        """Initialize the batch processor.

        Args:
            config: Optional batch configuration. Uses defaults if not provided.
        """
        self.config = config or BatchConfig()
        self._default_executor: Optional[ThreadPoolExecutor] = None

    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs,
    ) -> BatchResult:
        """Process items in batches sequentially.

        Processes items in configurable batch sizes, handling both async and
        sync processor functions. Tracks successful results and failures.

        Args:
            items: List of items to process.
            processor: Callable to process each batch. Can be async or sync.
            *args: Additional positional arguments for processor.
            **kwargs: Additional keyword arguments for processor.

        Returns:
            BatchResult containing successful items, failures, and metrics.
        """
        start_time = time.time()
        successful: List[Any] = []
        failed: Dict[int, Dict[str, Any]] = {}

        if not items:
            return BatchResult(
                successful=[],
                failed={},
                total_processed=0,
                total_time_ms=0.0,
            )

        for i in range(0, len(items), self.config.batch_size):
            batch = items[i : i + self.config.batch_size]
            batch_start = i

            try:
                if asyncio.iscoroutinefunction(processor):
                    results = await processor(batch, *args, **kwargs)
                else:
                    results = processor(batch, *args, **kwargs)

                if isinstance(results, list):
                    successful.extend(results)
                else:
                    successful.append(results)

            except Exception as e:
                logger.error(f"Batch processing failed at index {i}: {e}")
                for j, item in enumerate(batch):
                    failed[batch_start + j] = {
                        "item": item,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

        total_time = (time.time() - start_time) * 1000
        return BatchResult(
            successful=successful,
            failed=failed,
            total_processed=len(items),
            total_time_ms=total_time,
        )

    async def process_parallel(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs,
    ) -> BatchResult:
        """Process items in parallel using ThreadPoolExecutor.

        Uses a thread pool to process items concurrently, suitable for
        CPU-bound or sync operations. Supports timeout and error tracking.

        Args:
            items: List of items to process.
            processor: Sync callable to process each item.
            *args: Additional positional arguments for processor.
            **kwargs: Additional keyword arguments for processor.

        Returns:
            BatchResult with parallel processing results.
        """
        start_time = time.time()
        successful: List[tuple[int, Any]] = []
        failed: Dict[int, Dict[str, Any]] = {}

        if not items:
            return BatchResult(
                successful=[],
                failed={},
                total_processed=0,
                total_time_ms=0.0,
            )

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(processor, item, *args, **kwargs): i
                for i, item in enumerate(items)
            }

            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    successful.append((idx, result))
                except Exception as e:
                    logger.error(f"Parallel processing failed for index {idx}: {e}")
                    failed[idx] = {
                        "item": items[idx],
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

        successful.sort(key=lambda x: x[0])

        total_time = (time.time() - start_time) * 1000
        return BatchResult(
            successful=[s[1] for s in successful],
            failed=failed,
            total_processed=len(items),
            total_time_ms=total_time,
        )

    async def process_with_retry(
        self,
        items: List[Any],
        processor: Callable,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        *args,
        **kwargs,
    ) -> BatchResult:
        """Process items with automatic retry on failure.

        Retries failed items with exponential backoff, useful for handling
        transient errors in external API calls.

        Args:
            items: List of items to process.
            processor: Callable to process each item or batch.
            max_retries: Maximum retry attempts per batch.
            backoff_factor: Exponential backoff multiplier in seconds.
            *args: Additional positional arguments for processor.
            **kwargs: Additional keyword arguments for processor.

        Returns:
            BatchResult with all processed items including retries.
        """
        start_time = time.time()
        successful: List[Any] = []
        failed: Dict[int, Dict[str, Any]] = {}

        if not items:
            return BatchResult(
                successful=[],
                failed={},
                total_processed=0,
                total_time_ms=0.0,
            )

        remaining_items = list(enumerate(items))
        retry_counts: Dict[int, int] = {i: 0 for i in range(len(items))}

        while remaining_items:
            new_remaining: List[tuple[int, Any]] = []

            for idx, item in remaining_items:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor([item], *args, **kwargs)
                    else:
                        result = processor([item], *args, **kwargs)

                    if isinstance(result, list):
                        successful.extend(result)
                    else:
                        successful.append(result)

                except Exception as e:
                    retry_counts[idx] += 1
                    if retry_counts[idx] < max_retries:
                        new_remaining.append((idx, item))
                        wait_time = backoff_factor * (2 ** (retry_counts[idx] - 1))
                        logger.warning(
                            f"Retry {retry_counts[idx]}/{max_retries} for index {idx} "
                            f"after {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for index {idx}: {e}")
                        failed[idx] = {
                            "item": item,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "retries": retry_counts[idx],
                        }

            remaining_items = new_remaining

        total_time = (time.time() - start_time) * 1000
        return BatchResult(
            successful=successful,
            failed=failed,
            total_processed=len(items),
            total_time_ms=total_time,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration.

        Returns:
            Dictionary containing configuration and performance metrics.
        """
        return {
            "batch_size": self.config.batch_size,
            "max_workers": self.config.max_workers,
            "timeout_seconds": self.config.timeout_seconds,
        }

    async def close(self) -> None:
        """Cleanup resources and close executors."""
        if self._default_executor:
            self._default_executor.shutdown(wait=True)
            self._default_executor = None
