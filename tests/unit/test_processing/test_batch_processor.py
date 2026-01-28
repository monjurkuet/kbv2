"""Tests for batch processor functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge_base.processing.batch_processor import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
)


class TestBatchConfig:
    """Tests for BatchConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.batch_size == 10
        assert config.max_workers == 4
        assert config.timeout_seconds == 300

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchConfig(batch_size=20, max_workers=8, timeout_seconds=600)
        assert config.batch_size == 20
        assert config.max_workers == 8
        assert config.timeout_seconds == 600

    def test_validation_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValueError):
            BatchConfig(batch_size=0)

    def test_validation_max_workers(self):
        """Test max_workers validation."""
        with pytest.raises(ValueError):
            BatchConfig(max_workers=0)


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a batch processor instance."""
        return BatchProcessor(batch_size=5, max_workers=4)

    @pytest.fixture
    def sample_items(self):
        """Create sample items for testing."""
        return [f"item_{i}" for i in range(12)]

    @pytest.mark.asyncio
    async def test_process_batch_empty(self, processor):
        """Test processing empty list returns empty result."""
        result = await processor.process_batch([], MagicMock())
        assert isinstance(result, BatchResult)
        assert len(result.successful) == 0
        assert len(result.failed) == 0
        assert result.total_processed == 0

    @pytest.mark.asyncio
    async def test_process_batch_sync_processor(self, processor, sample_items):
        """Test batch processing with sync processor."""
        call_count = 0

        def sync_processor(batch):
            nonlocal call_count
            call_count += 1
            return [f"result_{item}" for item in batch]

        result = await processor.process_batch(sample_items, sync_processor)

        assert result.total_processed == 12
        assert len(result.successful) == 12
        assert len(result.failed) == 0
        assert call_count == 3  # 12 items / 5 batch_size = 3 batches

    @pytest.mark.asyncio
    async def test_process_batch_async_processor(self, processor, sample_items):
        """Test batch processing with async processor."""

        async def async_processor(batch):
            return [f"result_{item}" for item in batch]

        result = await processor.process_batch(sample_items, async_processor)

        assert result.total_processed == 12
        assert len(result.successful) == 12

    @pytest.mark.asyncio
    async def test_process_batch_with_error(self, processor, sample_items):
        """Test batch processing handles errors correctly."""

        def failing_processor(batch):
            if len(batch) > 0:
                raise ValueError("Processing failed")
            return [f"result_{item}" for item in batch]

        result = await processor.process_batch(sample_items, failing_processor)

        assert result.total_processed == 12
        assert len(result.failed) > 0

    @pytest.mark.asyncio
    async def test_process_batch_with_extra_args(self, processor, sample_items):
        """Test batch processing passes extra arguments."""
        received_args = []

        def processor_with_args(batch, multiplier):
            received_args.append((len(batch), multiplier))
            return [item * multiplier for item in batch]

        await processor.process_batch(
            ["a", "b", "c"], processor_with_args, multiplier=2
        )

        assert len(received_args) > 0
        assert all(mult == 2 for _, mult in received_args)

    @pytest.mark.asyncio
    async def test_process_parallel_empty(self, processor):
        """Test parallel processing empty list."""
        result = await processor.process_parallel([], MagicMock())
        assert result.total_processed == 0

    @pytest.mark.asyncio
    async def test_process_parallel_sync_processor(self, processor, sample_items):
        """Test parallel processing with sync processor."""

        def sync_processor(item):
            return f"processed_{item}"

        result = await processor.process_parallel(sample_items, sync_processor)

        assert result.total_processed == 12
        assert len(result.successful) == 12
        assert all("processed_" in str(s) for s in result.successful)

    @pytest.mark.asyncio
    async def test_process_parallel_preserves_order(self, processor):
        """Test parallel processing preserves item order."""
        items = [3, 1, 4, 1, 5, 9, 2, 6]

        def sync_processor(item):
            return item * 2

        result = await processor.process_parallel(items, sync_processor)

        assert result.successful == [6, 2, 8, 2, 10, 18, 4, 12]

    @pytest.mark.asyncio
    async def test_process_parallel_with_error(self, processor):
        """Test parallel processing handles errors."""

        def failing_processor(item):
            if item == "item_5":
                raise ValueError("Failed")
            return f"result_{item}"

        items = [f"item_{i}" for i in range(10)]
        result = await processor.process_parallel(items, failing_processor)

        assert result.total_processed == 10
        assert 5 in result.failed

    @pytest.mark.asyncio
    async def test_process_with_retry_empty(self, processor):
        """Test retry processing empty list."""
        result = await processor.process_with_retry([], MagicMock())
        assert result.total_processed == 0

    @pytest.mark.asyncio
    async def test_process_with_retry_success(self, processor):
        """Test retry processing succeeds on first try."""

        def processor_func(item):
            return f"result_{item}"

        items = ["a", "b", "c"]
        result = await processor.process_with_retry(
            items, processor_func, max_retries=3
        )

        assert result.total_processed == 3
        assert len(result.failed) == 0

    @pytest.mark.asyncio
    async def test_process_with_retry_eventual_success(self, processor):
        """Test retry processing eventually succeeds."""

        def flaky_processor(item):
            nonlocal call_count
            call_count += 1
            if call_count < len(item):
                raise ValueError("Temporary failure")
            return f"result_{item}"

        call_count = 0
        items = ["a", "b"]
        result = await processor.process_with_retry(
            items, flaky_processor, max_retries=3, backoff_factor=0.01
        )

        assert len(result.failed) == 0

    @pytest.mark.asyncio
    async def test_process_with_retry_max_exceeded(self, processor):
        """Test retry processing fails after max retries."""

        def always_failing_processor(item):
            raise ValueError("Always fails")

        items = ["a", "b"]
        result = await processor.process_with_retry(
            items, always_failing_processor, max_retries=2, backoff_factor=0.01
        )

        assert len(result.failed) > 0

    def test_get_stats(self, processor):
        """Test getting processor statistics."""
        stats = processor.get_stats()
        assert stats["batch_size"] == 5
        assert stats["max_workers"] == 4
        assert stats["timeout_seconds"] == 300


class TestBatchResult:
    """Tests for BatchResult model."""

    def test_default_values(self):
        """Test default result values."""
        result = BatchResult()
        assert len(result.successful) == 0
        assert len(result.failed) == 0
        assert result.total_processed == 0
        assert result.total_time_ms == 0.0

    def test_with_values(self):
        """Test result with values."""
        result = BatchResult(
            successful=["a", "b", "c"],
            failed={0: {"error": "test"}},
            total_processed=4,
            total_time_ms=100.0,
        )
        assert len(result.successful) == 3
        assert 0 in result.failed
        assert result.total_processed == 4
