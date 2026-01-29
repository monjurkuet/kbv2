import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from knowledge_base.persistence.v1.schema import Document, Chunk, DocumentStatus
from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.ingestion.v1.partitioning_service import PartitionedChunk

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.get_session.return_value.__aenter__.return_value = AsyncMock(spec=AsyncSession)
    mock.get_session.return_value.__aexit__ = AsyncMock()
    return mock

@pytest.fixture
def orchestrator(mock_vector_store):
    orchestrator = IngestionOrchestrator()
    orchestrator._vector_store = mock_vector_store
    orchestrator._observability = MagicMock()
    return orchestrator

@pytest.mark.asyncio
async def test_chunk_metadata_persistence(orchestrator, mock_vector_store):
    """Test that chunk metadata is correctly persisted to the chunk_metadata column."""
    document = Document(
        id=uuid4(),
        name="test.md",
        status=DocumentStatus.PENDING
    )
    
    mock_chunks = [
        PartitionedChunk(
            text="Chunk 1 content",
            token_count=10,
            chunk_index=0,
            metadata={"source": "header", "page": 1}
        )
    ]
    
    with patch("knowledge_base.ingestion.v1.partitioning_service.PartitioningService.partition_and_chunk", 
               new_callable=AsyncMock) as mock_partition:
        mock_partition.return_value = mock_chunks
        
        session_mock = mock_vector_store.get_session.return_value.__aenter__.return_value
        
        # We need to mock session.get and session.merge to avoid errors
        session_mock.get.return_value = document
        session_mock.merge.return_value = document
        
        await orchestrator._partition_document(document, "test.md")
        
        # Verify that session.add was called with a Chunk object having chunk_metadata populated
        added_objects = [call.args[0] for call in session_mock.add.call_args_list]
        chunk_objects = [obj for obj in added_objects if isinstance(obj, Chunk)]
        
        assert len(chunk_objects) == 1
        assert chunk_objects[0].chunk_metadata == {"source": "header", "page": 1}
        assert chunk_objects[0].token_count == 10
