import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from knowledge_base.persistence.v1.schema import Entity, Chunk
from knowledge_base.persistence.v1.vector_store import VectorStore, VectorStoreConfig

@pytest.fixture
def mock_session_factory():
    factory = MagicMock()
    session = AsyncMock(spec=AsyncSession)
    # Ensure execute returns a mock that has scalar_one_or_none
    session.execute = AsyncMock() 
    factory.return_value.__aenter__.return_value = session
    factory.return_value.__aexit__ = AsyncMock()
    return factory, session

@pytest.fixture
def vector_store(mock_session_factory):
    factory, session = mock_session_factory
    store = VectorStore(config=VectorStoreConfig())
    store._session_factory = factory
    return store, session

@pytest.mark.asyncio
async def test_update_entity_embeddings_batch(vector_store):
    """Test that multiple entity embeddings can be updated in one batch."""
    store, session = vector_store
    
    entity_id1 = str(uuid4())
    entity_id2 = str(uuid4())
    updates = [
        (entity_id1, [0.1, 0.2, 0.3]),
        (entity_id2, [0.4, 0.5, 0.6])
    ]
    
    mock_entity1 = Entity(id=entity_id1, name="E1")
    mock_entity2 = Entity(id=entity_id2, name="E2")
    
    async def mock_execute(stmt):
        mock_result = MagicMock()
        stmt_str = str(stmt)
        if entity_id1 in stmt_str:
            mock_result.scalar_one_or_none.return_value = mock_entity1
        elif entity_id2 in stmt_str:
            mock_result.scalar_one_or_none.return_value = mock_entity2
        else:
            mock_result.scalar_one_or_none.return_value = None
        return mock_result
    
    session.execute.side_effect = mock_execute
    
    await store.update_entity_embeddings_batch(updates)
    
    assert mock_entity1.embedding == [0.1, 0.2, 0.3]
    assert mock_entity2.embedding == [0.4, 0.5, 0.6]
    assert session.commit.called

@pytest.mark.asyncio
async def test_update_chunk_embeddings_batch(vector_store):
    """Test that multiple chunk embeddings can be updated in one batch."""
    store, session = vector_store
    
    chunk_id1 = str(uuid4())
    updates = [(chunk_id1, [0.7, 0.8, 0.9])]
    
    mock_chunk1 = Chunk(id=chunk_id1, text="C1")
    
    async def mock_execute(stmt):
        mock_result = MagicMock()
        if chunk_id1 in str(stmt):
            mock_result.scalar_one_or_none.return_value = mock_chunk1
        else:
            mock_result.scalar_one_or_none.return_value = None
        return mock_result
    
    session.execute.side_effect = mock_execute
    
    await store.update_chunk_embeddings_batch(updates)
    
    assert mock_chunk1.embedding == [0.7, 0.8, 0.9]
    assert session.commit.called
