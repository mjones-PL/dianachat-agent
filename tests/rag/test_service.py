"""Tests for RAG service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dianachat_agent.rag.service import RAGService
from dianachat_agent.config import AgentSettings

@pytest.fixture
def mock_settings():
    return AgentSettings(
        rag_model="text-embedding-3-small",
        rag_index_path="/tmp/test_index.ann"
    )

@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    client.embeddings.create = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=[0.1]*384)]
    ))
    return client

@pytest.fixture
async def rag_service(mock_settings, mock_openai_client):
    service = RAGService(mock_settings)
    service.client = mock_openai_client
    service.index = MagicMock()
    service.index.get_nns_by_vector = MagicMock(return_value=([1], [0.9]))
    service.paragraphs_by_id = {1: "test paragraph"}
    await service.initialize()
    return service

@pytest.mark.asyncio
async def test_get_similar_texts(rag_service):
    results = await rag_service.get_similar_texts("test query")
    assert len(results) == 1
    assert results[0][0] == "test paragraph"
    rag_service.client.embeddings.create.assert_awaited_once()
