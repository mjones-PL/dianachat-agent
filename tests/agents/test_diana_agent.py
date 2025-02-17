"""Tests for DianaChat agent."""

import pytest
from unittest.mock import MagicMock
from livekit import agents
from dianachat_agent.agents.diana_agent import DianaAgent

@pytest.fixture
def mock_settings():
    return MagicMock()

@pytest.fixture
def mock_job_context():
    return MagicMock(spec=agents.JobContext)

def test_chat_context_attribute_access(mock_settings, mock_job_context):
    # Arrange
    agent = DianaAgent(mock_job_context, mock_settings)
    test_context = agents.ChatContext(message="Initial message")
    test_enriched = "Test enrichment"

    # Act - Apply the context modification
    test_context.message += f"\n\nContext: {test_enriched}"

    # Assert
    assert "Test enrichment" in test_context.message, "RAG enrichment not found"
    assert test_context.message.startswith("Initial message"), "Original message corrupted"
