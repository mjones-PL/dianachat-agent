"""Tests for the main entry point of DianaChat Agent."""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

from dianachat_agent.main import entrypoint
from dianachat_agent.config.agent_config import AgentSettings

@pytest.fixture
def mock_env_vars():
    """Setup mock environment variables."""
    env_vars = {
        'LIVEKIT_URL': 'wss://test.livekit.io',
        'LIVEKIT_API_KEY': 'test_key',
        'LIVEKIT_API_SECRET': 'test_secret',
        'OPENAI_API_KEY': 'test_openai_key',
        'DEEPGRAM_API_KEY': 'test_deepgram_key'
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_job_context():
    """Create a mock JobContext."""
    context = MagicMock()
    context.connect = AsyncMock()
    context.wait_for_participant = AsyncMock()
    context.wait_for_participant.return_value = MagicMock(identity="test_user")
    return context

@pytest.mark.asyncio
async def test_entrypoint_successful_startup(mock_env_vars, mock_job_context):
    """Test successful startup of the agent."""
    with patch('dianachat_agent.main.DianaAgent') as mock_agent_class:
        # Setup mock agent
        mock_agent = MagicMock()
        mock_agent.start = AsyncMock()
        mock_agent.wait = AsyncMock()
        mock_agent_class.return_value = mock_agent

        # Run entrypoint
        await entrypoint(mock_job_context)

        # Verify LiveKit connection
        mock_job_context.connect.assert_called_once()
        mock_job_context.wait_for_participant.assert_called_once()

        # Verify agent creation and startup
        mock_agent_class.assert_called_once()
        mock_agent.start.assert_called_once()
        mock_agent.wait.assert_called_once()

@pytest.mark.asyncio
async def test_entrypoint_missing_env_file(mock_job_context):
    """Test handling of missing .env file."""
    with patch('pathlib.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            await entrypoint(mock_job_context)

@pytest.mark.asyncio
async def test_entrypoint_connection_error(mock_env_vars, mock_job_context):
    """Test handling of LiveKit connection error."""
    mock_job_context.connect.side_effect = Exception("Connection failed")

    with pytest.raises(Exception) as exc_info:
        await entrypoint(mock_job_context)
    assert "Connection failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_entrypoint_agent_startup_error(mock_env_vars, mock_job_context):
    """Test handling of agent startup error."""
    with patch('dianachat_agent.main.DianaAgent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent.start = AsyncMock(side_effect=Exception("Agent startup failed"))
        mock_agent_class.return_value = mock_agent

        with pytest.raises(Exception) as exc_info:
            await entrypoint(mock_job_context)
        assert "Agent startup failed" in str(exc_info.value)

def test_settings_validation(mock_env_vars):
    """Test AgentSettings validation."""
    settings = AgentSettings()
    
    assert settings.livekit_url == mock_env_vars['LIVEKIT_URL']
    assert settings.livekit_api_key == mock_env_vars['LIVEKIT_API_KEY']
    assert settings.livekit_api_secret == mock_env_vars['LIVEKIT_API_SECRET']

def test_settings_url_parsing(mock_env_vars):
    """Test URL parsing in settings."""
    settings = AgentSettings()
    parsed_url = urlparse(settings.livekit_url)
    
    assert parsed_url.scheme == 'wss'
    assert parsed_url.netloc == 'test.livekit.io'
