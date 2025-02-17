"""Tests for the DianaChat Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from livekit.agents import AgentConfig
from src.agents.diana_agent import DianaAgent, ConversationState, TurnDetectionError
from src.config.agent_config import AgentSettings

@pytest.fixture
def mock_config():
    """Create a mock LiveKit agent configuration."""
    return AgentConfig(
        url="wss://test.livekit.io",
        api_key="test_key",
        api_secret="test_secret",
    )

@pytest.fixture
def mock_settings():
    """Create mock agent settings."""
    return AgentSettings(
        livekit_url="wss://test.livekit.io",
        livekit_api_key="test_key",
        livekit_secret_key="test_secret",
        openai_api_key="test_openai_key",
        deepgram_api_key="test_deepgram_key",
    )

@pytest.fixture
def mock_openai_key():
    """Provide a mock OpenAI API key."""
    return "test_openai_key"

@pytest.fixture
async def agent(mock_config, mock_settings):
    """Create a DianaAgent instance with mocked dependencies."""
    with patch("livekit.plugins.silero.VAD") as mock_vad, \
         patch("livekit.plugins.turn_detector.EOUModel") as mock_eou:
        
        # Setup mocks
        mock_vad.load.return_value = AsyncMock()
        mock_eou.return_value = AsyncMock()
        
        agent = DianaAgent(mock_config, mock_settings)
        
        # Mock the AI services
        agent.stt = AsyncMock()
        agent.tts = AsyncMock()
        agent.llm = AsyncMock()
        
        yield agent

@pytest.mark.asyncio
async def test_participant_joined(agent):
    """Test participant join handling."""
    participant = MagicMock()
    participant.identity = "test_user"
    
    # Mock publish_data
    agent.publish_data = AsyncMock()
    
    await agent.on_participant_joined(participant)
    
    # Verify welcome message was sent
    agent.tts.synthesize_speech.assert_called_once()
    agent.publish_data.assert_called_once()
    assert "welcome" in agent.publish_data.call_args[0][0]["type"]

@pytest.mark.asyncio
async def test_audio_processing(agent):
    """Test audio processing with turn detection."""
    # Mock VAD to detect voice
    agent.vad.predict.return_value = True
    
    # Send some audio data
    test_audio = b"test_audio_data"
    await agent.on_audio_received(test_audio)
    
    # Verify audio was buffered
    assert agent.state.is_speaking
    assert test_audio in agent.state.audio_buffer
    
    # Mock VAD to detect silence and EOU to detect end of turn
    agent.vad.predict.return_value = False
    agent.turn_detector.predict.return_value = True
    
    # Send silence
    await agent.on_audio_received(b"silence")
    
    # Verify turn was processed
    agent.stt.transcribe.assert_called_once()

@pytest.mark.asyncio
async def test_data_message_processing(agent):
    """Test processing of data messages."""
    participant = MagicMock()
    test_message = {"message": "Hello", "speak": True}
    
    # Mock response generation
    agent.llm.generate.return_value = "Hi there!"
    agent.publish_data = AsyncMock()
    
    await agent.on_data_received(test_message, participant)
    
    # Verify message was processed
    assert len(agent.state.history) == 2  # User message + AI response
    agent.publish_data.assert_called_once()
    agent.tts.synthesize_speech.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(agent):
    """Test error handling in audio processing."""
    # Mock an error in speech-to-text
    agent.stt.transcribe.side_effect = Exception("STT Error")
    agent.publish_data = AsyncMock()
    
    # Process some audio
    await agent.process_utterance(b"test_audio")
    
    # Verify error was handled
    error_message = agent.publish_data.call_args[0][0]
    assert error_message["type"] == "error"
    assert "error" in error_message["message"].lower()

@pytest.mark.asyncio
async def test_conversation_state_management(agent):
    """Test conversation state management."""
    # Initial state
    assert isinstance(agent.state, ConversationState)
    assert len(agent.state.history) == 0
    
    # Simulate a conversation turn
    test_text = "Hello"
    test_response = "Hi there!"
    
    agent.llm.generate.return_value = test_response
    agent.publish_data = AsyncMock()
    
    await agent._send_responses(test_text, test_response)
    
    # Verify state updates
    assert agent.publish_data.called
    agent.tts.synthesize_speech.assert_called_with(test_response)
