"""Main entry point for the DianaChat Agent."""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents import llm, stt, tokenize, transcription, vad
from livekit.plugins import openai, deepgram, silero, turn_detector

from dianachat_agent.agents.diana_agent import DianaAgent
from dianachat_agent.config.agent_config import AgentSettings
from dianachat_agent.log import logger

# Load environment variables
project_root = Path(__file__).parent.parent.parent.resolve()
env_path = project_root / ".env"
logger.info(f"Loading environment from: {env_path}")
load_dotenv(env_path)

def prewarm(proc):
    """Prewarm function to load models before starting the agent."""
    logger.info("Prewarming models...")
    
    # Load Silero VAD model
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.55,
        prefix_padding_duration=0.5,
        activation_threshold=0.5,
    )
    logger.info("Loaded Silero VAD model")

async def entrypoint(ctx: JobContext):
    """Main entry point for the DianaChat Agent."""
    try:
        # Find .env file
        project_root = Path(__file__).parent.parent.parent.resolve()
        env_path = project_root / ".env"
        logger.info(f"Looking for .env file at: {env_path}")
        if not env_path.exists():
            logger.error(f".env file not found at {env_path}")
            raise FileNotFoundError(f".env file not found at {env_path}")
            
        # Load configuration from environment variables
        settings = AgentSettings.from_env(str(env_path))
        logger.info("Loaded agent settings")
        
        # Debug: Check settings
        logger.info(f"LIVEKIT_URL: {settings.livekit_url}")
        logger.info(f"OpenAI API key length: {len(settings.openai_api_key)}")
        
        # Connect to LiveKit room
        await ctx.connect(
            auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL
        )
        logger.info("Connected to LiveKit room")
        
        # Wait for participant
        participant = await ctx.wait_for_participant()
        logger.info(f"Participant {participant.identity} joined")
        
        # Create and start the agent using prewarmed models
        agent = DianaAgent(
            ctx=ctx,
            settings=settings,
            vad=ctx.proc.userdata["vad"],
            turn_detector_model=None,
        )
        
        try:
            await agent.start()
            logger.info("Started DianaChat Agent")
            
            # Wait for the agent to finish
            await agent.wait()
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise
        finally:
            # Ensure we stop the agent
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent: {e}")
        
    except Exception as e:
        logger.error(f"Error running DianaChat Agent: {str(e)}", exc_info=True)
        raise

def main():
    """Main function to start the agent."""
    try:
        # Debug: Check raw environment variables
        logger.info("Raw environment variables:")
        logger.info(f"LIVEKIT_URL from env: {os.getenv('LIVEKIT_URL', 'Not set')}")
        logger.info(f"LIVEKIT_API_KEY from env: {'*' * len(os.getenv('LIVEKIT_API_KEY', ''))}")
        logger.info(f"LIVEKIT_API_SECRET from env: {'*' * len(os.getenv('LIVEKIT_API_SECRET', ''))}")
        
        # Get settings for worker options
        settings = AgentSettings()
        logger.info("\nPydantic settings:")
        logger.info(f"LIVEKIT_URL from settings: {settings.livekit_url}")
        logger.info(f"LIVEKIT_API_KEY from settings: {'*' * len(settings.livekit_api_key)}")
        logger.info(f"LIVEKIT_API_SECRET from settings: {'*' * len(settings.livekit_api_secret)}")
        
        # Parse URL for server configuration
        parsed_url = urlparse(settings.livekit_url)
        server_url = f"https://{parsed_url.netloc}"  # Convert wss:// to https://
        
        # Create worker options
        options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
        
        # Run the app
        logger.info("Starting agent to listen to all rooms")
        cli.run_app(options)
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
