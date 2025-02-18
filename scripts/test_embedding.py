#!/usr/bin/env python3
"""Script to test OpenAI embeddings."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from src.dianachat_agent.config.agent_config import AgentSettings
from openai import AsyncOpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_embedding():
    """Test OpenAI embeddings with a simple text."""
    settings = AgentSettings()
    
    # Log API key length for debugging (never log the actual key)
    logger.info(f"OpenAI API key length: {len(settings.openai_api_key)}")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Test text
    test_text = "This is a test of the OpenAI embeddings API."
    
    try:
        logger.info("Attempting to create embedding...")
        response = await client.embeddings.create(
            model=settings.rag_model or "text-embedding-3-small",
            input=test_text
        )
        
        # Get the embedding vector
        embedding = response.data[0].embedding
        
        logger.info(f"Success! Embedding vector length: {len(embedding)}")
        logger.info(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_embedding())
