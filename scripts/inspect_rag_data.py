#!/usr/bin/env python3
"""Script to inspect RAG data."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from src.dianachat_agent.config.agent_config import AgentSettings
from src.dianachat_agent.rag.examine_embeddings import list_all_embeddings, examine_embeddings
from src.dianachat_agent.rag.service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function."""
    # Ensure we're in the project root directory
    os.chdir(parent_dir)
    
    # Initialize RAG service
    settings = AgentSettings()
    rag_service = RAGService(settings)
    
    # List all embeddings
    print("\nListing all embeddings:")
    print("=" * 80)
    await list_all_embeddings(rag_service)
    
    # Example query
    query = "What is DianaChat?"
    print(f"\nExample query: {query}")
    print("=" * 80)
    await examine_embeddings(query, rag_service)

if __name__ == "__main__":
    asyncio.run(main())
