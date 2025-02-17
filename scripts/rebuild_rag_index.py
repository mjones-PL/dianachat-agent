#!/usr/bin/env python3
"""Script to rebuild the RAG index."""

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
from src.dianachat_agent.rag.create_vector import create_vector_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function."""
    logger.info("Starting RAG index rebuild...")
    
    # Ensure we're in the project root directory
    os.chdir(parent_dir)
    
    # Get paths
    app_root = Path(__file__).parent.parent
    docs_dir = app_root / 'docs'
    data_dir = app_root / 'data'
    
    # Log paths
    logger.info(f"Using paths:")
    logger.info(f"  Documents directory: {docs_dir}")
    logger.info(f"  Data directory: {data_dir}")
    
    # Log document count
    doc_files = list(docs_dir.rglob('*.txt'))
    doc_files.extend(docs_dir.rglob('*.html'))
    logger.info(f"Found {len(doc_files)} document files to process:")
    for doc in doc_files:
        logger.info(f"  - {doc.relative_to(app_root)}")
    
    # Create vector database
    await create_vector_db()
    logger.info("RAG index rebuild completed!")

if __name__ == "__main__":
    asyncio.run(main())
