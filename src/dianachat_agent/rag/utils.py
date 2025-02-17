"""Utilities for RAG functionality."""

import asyncio
import pickle
from pathlib import Path
from typing import List, Dict
import aiohttp
from openai import AsyncOpenAI
import logging
import annoy

from dianachat_agent.config.agent_config import AgentSettings

logger = logging.getLogger(__name__)

async def build_index(documents: List[str], settings: AgentSettings) -> None:
    """Build RAG index from documents.
    
    Args:
        documents: List of document chunks to index
        settings: Agent settings containing RAG configuration
    """
    # Get the app root directory (3 levels up from this file)
    app_root = Path(__file__).parent.parent.parent.parent
    
    # Setup paths
    data_dir = app_root / 'data'
    docs_dir = app_root / 'docs'  # Changed to be at root level
    
    # Setup specific file paths
    index_path = data_dir / 'vdb_data'
    paragraphs_path = data_dir / 'paragraphs.pkl'
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Create index
    index = annoy.AnnoyIndex(settings.rag_embeddings_dimension, 'angular')
    paragraphs_by_id = {}
    
    logger.info(f"Processing {len(documents)} document chunks")
    
    # Process each document chunk
    for i, text in enumerate(documents):
        try:
            # Get embeddings for the text
            response = await client.embeddings.create(
                model=settings.rag_embeddings_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Add to index
            index.add_item(i, embedding)
            paragraphs_by_id[i] = text
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} chunks")
                
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue
    
    if not paragraphs_by_id:
        logger.error("No chunks were successfully processed")
        return
    
    # Build and save the index
    logger.info("Building index...")
    index.build(settings.rag_index_trees)
    index.save(str(index_path))
    logger.info(f"Index saved to {index_path}")
    
    # Save paragraphs
    with open(paragraphs_path, 'wb') as f:
        pickle.dump(paragraphs_by_id, f)
    logger.info(f"Paragraphs saved to {paragraphs_path}")
    
    logger.info(f"Successfully processed {len(paragraphs_by_id)} chunks")
