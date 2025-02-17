"""Script to examine and test RAG embeddings."""

import asyncio
import logging
import argparse
from typing import Optional
from dotenv import load_dotenv

from dianachat_agent.config.agent_config import AgentSettings
from dianachat_agent.rag.service import RAGService

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def list_all_embeddings(rag_service: RAGService):
    """List all embeddings in the database."""
    try:
        await rag_service.initialize()
        
        if not rag_service.index:
            logger.error("No index loaded")
            return
            
        if not rag_service.paragraphs_by_id:
            logger.error("No paragraphs loaded")
            return
            
        num_items = rag_service.index.get_n_items()
        logger.info(f"Found {num_items} embeddings in the database")
        logger.info(f"Index dimension: {rag_service.index.f}")
        logger.info(f"Number of paragraphs: {len(rag_service.paragraphs_by_id)}")
        
        # Print first few paragraphs as sample
        for i in range(min(3, len(rag_service.paragraphs_by_id))):
            logger.info(f"\nParagraph {i}:")
            logger.info(f"{rag_service.paragraphs_by_id[i][:200]}...")
            
    except Exception as e:
        logger.error(f"Error examining embeddings: {e}")

async def examine_embeddings(query: str, rag_service: RAGService):
    """Examine embeddings for a given query."""
    try:
        await rag_service.initialize()
        
        if not rag_service.index:
            logger.error("No index loaded")
            return
            
        if not rag_service.paragraphs_by_id:
            logger.error("No paragraphs loaded")
            return
            
        results = await rag_service.get_similar_texts(query)
        
        logger.info(f"\nQuery: {query}\n")
        logger.info("Top 5 similar texts:")
        logger.info("-" * 80)
        
        if not results:
            logger.info("\nNo similar texts found. This might indicate an issue with the index or data storage.")
            return
            
        for i, (text, similarity) in enumerate(results, 1):
            logger.info(f"\n{i}. Similarity: {similarity:.4f}")
            logger.info(f"Text: {text[:200]}...")
            logger.info("-" * 80)
            
    except Exception as e:
        logger.error(f"Error examining embeddings: {e}")
        raise

async def interactive_examination(rag_service: RAGService):
    """Interactive examination of embeddings."""
    print("\nRAG Embeddings Examination Tool")
    print("=" * 40)
    print("\nCommands:")
    print("  query <text>  - Search for similar texts")
    print("  list         - List all embeddings")
    print("  exit         - Exit the tool")
    print()
    
    while True:
        command = input("\nEnter command: ").strip()
        if command.lower() == 'exit':
            break
            
        if not command:
            continue
            
        try:
            if command.lower() == 'list':
                await list_all_embeddings(rag_service)
            elif command.lower().startswith('query '):
                query = command[6:].strip()
                if query:
                    await examine_embeddings(query, rag_service)
                else:
                    print("Please provide a query text")
            else:
                print("Unknown command. Available commands: query <text>, list, exit")
        except Exception as e:
            logger.error(f"Error processing command: {e}")

async def main(list_all: bool = False, query: Optional[str] = None):
    """Main function."""
    # Initialize RAG service
    settings = AgentSettings()
    rag_service = RAGService(settings)
    
    if list_all:
        await list_all_embeddings(rag_service)
    elif query:
        await examine_embeddings(query, rag_service)
    else:
        await interactive_examination(rag_service)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine RAG embeddings")
    parser.add_argument('--list', action='store_true', help='List all embeddings in the database')
    parser.add_argument('--query', type=str, help='Query to search for similar texts')
    
    args = parser.parse_args()
    asyncio.run(main(list_all=args.list, query=args.query))
