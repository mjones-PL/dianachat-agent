import asyncio
import logging
from pathlib import Path
from typing import List
import aiofiles
from bs4 import BeautifulSoup

from ..config.agent_config import AgentSettings
from .service import RAGService

logger = logging.getLogger(__name__)

async def read_text_file(file_path: Path) -> str:
    """Read text from a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        str: The text content of the file
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

async def process_html_file(file_path: Path) -> str:
    """Process an HTML file and extract its text content.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        str: The text content of the HTML file
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            html_content = await f.read()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it up
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Error processing HTML file {file_path}: {e}")
        return ""

async def load_documents(docs_dir: Path) -> List[str]:
    """Load documents from a directory recursively.
    
    Args:
        docs_dir: Directory containing documents to load
        
    Returns:
        List[str]: List of document texts
    """
    documents = []
    
    try:
        # Recursively find all .txt and .html files
        files = []
        files.extend(docs_dir.rglob('*.txt'))
        files.extend(docs_dir.rglob('*.html'))
        
        if not files:
            logger.warning(f"No .txt or .html files found in {docs_dir}")
            return documents
            
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for file_path in files:
            logger.info(f"Processing {file_path}")
            
            if file_path.suffix.lower() == '.html':
                text = await process_html_file(file_path)
            else:  # .txt file
                text = await read_text_file(file_path)
                
            if text:
                # Split into manageable chunks if text is too long
                # This is a simple split, you might want to use more sophisticated chunking
                chunks = text.split('\n\n')
                meaningful_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
                documents.extend(meaningful_chunks)
                logger.info(f"Added {len(meaningful_chunks)} chunks from {file_path}")
            
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        
    return documents

async def create_vector_db(documents_dir: str = "docs"):
    """Create vector database from documents in the specified directory.
    
    Args:
        documents_dir: Directory containing documents to index
    """
    settings = AgentSettings()
    
    # Get the app root directory (2 levels up from this file)
    app_root = Path(__file__).parent.parent.parent.parent
    
    # Setup paths
    data_dir = app_root / 'data'
    docs_dir = app_root / 'docs'  # Changed to be at root level
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load documents from the specified directory
    documents = await load_documents(docs_dir)
    
    if not documents:
        logger.error("No documents found to process")
        return
        
    logger.info(f"Loaded {len(documents)} document chunks")
    
    # Initialize RAG service
    logger.info("Initializing RAG service...")
    rag_service = RAGService(settings)
    
    # Build the index
    logger.info("Starting to build index...")
    await rag_service.build_index(documents)
    logger.info("Vector database created successfully")

if __name__ == "__main__":
    asyncio.run(create_vector_db())
