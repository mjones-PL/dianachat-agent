"""RAG service for DianaChat agent."""

import pickle
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
import logging
import time
from openai import AsyncOpenAI
import annoy
from dotenv import load_dotenv

from dianachat_agent.config.agent_config import AgentSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RAGService:
    """Service for managing RAG operations."""
    
    def __init__(self, settings: AgentSettings):
        """Initialize RAG service.
        
        Args:
            settings: Application settings containing API keys and configurations
        """
        self.settings = settings
        self.index: Optional[annoy.AnnoyIndex] = None
        self.paragraphs_by_id: Dict[int, str] = {}
        
        # Validate OpenAI API key
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required but not found in settings")
            
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._embedding_cache: Dict[str, List[float]] = {}  # Cache for query embeddings
        self._embedding_dim = settings.rag_embeddings_dimension
        
        # Get the app root directory (3 levels up from this file)
        app_root = Path(__file__).parent.parent.parent.parent
        
        # Setup paths
        self.data_dir = app_root / 'data'
        self.docs_dir = app_root / 'docs'  # Changed to be at root level
        
        # Setup specific file paths
        self.index_path = self.data_dir / 'vdb_data'
        self.paragraphs_path = self.data_dir / 'paragraphs.pkl'
        
        logger.info(f"RAG paths configured:")
        logger.info(f"  Index file: {self.index_path}")
        logger.info(f"  Paragraphs file: {self.paragraphs_path}")
        logger.info(f"  Documents directory: {self.docs_dir}")
        
    async def initialize(self):
        """Initialize RAG service."""
        try:
            # Create all necessary directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize the index
            self.index = annoy.AnnoyIndex(self._embedding_dim, 'dot')
            
            if not self.index_path.exists():
                logger.warning(f"No existing index found at {self.index_path}")
                return
                
            try:
                # Load the index
                self.index.load(str(self.index_path))
                logger.info(f"Loaded existing index from {self.index_path}")
                
                # Load paragraphs if they exist
                if self.paragraphs_path.exists():
                    with open(self.paragraphs_path, 'rb') as f:
                        self.paragraphs_by_id = pickle.load(f)
                    logger.info(f"Loaded {len(self.paragraphs_by_id)} paragraphs from {self.paragraphs_path}")
                else:
                    logger.warning(f"No paragraphs file found at {self.paragraphs_path}")
                    
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                # If there's an error loading, initialize a new index
                self.index = annoy.AnnoyIndex(self._embedding_dim, 'dot')
                logger.info("Initialized new empty index")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
            
    async def build_index(self, texts: List[str]) -> None:
        """Build the Annoy index from a list of texts.
        
        Args:
            texts: List of text paragraphs to index
        """
        try:
            # Generate embeddings for all texts
            logger.info(f"Generating embeddings for {len(texts)} texts")
            all_embeddings = []
            for i, text in enumerate(texts):
                response = await self.client.embeddings.create(
                    model=self.settings.rag_model,
                    input=text
                )
                embedding = response.data[0].embedding
                # Normalize the embedding to unit length
                norm = sum(x * x for x in embedding) ** 0.5
                normalized_embedding = [x / norm for x in embedding]
                all_embeddings.append(normalized_embedding)
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
                    
            # Create and build the index
            embedding_dim = len(all_embeddings[0])
            logger.info(f"Creating index with dimension {embedding_dim}")
            self.index = annoy.AnnoyIndex(embedding_dim, 'dot')
            
            # Add items to index
            for i, embedding in enumerate(all_embeddings):
                self.index.add_item(i, embedding)
                
            # Build index
            logger.info("Building index...")
            self.index.build(n_trees=10)  # More trees = better accuracy but larger index
            
            # Save index and data
            logger.info(f"Saving index to {self.index_path}")
            self.index.save(str(self.index_path))
            
            # Save text data
            self.paragraphs_by_id = {i: text for i, text in enumerate(texts)}
            with open(self.paragraphs_path, "wb") as f:
                pickle.dump(self.paragraphs_by_id, f)
                
            logger.info("Successfully built and saved index")
            
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)
            raise
            
    async def get_similar_texts(
        self, query: str, n: int = 5, min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Get texts similar to query from the RAG index.
        
        Args:
            query: Query text to find similar texts for
            n: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of (text, similarity) tuples sorted by similarity
        """
        if not self.index or not self.paragraphs_by_id:
            return []

        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        if query_embedding is None:
            return []
            
        # Normalize query embedding
        norm = sum(x * x for x in query_embedding) ** 0.5
        query_embedding = [x / norm for x in query_embedding]

        # Get nearest neighbors - search through 8x requested neighbors
        search_k = n * 8
        nearest_ids, distances = self.index.get_nns_by_vector(
            query_embedding, search_k, include_distances=True
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Found {len(nearest_ids)} nearest IDs")
            logger.debug("Retrieved embeddings details:")
            for i, (id, distance) in enumerate(zip(nearest_ids, distances), 1):
                # Distance is already cosine similarity since we use dot product
                similarity = (distance + 1) / 2  # Convert from [-1,1] to [0,1]
                text = self.paragraphs_by_id.get(id, "")
                logger.debug(f"Embedding {i}:\n- ID: {id}\n- Distance: {distance:.4f}\n- Similarity: {similarity:.4f}\n- Text: {text}")

        # Convert distances to similarities
        similarities = [(d + 1) / 2 for d in distances]  # Convert from [-1,1] to [0,1]

        # Get corresponding texts and filter by similarity
        results = []
        for idx, similarity in zip(nearest_ids, similarities):
            if similarity < min_similarity:
                continue

            text = self.paragraphs_by_id.get(idx)
            if text:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Found text for ID {idx}:\n"
                        f"- Similarity: {similarity:.4f}\n"
                        f"- Text: {text}"
                    )
                results.append((text, similarity))
            else:
                logger.warning(f"No text found for index {idx}")

        # Sort by similarity and take top n
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        try:
            # Check cache first
            if text in self._embedding_cache:
                logger.debug("Using cached embedding")
                return self._embedding_cache[text]
            
            start_time = time.time()
            logger.debug(f"Generating embedding for query: {text[:100]}...")
            
            response = await self.client.embeddings.create(
                model=self.settings.rag_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self._embedding_cache[text] = embedding
            
            if logger.isEnabledFor(logging.DEBUG):
                duration = time.time() - start_time
                logger.debug(
                    "Generated embedding:\n"
                    f"- Model: {self.settings.rag_model}\n"
                    f"- Dimensions: {len(embedding)}\n"
                    f"- First 5 elements: {embedding[:5]}\n"
                    f"- Generation time: {duration:.4f}s"
                )
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def enrich_context(self, message: str) -> Optional[str]:
        """Enrich user message with relevant context.
        
        Args:
            message: User message to enrich with context
            
        Returns:
            Enriched context string, or None if no relevant context found
        """
        try:
            # Get similar texts with higher base threshold since we have better scoring now
            results = await self.get_similar_texts(message, n=5, min_similarity=0.35)
            if not results:
                return None
                
            # Use dynamic threshold based on best match
            best_similarity = results[0][1]
            # More aggressive dynamic threshold since scores are better spread
            dynamic_threshold = max(0.35, best_similarity * 0.8)  # At least 80% as similar as best match
            
            # Filter and format results
            context_parts = []
            context_parts.append("Here is relevant information from our knowledge base:")
            
            # Add numbered list of relevant sections with similarity scores
            filtered_results = [(text, sim) for text, sim in results if sim >= dynamic_threshold]
            if not filtered_results:
                return None
                
            for i, (text, similarity) in enumerate(filtered_results, 1):
                # Format score as percentage and text
                context_parts.append(f"\n{i}. ({similarity:.0%} match) {text.strip()}")
            
            # Combine all parts
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error enriching context: {e}")
            return None
