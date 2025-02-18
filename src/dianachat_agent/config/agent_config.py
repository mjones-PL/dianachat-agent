"""Configuration settings for the DianaChat Agent."""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from ..log import logger

def load_env_file(env_file: str = ".env") -> Dict[str, Any]:
    """Load environment variables from file."""
    env_vars = {}
    if os.path.exists(env_file):
        logger.debug(f"Loading environment variables from {os.path.abspath(env_file)}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")
                    except ValueError:
                        continue  # Skip malformed lines
    else:
        logger.warning(f"Environment file not found at {os.path.abspath(env_file)}")
    return env_vars

class AgentSettings(BaseModel):
    """Settings for the DianaChat Agent loaded from environment variables."""
    
    # LiveKit configuration
    livekit_url: str = Field(default="")
    livekit_api_key: str = Field(default="")
    livekit_api_secret: str = Field(default="")
    
    # OpenAI configuration
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4-turbo-preview")
    openai_voice: str = Field(default="shimmer")
    openai_temperature: float = Field(default=0.7)
    
    # Deepgram configuration
    deepgram_api_key: str = Field(default="")
    deepgram_model: str = Field(default="nova-2")
    deepgram_language: str = Field(default="en-US")
    deepgram_tier: str = Field(default="enhanced")
    
    # Agent configuration
    enable_response_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)
    default_greeting: str = Field(default="Hello! I'm Diana, Bridge House Health's Admissions and Information Assistant. How can I help you?")
    
    # RAG configuration
    rag_embeddings_dimension: int = Field(default=1536)
    rag_model: str = Field(default="text-embedding-3-small")
    system_prompt: str = Field(default="")

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "AgentSettings":
        """Create settings from environment file."""
        # Load from env file first
        env_vars = load_env_file(env_file)
        logger.debug(f"Loaded variables from env file: {list(env_vars.keys())}")
        
        # System env vars take precedence
        system_vars = dict(os.environ)
        env_vars.update(system_vars)
        logger.debug(f"Environment variables after system update: {list(env_vars.keys())}")
        
        def parse_bool(value: str) -> bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        
        # Convert string to bool for enable_response_caching
        if 'ENABLE_RESPONSE_CACHING' in env_vars:
            env_vars['ENABLE_RESPONSE_CACHING'] = parse_bool(env_vars['ENABLE_RESPONSE_CACHING'])
        
        # Get OpenAI key directly from environment first, then fallback to env_vars
        openai_key = os.getenv('OPENAI_API_KEY') or env_vars.get('OPENAI_API_KEY', '')
        if isinstance(openai_key, str):
            openai_key = openai_key.strip().strip('"').strip("'")
        logger.debug(f"OpenAI API key loaded, length: {len(openai_key)}")
        
        settings = cls(
            livekit_url=env_vars.get('LIVEKIT_URL', ''),
            livekit_api_key=env_vars.get('LIVEKIT_API_KEY', ''),
            livekit_api_secret=env_vars.get('LIVEKIT_API_SECRET', ''),
            openai_api_key=openai_key,
            openai_model=env_vars.get('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            openai_voice=env_vars.get('OPENAI_VOICE', 'shimmer'),
            openai_temperature=float(env_vars.get('OPENAI_TEMPERATURE', '0.7')),
            deepgram_api_key=env_vars.get('DEEPGRAM_API_KEY', ''),
            deepgram_model=env_vars.get('DEEPGRAM_MODEL', 'nova-2'),
            deepgram_language=env_vars.get('DEEPGRAM_LANGUAGE', 'en-US'),
            deepgram_tier=env_vars.get('DEEPGRAM_TIER', 'enhanced'),
            enable_response_caching=env_vars.get('ENABLE_RESPONSE_CACHING', True),
            cache_ttl_seconds=int(env_vars.get('CACHE_TTL_SECONDS', '3600')),
            default_greeting=env_vars.get('DEFAULT_GREETING', "Hello! I am Diana, The Bridge House Health Admissions and Information Assistant. How can I help you?"),
            rag_embeddings_dimension=int(env_vars.get('RAG_EMBEDDINGS_DIMENSION', '1536')),
            rag_model=env_vars.get('RAG_MODEL', 'text-embedding-3-small'),
            system_prompt=env_vars.get('SYSTEM_PROMPT', '')
        )
        
        logger.debug(f"OpenAI API key in settings, length: {len(settings.openai_api_key)}")
        return settings

# Create a global settings instance
settings = AgentSettings.from_env()
