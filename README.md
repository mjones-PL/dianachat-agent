# DianaChat Agent

LiveKit-powered chat agent for the DianaChat platform.

## Overview

DianaChat Agent is a Python-based AI agent that enables real-time voice and text interactions through the DianaChat platform. It processes natural language input and generates intelligent responses using state-of-the-art AI models.

## Technical Stack

- **Python**: 3.11+
- **LiveKit**: Real-time communication
- **OpenAI GPT-4**: Language model
- **Deepgram**: Speech-to-text
- **OpenAI TTS**: Text-to-speech
- **Redis**: Response caching
- **pytest**: Testing framework

## System Architecture

### Components

1. **DianaChat Frontend** (separate repository)
   - Web and mobile clients
   - LiveKit integration
   - Real-time communication

2. **DianaChat Agent** (this repository)
   - LiveKit Multimodal Agent
   - Async I/O operations
   - Error handling and recovery
   - Response caching

### AI Components

1. **Speech-to-Text (Deepgram)**
   - Nova-2 model
   - Real-time transcription
   - Language detection
   - Error recovery

2. **Language Model (OpenAI)**
   - GPT-4 Turbo
   - Context management
   - Token optimization
   - Safety filters

3. **Text-to-Speech (OpenAI)**
   - Shimmer voice
   - Stream processing
   - Error handling

## Project Structure

```
dianachat-agent/
├── src/dianachat_agent/
│   ├── agents/          # Agent implementations
│   ├── api/            # API endpoints
│   ├── config/         # Configuration
│   ├── models/         # Data models
│   ├── services/       # Core services
│   └── utils/          # Utilities
├── tests/              # Test suites
├── alembic/            # Database migrations
├── .env.example        # Environment template
├── pyproject.toml      # Project metadata
└── README.md          # Documentation
```

## Installation

1. Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Required Environment Variables

```bash
# LiveKit
LIVEKIT_URL=wss://your-livekit-server
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_VOICE=shimmer
OPENAI_TEMPERATURE=0.7

# Deepgram
DEEPGRAM_API_KEY=your_deepgram_key
DEEPGRAM_MODEL=nova-2
DEEPGRAM_LANGUAGE=en-US
DEEPGRAM_TIER=enhanced

# Agent Settings
ENABLE_VIDEO=false
MAX_RESPONSE_TOKENS=400
ENABLE_RESPONSE_CACHING=true
CACHE_TTL_SECONDS=3600
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=dianachat_agent
```

### Error Handling

The agent implements multiple layers of error handling:

1. **Service Errors**
   - STT transcription failures
   - LLM timeouts or errors
   - TTS synthesis issues
   - Network connectivity problems

2. **Recovery Strategies**
   - Automatic retries for transient failures
   - Graceful degradation (text-only mode)
   - User-friendly error messages
   - Detailed error logging

3. **Monitoring**
   - Error rate tracking
   - Response latency monitoring
   - Token usage metrics
   - Cost optimization alerts

### Contributing

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feat/your-feature
   ```
3. Make changes following our standards:
   - Type annotations
   - PEP 8 style
   - Comprehensive tests
   - Clear documentation
4. Submit a pull request

### Deployment

The agent supports multiple deployment strategies:

1. **Production**
   - Main branch
   - Zero-downtime updates
   - Automatic rollbacks
   - Full monitoring

2. **Staging**
   - Staging branch
   - Integration testing
   - Performance testing
   - Cost monitoring

## RAG (Retrieval-Augmented Generation)

DianaChat Agent uses RAG to enhance responses with relevant context from your knowledge base.

### Architecture

The RAG system consists of three main components:

1. **Document Processing**
   - Splits documents into paragraphs
   - Generates embeddings using OpenAI's text-embedding-3-small model
   - Stores vectors in an Annoy index for efficient similarity search
   - Saves paragraph text in a pickle file for retrieval

2. **RAG Service**
   - Manages vector database operations
   - Performs real-time embedding generation
   - Handles similarity search and context retrieval
   - Integrates with both voice and text pipelines

3. **Agent Integration**
   - Enriches user messages with relevant context
   - Uses before_llm_cb hook for voice pipeline
   - Directly enriches messages in text processing
   - Gracefully handles RAG failures

### Setup

1. **Environment Variables**
```bash
# OpenAI API (required for embeddings)
OPENAI_API_KEY=your_api_key

# RAG Configuration (optional, shown with defaults)
RAG_MODEL=text-embedding-3-small
RAG_EMBEDDINGS_DIMENSION=1536
RAG_INDEX_PATH=src/dianachat_agent/rag/data/vdb_data
RAG_DATA_PATH=src/dianachat_agent/rag/data/paragraphs.pkl
```

2. **Document Preparation**
```bash
# Create docs directory if it doesn't exist
mkdir -p docs

# Add your documents to docs/
# Supported formats: .txt, .md, .rst, .py
```

3. **Build Vector Database**
```bash
# From project root
python -m dianachat_agent.rag.create_vector
```

### Usage

1. **Examine Embeddings**
```bash
# Interactive mode
python -m dianachat_agent.rag.examine_embeddings

# List all embeddings
python -m dianachat_agent.rag.examine_embeddings --list

# Direct query
python -m dianachat_agent.rag.examine_embeddings --query "your query"
```

2. **RAG Pipeline**
The RAG system automatically:
- Processes each user message
- Finds relevant context from your documents
- Enriches the prompt with this context
- Works for both voice and text interactions

### Implementation Details

1. **Vector Storage**
- Uses Annoy (Approximate Nearest Neighbors) for efficient similarity search
- Angular distance metric for comparing embeddings
- In-memory index for fast retrieval
- Persistent storage for both vectors and text

2. **Context Enrichment**
```python
# Example enriched message format
Here is some relevant context:
[Retrieved relevant text from your documents]

User message: [Original user message]
Please use this context to inform your response.
```

3. **Error Handling**
- Graceful degradation if RAG service fails
- Fallback to raw message if no relevant context
- Automatic recovery on service initialization
- Detailed logging for troubleshooting

### Maintenance

1. **Updating Knowledge Base**
- Add new documents to `docs/`
- Run `create_vector.py` to rebuild index
- No need to restart the agent

2. **Monitoring**
- Check logs for RAG-related messages
- Monitor embedding API usage
- Review context relevance using examine_embeddings
- Adjust similarity thresholds if needed

### Performance Considerations

1. **Memory Usage**
- Annoy index stays in memory
- Paragraph text stored on disk
- Lazy loading of components

2. **API Costs**
- One embedding per user message
- Uses efficient embedding model
- Caches embeddings where possible

3. **Response Time**
- Fast vector similarity search (~ms)
- Async embedding generation
- Minimal impact on response time

## Security

- API keys in environment variables
- Input sanitization
- Rate limiting
- Access control
- Audit logging
- Regular security updates

## License

[Insert License Information]
