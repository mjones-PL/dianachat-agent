# DianaChat Agent Customization Guide

This guide outlines various approaches for customizing and extending the DianaChat agent's language model capabilities.

## Table of Contents
- [Overview](#overview)
- [Custom LLM Implementations](#custom-llm-implementations)
- [RAG Integration](#rag-integration)
- [Autogen2 Integration](#autogen2-integration)
- [Hybrid Approaches](#hybrid-approaches)

## Overview

DianaChat's agent architecture is designed to be modular and extensible. The core component that can be customized is the LLM (Language Model) implementation, which handles the processing of user messages and generation of responses.

The agent supports both text-only and voice interactions, with all responses being streamed for optimal user experience.

## Custom LLM Implementations

### Base Interface
Custom LLM implementations must conform to the LiveKit agents framework interface:

```python
class CustomLLM(llm.LLM):
    async def chat(self, chat_ctx: llm.ChatContext) -> AsyncIterable[llm.ChatCompletionChunk]:
        # Implementation here
        pass
```

### Key Requirements
- Must support async streaming
- Must format responses as ChatCompletionChunks
- Must handle chat context appropriately
- Should implement proper error handling
- Should manage resources efficiently

## RAG Integration

### Approach 1: Direct RAG Integration

```python
class RAGEnhancedLLM(llm.LLM):
    def __init__(self, vector_store, base_llm):
        self.vector_store = vector_store
        self.base_llm = base_llm
        
    async def chat(self, chat_ctx):
        # 1. Extract query from chat context
        query = chat_ctx.messages[-1].text
        
        # 2. Query vector store
        relevant_docs = await self.vector_store.query(query)
        
        # 3. Enhance prompt with retrieved context
        enhanced_context = self._enhance_context(chat_ctx, relevant_docs)
        
        # 4. Stream response using base LLM
        async for chunk in self.base_llm.chat(enhanced_context):
            yield chunk
```

### Approach 2: Middleware RAG

```python
class RAGMiddleware:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    async def enhance_message(self, message: str) -> str:
        docs = await self.vector_store.query(message)
        return self._format_with_context(message, docs)
```

## Autogen2 Integration

### Approach 1: Direct Autogen2 Integration

```python
class Autogen2LLM(llm.LLM):
    def __init__(self, config: Dict):
        self.agent = ag2.Agent(config)
        
    async def chat(self, chat_ctx):
        message = chat_ctx.messages[-1].text
        
        async for chunk in self.agent.generate_stream(message):
            yield llm.ChatCompletionChunk(
                choices=[
                    llm.ChatCompletionChoice(
                        delta=llm.ChatCompletionDelta(
                            content=chunk
                        )
                    )
                ]
            )
```

### Approach 2: Multi-Agent Integration

```python
class MultiAgentLLM(llm.LLM):
    def __init__(self, agent_configs: List[Dict]):
        self.agents = [ag2.Agent(config) for config in agent_configs]
        
    async def chat(self, chat_ctx):
        message = chat_ctx.messages[-1].text
        
        # Coordinate multiple agents
        responses = await asyncio.gather(
            *[agent.generate_stream(message) for agent in self.agents]
        )
        
        # Combine and stream responses
        async for chunk in self._combine_responses(responses):
            yield self._format_chunk(chunk)
```

## Hybrid Approaches

### RAG + Autogen2 Integration

```python
class HybridLLM(llm.LLM):
    def __init__(self, rag_config: Dict, autogen_config: Dict):
        self.rag = RAGMiddleware(rag_config)
        self.agent = ag2.Agent(autogen_config)
        
    async def chat(self, chat_ctx):
        # 1. Get original message
        message = chat_ctx.messages[-1].text
        
        # 2. Enhance with RAG
        enhanced_message = await self.rag.enhance_message(message)
        
        # 3. Process with Autogen2
        async for chunk in self.agent.generate_stream(enhanced_message):
            yield self._format_chunk(chunk)
```

## Implementation Guidelines

1. **Streaming Performance**
   - Implement proper chunking for responses
   - Use appropriate buffer sizes
   - Handle backpressure appropriately

2. **Error Handling**
   - Implement graceful fallbacks
   - Log errors comprehensively
   - Maintain user experience during failures

3. **Resource Management**
   - Properly initialize and clean up resources
   - Implement connection pooling where appropriate
   - Handle concurrent requests efficiently

4. **Context Management**
   - Preserve conversation history
   - Handle system prompts correctly
   - Manage token limits

## Integration Steps

1. Create your custom LLM implementation
2. Update agent configuration:
```python
custom_llm = YourCustomLLM(config)

agent = VoicePipelineAgent(
    chat_ctx=chat_ctx,
    vad=vad,
    turn_detector=turn_detector,
    stt=stt,
    llm=custom_llm,  # Your custom implementation
    tts=tts,
    before_tts_cb=replace_words,
)
```

3. Test thoroughly:
   - Text-only mode
   - Voice mode
   - Error scenarios
   - Performance under load

## Best Practices

1. **Modularity**
   - Keep implementations focused and single-purpose
   - Use dependency injection
   - Make components easily replaceable

2. **Testing**
   - Write unit tests for custom implementations
   - Test streaming behavior
   - Test error scenarios
   - Benchmark performance

3. **Monitoring**
   - Log important events
   - Track performance metrics
   - Monitor resource usage

4. **Documentation**
   - Document configuration options
   - Provide usage examples
   - Include performance considerations
