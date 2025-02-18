#!/usr/bin/env python3
"""Simple script to test OpenAI API key."""

import os
import asyncio
from openai import AsyncOpenAI

async def test_key():
    """Test the OpenAI API key directly."""
    # Get the key directly from environment
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API key found: {'Yes' if api_key else 'No'}")
    print(f"API key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        print("No API key found in environment!")
        return
        
    # Test the key
    client = AsyncOpenAI(api_key=api_key)
    
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input="Test"
        )
        print("Success! API key is working.")
        print(f"Embedding dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"Error testing API key: {e}")

if __name__ == "__main__":
    asyncio.run(test_key())
