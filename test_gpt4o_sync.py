#!/usr/bin/env python3
"""
Quick test to verify Gpt4o API sync is working correctly
Tests the updated chat_service.py with new API configuration
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.chat_service import ChatService


async def test_chat_service():
    """Test that ChatService works with new Gpt4o configuration"""
    print("=" * 60)
    print("Gpt4o API Sync Verification Test")
    print("=" * 60)

    # Check environment variables
    api_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "Gpt4o")

    print(f"\n📋 Configuration:")
    print(f"  OPENAI_BASE_URL: {api_url}")
    print(f"  OPENAI_API_KEY: {api_key[:20]}..." if api_key else "  OPENAI_API_KEY: Not set")
    print(f"  OPENAI_MODEL: {model}")
    print()

    if not api_key:
        print("❌ OPENAI_API_KEY not set in .env")
        return False

    try:
        # Initialize ChatService
        print("🔧 Initializing ChatService...")
        service = ChatService()
        print(f"✅ ChatService initialized with model: {service.model_name}")
        print()

        # Test non-streaming chat
        print("🧪 Testing non-streaming chat completion...")
        response = await service.chat_completion(
            user_message="What is 2+2? Answer in one short sentence.",
            max_history=1
        )

        print(f"✅ Response received:")
        print(f"  Message: {response.message}")
        print(f"  Prompt tokens: {response.prompt_tokens}")
        print(f"  Completion tokens: {response.completion_tokens}")
        print(f"  Total tokens: {response.total_tokens}")
        print(f"  Cost: ${response.cost_usd:.6f}")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        print()

        # Test streaming chat
        print("🧪 Testing streaming chat completion...")
        print("  Assistant: ", end="", flush=True)

        chunks = []
        async for chunk in service.chat_completion_stream(
            user_message="Tell me one interesting fact about New Zealand in one sentence.",
            max_history=2
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        print()
        print(f"✅ Streaming completed - received {len(chunks)} chunks")
        print()

        print("=" * 60)
        print("✅ All tests passed! Gpt4o API sync is working correctly.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry point"""
    success = await test_chat_service()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
