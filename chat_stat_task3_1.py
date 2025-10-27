#!/usr/bin/env python3
"""
Acceptance test for Task 3.1: Conversational Chat
Run: python chat.py
Then type "Hello" to test streaming chat with metrics
"""
import asyncio
import sys
import json
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import time
import tiktoken

# Load environment variables
load_dotenv()


class ChatClient:
    """Simple chat client for testing"""

    def __init__(self):
        """Initialize chat client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = AsyncOpenAI(api_key=api_key)
        # Force use of gpt-3.5-turbo-0125 (the only accessible model)
        # This overrides any system OPENAI_MODEL env var
        self.model = "gpt-3.5-turbo-0125"
        self.conversation_history = []

        # Initialize tokenizer for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base encoding for gpt-3.5-turbo
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Token pricing (for gpt-3.5-turbo)
        self.input_cost_per_1k = 0.0015
        self.output_cost_per_1k = 0.002

    async def chat(self, user_message: str):
        """
        Send a chat message and stream the response.

        Args:
            user_message: User's message
        """
        start_time = time.time()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Keep only last 10 messages for context
        messages_to_send = self.conversation_history[-10:]

        try:
            # Create streaming completion
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
                stream=True,
                temperature=0.7,
                max_tokens=500
            )

            # Stream the response
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Print without newline for streaming effect
                    print(content, end="", flush=True)

            print()  # New line after streaming

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000

            # Count tokens accurately using tiktoken
            prompt_tokens = 0
            for msg in messages_to_send:
                # Count tokens in message content
                prompt_tokens += len(self.encoding.encode(msg["content"]))
                # Add tokens for message formatting (role, etc.)
                prompt_tokens += 4  # Overhead per message

            completion_tokens = len(self.encoding.encode(full_response))

            # Calculate cost
            cost_usd = (
                (prompt_tokens / 1000 * self.input_cost_per_1k) +
                (completion_tokens / 1000 * self.output_cost_per_1k)
            )

            # Print metrics line
            print(f"[stats] prompt={prompt_tokens}  completion={completion_tokens}  "
                  f"cost=${cost_usd:.6f}  latency={int(latency_ms)} ms")

        except Exception as e:
            print(f"\n❌ Error: {e}")

            # If API key permission issue, show helpful message
            if "does not have access to model" in str(e):
                print(f"\n⚠️  Your OpenAI API key doesn't have access to model '{self.model}'")
                print("Please check your OpenAI account or update OPENAI_MODEL in .env file")

    def run_interactive(self):
        """Run interactive chat loop"""
        print("=" * 60)
        print("Task 3.1: Conversational Chat - Acceptance Test")
        print("=" * 60)
        print(f"Model: {self.model}")
        print("Type 'exit' or 'quit' to end the conversation")
        print("=" * 60)
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                # Send chat message and stream response
                print("Assistant: ", end="", flush=True)
                asyncio.run(self.chat(user_input))
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    try:
        client = ChatClient()
        client.run_interactive()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure OPENAI_API_KEY is set in your .env file")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
