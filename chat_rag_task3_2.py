#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive Chat RAG System"""

import requests
import json
import sys


class RAGChatClient:
    def __init__(self, base_url="http://localhost:8888"):
        self.base_url = base_url
        self.rag_endpoint = f"{base_url}/api/rag/ask"
        self.health_endpoint = f"{base_url}/api/rag/health"

    def check_health(self):
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Cannot connect to RAG API: {e}")
            return False

    def ask(self, question, top_k=5):
        try:
            payload = {"question": question, "top_k": top_k}
            response = requests.post(self.rag_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def display_result(self, result):
        if not result:
            return
        
        print("\n" + "=" * 100)
        print("RAG ANSWER")
        print("=" * 100 + "\n")
        
        answer = result.get("answer", "")
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print()
        
        print("-" * 100)
        print("METRICS")
        print("-" * 100)
        print(f"  Retrieval Time: {result.get('retrieval_time_ms', 0):.1f}ms")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Chunks Retrieved: {result.get('num_chunks_retrieved', 0)}")
        print()
        
        citations = result.get("citations", [])
        if citations:
            print("-" * 100)
            print("SOURCES")
            print("-" * 100)
            for i, citation in enumerate(citations, 1):
                print(f"\n[{i}] {citation.get('source', 'Unknown')}")
                print(f"    Score: {citation.get('score', 0):.3f}")
                print(f"    Document ID: {citation.get('metadata', {}).get('document_id', 'N/A')}")
                content = citation.get('content', '')
                snippet = content[:200] + "..." if len(content) > 200 else content
                print(f"    Content: {snippet}")
        
        print("\n" + "=" * 100)

    def interactive_chat(self):
        print("=" * 100)
        print("Interactive RAG Chat System")
        print("=" * 100 + "\n")
        print("Commands:")
        print("  - Type a question to query")
        print("  - 'quit' or 'exit' to exit")
        print("  - 'help' for help")
        print("=" * 100 + "\n")
        
        if not self.check_health():
            print("\nRAG API not running!")
            print("Start server: python -m uvicorn backend.main:app --port 8888\n")
            return
        
        print("RAG API connected!\n")
        query_count = 0
        
        while True:
            try:
                print("-" * 100)
                user_input = input("\nYour question: ").strip()
                print()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("Type questions to search the document collection.")
                    print("Using INT8-quantized BGE-M3 embeddings for semantic search.")
                    continue
                
                print("Querying...")
                result = self.ask(user_input, top_k=5)
                
                if result:
                    query_count += 1
                    self.display_result(result)
                else:
                    print("Query failed")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive RAG Chat")
    parser.add_argument("--url", default="http://localhost:8888", help="API URL")
    parser.add_argument("--query", help="Single query (non-interactive)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    args = parser.parse_args()
    
    client = RAGChatClient(base_url=args.url)
    
    if args.query:
        print(f"Query: {args.query}\n")
        result = client.ask(args.query, top_k=args.top_k)
        if result:
            client.display_result(result)
        else:
            sys.exit(1)
    else:
        client.interactive_chat()


if __name__ == "__main__":
    main()
