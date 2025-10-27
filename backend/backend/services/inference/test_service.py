#!/usr/bin/env python3
"""
Inference service smoke-test script.

Validates the health, embedding, rerank, and model endpoints.
"""
import asyncio
import httpx
import sys


async def test_health():
    """Exercise the /health endpoint."""
    print("1️⃣  Testing /health endpoint...")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8001/health", timeout=5.0)

            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✅ Health check passed")
                print(f"   Status: {data['status']}")
                print(f"   Models loaded: {data['models_loaded']}")
                return True
            else:
                print(f"   ❌ Health check failed: HTTP {resp.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   Make sure the service is running: python -m uvicorn inference_service.main:app --port 8001")
        return False


async def test_embedding():
    """Exercise the /embed endpoint."""
    print("\n2️⃣  Testing /embed endpoint...")

    texts = [
        "Hello world",
        "Artificial intelligence",
        "This is a test"
    ]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8001/embed",
                json={"texts": texts, "normalize": True},
                timeout=30.0
            )

            if resp.status_code == 200:
                data = resp.json()
                embeddings = data["embeddings"]

                print(f"   ✅ Embedding successful")
                print(f"   Texts: {len(texts)}")
                print(f"   Dimension: {data['dimension']}")
                print(f"   Processing time: {data['processing_time_ms']:.2f} ms")
                print(f"   First embedding preview: {embeddings[0][:5]}...")

                # Basic shape validation
                assert len(embeddings) == len(texts), "Number of embeddings mismatch"
                assert all(len(emb) == 1024 for emb in embeddings), "Dimension should be 1024"

                return True
            else:
                print(f"   ❌ Embedding failed: HTTP {resp.status_code}")
                print(f"   Response: {resp.text}")
                return False
    except Exception as e:
        print(f"   ❌ Embedding error: {e}")
        return False


async def test_rerank():
    """Exercise the /rerank endpoint."""
    print("\n3️⃣  Testing /rerank endpoint...")

    query = "What is artificial intelligence?"
    documents = [
        "Artificial intelligence (AI) is a branch of computer science focused on building systems that mimic human intelligence.",
        "The weather is great today, perfect for a walk outside.",
        "Machine learning is a subfield of AI focused on learning from data.",
        "I enjoy eating pizza and burgers."
    ]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8001/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": 3,
                    "return_documents": True
                },
                timeout=30.0
            )

            if resp.status_code == 200:
                data = resp.json()
                scores = data["scores"]
                indices = data["indices"]
                reranked_docs = data.get("documents", [])

                print(f"   ✅ Rerank successful")
                print(f"   Query: {query}")
                print(f"   Top 3 results:")
                for i, (idx, score) in enumerate(zip(indices[:3], scores[:3])):
                    print(f"   {i+1}. [Score: {score:.4f}] {documents[idx][:50]}...")

                # Ensure scores are in descending order
                assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

                return True
            else:
                print(f"   ❌ Rerank failed: HTTP {resp.status_code}")
                print(f"   Response: {resp.text}")
                return False
    except Exception as e:
        print(f"   ❌ Rerank error: {e}")
        return False


async def test_models():
    """Exercise the /models endpoint."""
    print("\n4️⃣  Testing /models endpoint...")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8001/models", timeout=5.0)

            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✅ Models endpoint working")
                print(f"   Embedding models: {list(data['embedding_models'].keys())}")
                print(f"   Rerank models: {list(data['rerank_models'].keys())}")
                return True
            else:
                print(f"   ❌ Failed: HTTP {resp.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def main():
    """Run all integration checks."""
    print("=" * 60)
    print("Inference Service Test Suite")
    print("=" * 60)

    results = []

    # Health check
    results.append(("Health Check", await test_health()))

    # Abort early if the service is down
    if not results[0][1]:
        print("\n" + "=" * 60)
        print("❌ Service not available. Tests aborted.")
        print("=" * 60)
        return 1

    # Execute remaining tests
    results.append(("Embedding", await test_embedding()))
    results.append(("Rerank", await test_rerank()))
    results.append(("Models", await test_models()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("-" * 60)
    print(f"Total: {passed}/{total} passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
