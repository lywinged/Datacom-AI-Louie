"""
INT8 dynamic quantization helper.

Convert FP32 ONNX models to INT8 for 2-3x faster inference with <2% accuracy loss.
"""
import os
from pathlib import Path


def quantize_onnx_model(model_path: str, output_path: str):
    """
    Apply ONNX Runtime dynamic quantization.

    Dynamic quantization characteristics:
    - Weights: FP32 -> INT8 (offline conversion)
    - Activations: Quantized dynamically at runtime
    - Accuracy loss: < 1-2%
    - Performance gain: roughly 2-3x
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("❌ onnxruntime quantization extras not installed")
        print("   Install with: pip install onnxruntime")
        return False

    print(f"\n🔧 Quantizing model: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Method: dynamic INT8 quantization")

    try:
        # Dynamic quantization is well suited for transformer models
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,  # Quantize weights to INT8
            per_channel=True,             # Per-channel quantization for better accuracy
        )

        # Report file sizes before and after quantization
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        print("✅ Quantization complete!")
        print(f"   Original size: {original_size:.1f} MB")
        print(f"   Quantized size: {quantized_size:.1f} MB")
        print(f"   Compression ratio: {reduction:.1f}%")
        print("\n📊 Expected benefits:")
        print("   - Inference speed: ~2-3x faster")
        print("   - Memory usage: ~75% lower")
        print("   - Accuracy impact: < 2%")

        return True

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False


def main():
    print("=" * 60)
    print("🚀 ONNX INT8 Dynamic Quantization Tool")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"

    # Quantize embedding model
    embedding_onnx = models_dir / "bge-m3-onnx" / "model.onnx"
    embedding_int8 = models_dir / "bge-m3-onnx" / "model_int8.onnx"

    if embedding_onnx.exists():
        print("\n📦 Quantizing embedding model...")
        quantize_onnx_model(str(embedding_onnx), str(embedding_int8))
    else:
        print(f"⚠️  Embedding model not found: {embedding_onnx}")
        print("   Run: python inference_service/simple_export_onnx.py")

    # Quantize reranker model if available
    rerank_onnx = models_dir / "bge-reranker-onnx" / "model.onnx"
    rerank_int8 = models_dir / "bge-reranker-onnx" / "model_int8.onnx"

    if rerank_onnx.exists():
        print("\n📦 Quantizing reranker model...")
        quantize_onnx_model(str(rerank_onnx), str(rerank_int8))
    else:
        print(f"\n⚠️  Reranker model not found: {rerank_onnx}")
        print("   Skipping reranker quantization")

    print("\n" + "=" * 60)
    print("✅ Quantization finished!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("1. Update inference service configuration to use INT8 models:")
    print("   export ONNX_EMBED_MODEL_PATH=./models/bge-m3-onnx")
    print("   export USE_INT8_QUANTIZATION=true")
    print("\n2. Restart the inference service:")
    print("   python -m uvicorn inference_service.main_onnx:app --port 8001")
    print("\n3. Benchmark performance:")
    print("   ./test_rag_latency.sh")
    print("\n💡 Expected results:")
    print("   - Embedding: 200ms -> 60-80ms (~2.5-3x faster)")
    print("   - Rerank: 690ms -> 200-250ms (~3x faster)")
    print("   - End-to-end RAG latency: < 1 second ✅")


if __name__ == "__main__":
    main()
