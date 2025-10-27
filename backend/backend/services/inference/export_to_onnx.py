"""
Export PyTorch models to ONNX format.

Usage:
    python export_to_onnx.py --model BAAI/bge-m3 --output ./models/bge-m3-onnx
    python export_to_onnx.py --model BAAI/bge-reranker-base --output ./models/reranker-onnx --task text-classification
"""
import argparse
import os
from pathlib import Path


def export_embedding_model(model_name: str, output_dir: str):
    """Export an embedding model to ONNX."""
    print(f"\n📦 Exporting embedding model: {model_name}")
    print(f"   Output: {output_dir}")

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import torch

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Export to ONNX
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True
        )

        # Save artifacts
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"✅ Embedding model exported to {output_dir}")
        print(f"   Files: model.onnx, tokenizer_config.json, etc.")

    except ImportError as e:
        print(f"\n❌ Missing dependencies. Please install:")
        print(f"   pip install optimum[onnxruntime-gpu] transformers")
        raise e


def export_reranker_model(model_name: str, output_dir: str):
    """Export a reranker model to ONNX."""
    print(f"\n📦 Exporting reranker model: {model_name}")
    print(f"   Output: {output_dir}")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Export to ONNX
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True
        )

        # Save artifacts
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"✅ Reranker model exported to {output_dir}")
        print(f"   Files: model.onnx, tokenizer_config.json, etc.")

    except ImportError as e:
        print(f"\n❌ Missing dependencies. Please install:")
        print(f"   pip install optimum[onnxruntime-gpu] transformers")
        raise e


def optimize_onnx_model(model_path: str):
    """Optionally optimize an ONNX model."""
    print(f"\n🔧 Optimizing ONNX model: {model_path}")

    try:
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import OptimizationConfig

        optimizer = ORTOptimizer.from_pretrained(model_path)

        optimization_config = OptimizationConfig(
            optimization_level=2,  # 0-99, higher = more aggressive
            enable_transformers_specific_optimizations=True,
            fp16=True  # FP16 quantization for GPU
        )

        optimizer.optimize(
            save_dir=model_path,
            optimization_config=optimization_config
        )

        print(f"✅ Model optimized")

    except Exception as e:
        print(f"⚠️  Optimization failed (optional): {e}")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--task",
        default="feature-extraction",
        choices=["feature-extraction", "text-classification"],
        help="Model task type"
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ONNX Model Export Tool")
    print("=" * 60)

    if args.task == "feature-extraction":
        export_embedding_model(args.model, args.output)
    elif args.task == "text-classification":
        export_reranker_model(args.model, args.output)

    if args.optimize:
        optimize_onnx_model(args.output)

    print("\n" + "=" * 60)
    print("✅ Export Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Set environment variable:")
    print(f"   export ONNX_EMBED_MODEL_PATH={args.output}")
    print(f"2. Start inference service:")
    print(f"   python -m inference_service.main_onnx")
    print(f"3. Test with:")
    print(f'   curl -X POST http://localhost:8001/embed \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"texts": ["hello world"]}}\'')


if __name__ == "__main__":
    main()
