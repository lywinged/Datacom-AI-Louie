#!/usr/bin/env python3
"""
Export and quantize BGE-M3 embedding and reranker models to INT8 ONNX format.

This script:
1. Exports clean model files (ONNX, tokenizer, config) to a new directory
2. Quantizes ONNX models to INT8 for faster inference
3. Creates separate directories for embedding and reranker models
"""

import os
import shutil
from pathlib import Path
from typing import List

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
    from transformers import AutoTokenizer
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    print("⚠️  optimum library not available. Install with: pip install optimum[onnxruntime]")

try:
    import onnxruntime.quantization as ort_quant
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("⚠️  onnxruntime quantization not available")


def copy_model_files(source_dir: Path, dest_dir: Path, files_to_copy: List[str]):
    """Copy essential model files to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files_to_copy:
        source_file = source_dir / file_name
        if source_file.exists():
            dest_file = dest_dir / file_name
            if source_file.is_file():
                shutil.copy2(source_file, dest_file)
                print(f"  ✅ Copied: {file_name}")
            elif source_file.is_dir():
                shutil.copytree(source_file, dest_file, dirs_exist_ok=True)
                print(f"  ✅ Copied directory: {file_name}")
        else:
            print(f"  ⚠️  Not found: {file_name}")


def quantize_onnx_model_manual(model_path: Path, output_path: Path):
    """Manually quantize ONNX model to INT8 using onnxruntime."""
    if not ONNXRUNTIME_AVAILABLE:
        print("  ❌ onnxruntime quantization not available")
        return False

    try:
        print(f"  🔧 Quantizing {model_path.name} to INT8...")

        # Use dynamic quantization (INT8)
        ort_quant.quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=ort_quant.QuantType.QInt8,
        )

        # Check file sizes
        original_size = model_path.stat().st_size / (1024**2)  # MB
        quantized_size = output_path.stat().st_size / (1024**2)  # MB
        ratio = (original_size - quantized_size) / original_size * 100

        print(f"  ✅ Quantized successfully!")
        print(f"     Original:  {original_size:.1f} MB")
        print(f"     Quantized: {quantized_size:.1f} MB ({ratio:.1f}% reduction)")
        return True
    except Exception as e:
        print(f"  ❌ Quantization failed: {e}")
        return False


def export_bge_m3_embedding():
    """Export and quantize BGE-M3 embedding model."""
    print("\n" + "="*60)
    print("📦 Exporting BGE-M3 Embedding Model")
    print("="*60)

    source_dir = Path("./models/bge-m3-onnx-local/BAAI__bge-m3")
    dest_dir = Path("./models/bge-m3-embed-int8")

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False

    print(f"\n1. Creating clean export directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Essential files to copy
    essential_files = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    ]

    print(f"\n2. Copying essential files...")
    copy_model_files(source_dir, dest_dir, essential_files)

    # Copy ONNX model
    print(f"\n3. Processing ONNX model...")
    onnx_source = source_dir / "onnx" / "model.onnx"
    onnx_data_source = source_dir / "onnx" / "model.onnx_data"

    if onnx_source.exists():
        # Create onnx subdirectory
        onnx_dir = dest_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True)

        # Copy FP32 model
        shutil.copy2(onnx_source, onnx_dir / "model.onnx")
        if onnx_data_source.exists():
            shutil.copy2(onnx_data_source, onnx_dir / "model.onnx_data")
        print(f"  ✅ Copied FP32 model")

        # Quantize to INT8
        model_fp32 = onnx_dir / "model.onnx"
        model_int8 = onnx_dir / "model_int8.onnx"

        if quantize_onnx_model_manual(model_fp32, model_int8):
            print(f"  ✅ Created INT8 quantized model")
        else:
            print(f"  ⚠️  INT8 quantization failed, FP32 model still available")
    else:
        print(f"  ❌ ONNX model not found at: {onnx_source}")
        return False

    print(f"\n✅ Export complete: {dest_dir}")
    print(f"   You can now use: ONNX_EMBED_MODEL_PATH={dest_dir}")
    return True


def export_bge_reranker():
    """Export and quantize BGE reranker model."""
    print("\n" + "="*60)
    print("📦 Exporting BGE Reranker Model")
    print("="*60)

    # Check if source exists
    source_candidates = [
        Path("./models/bge-reranker-onnx"),
        Path("./models/bge-reranker-v2-m3"),
        Path("./models/BAAI/bge-reranker-v2-m3"),
    ]

    source_dir = None
    for candidate in source_candidates:
        if candidate.exists():
            source_dir = candidate
            break

    if not source_dir:
        print(f"⚠️  Reranker model not found. Searched:")
        for candidate in source_candidates:
            print(f"   - {candidate}")
        print("\nTo download, run:")
        print("   huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3")
        return False

    dest_dir = Path("./models/bge-reranker-int8")

    print(f"\n1. Source: {source_dir}")
    print(f"   Destination: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Essential files to copy
    essential_files = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "vocab.txt",
    ]

    print(f"\n2. Copying essential files...")
    copy_model_files(source_dir, dest_dir, essential_files)

    # Find and process ONNX model
    print(f"\n3. Processing ONNX model...")
    onnx_candidates = [
        source_dir / "model.onnx",
        source_dir / "onnx" / "model.onnx",
    ]

    onnx_source = None
    for candidate in onnx_candidates:
        if candidate.exists():
            onnx_source = candidate
            break

    if onnx_source:
        onnx_dir = dest_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True)

        # Copy FP32 model
        shutil.copy2(onnx_source, onnx_dir / "model.onnx")

        # Copy data file if exists
        onnx_data = onnx_source.parent / "model.onnx_data"
        if onnx_data.exists():
            shutil.copy2(onnx_data, onnx_dir / "model.onnx_data")

        print(f"  ✅ Copied FP32 model")

        # Quantize to INT8
        model_fp32 = onnx_dir / "model.onnx"
        model_int8 = onnx_dir / "model_int8.onnx"

        if quantize_onnx_model_manual(model_fp32, model_int8):
            print(f"  ✅ Created INT8 quantized model")
    else:
        print(f"  ⚠️  ONNX model not found. Searched:")
        for candidate in onnx_candidates:
            print(f"     - {candidate}")

    print(f"\n✅ Export complete: {dest_dir}")
    print(f"   You can now use: ONNX_RERANK_MODEL_PATH={dest_dir}")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   BGE Model Export & INT8 Quantization Tool                  ║
║                                                              ║
║   This script exports clean BGE models and quantizes them    ║
║   to INT8 for faster inference with lower memory usage.     ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Check dependencies
    if not ONNXRUNTIME_AVAILABLE:
        print("\n❌ Missing dependency: onnxruntime")
        print("   Install with: pip install onnxruntime")
        return

    # Export embedding model
    embed_success = export_bge_m3_embedding()

    # Export reranker model
    rerank_success = export_bge_reranker()

    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"  Embedding model: {'✅ Success' if embed_success else '❌ Failed'}")
    print(f"  Reranker model:  {'✅ Success' if rerank_success else '❌ Failed'}")

    if embed_success or rerank_success:
        print("\n🎉 Export complete! Update your .env file:")
        if embed_success:
            print("   ONNX_EMBED_MODEL_PATH=./models/bge-m3-embed-int8")
        if rerank_success:
            print("   ONNX_RERANK_MODEL_PATH=./models/bge-reranker-int8")

    print("="*60)


if __name__ == "__main__":
    main()
