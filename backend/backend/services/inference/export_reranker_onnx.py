# -*- coding: utf-8 -*-
"""
Export reranker models to ONNX format.
"""
import os
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def export_reranker_to_onnx(model_name: str, output_dir: str):
    """
    Export a reranker model to ONNX.

    Args:
        model_name: HuggingFace model name
        output_dir: output directory
    """
    print("=" * 60)
    print("🚀 Reranker ONNX Export Tool")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Loading model: {model_name}")
    # Try to load full classification model via FlagReranker
    try:
        from FlagEmbedding import FlagReranker
        model_wrapper = FlagReranker(model_name, use_fp16=False, device="cpu")
        model = model_wrapper.model.cpu()  # Force CPU usage
        tokenizer = model_wrapper.tokenizer
        model.eval()
        print("✅ Loaded model with FlagReranker (classification head included)")
    except ImportError:
        print("⚠️  FlagEmbedding is not installed, falling back to the base model")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception:
            model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

    print(f"\n🔧 Preparing export...")
    # Reranker input consists of query + document pairs
    dummy_texts = ["This is a query", "This is a document"]
    inputs = tokenizer(
        dummy_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    output_path = os.path.join(output_dir, "model.onnx")

    print(f"📝 Export destination: {output_path}")

    # Dynamic axis configuration
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'output': {0: 'batch'}
    }

    # Add token_type_ids to the dynamic axes if present
    input_names = ['input_ids', 'attention_mask']
    if 'token_type_ids' in inputs:
        input_names.append('token_type_ids')
        dynamic_axes['token_type_ids'] = {0: 'batch', 1: 'sequence'}

    # Export ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask'],
             inputs.get('token_type_ids')) if 'token_type_ids' in inputs
            else (inputs['input_ids'], inputs['attention_mask']),
            output_path,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True
        )

    print("✅ ONNX model exported successfully!")

    # Save tokenizer
    print(f"\n💾 Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # Check file size
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n📊 Model size: {model_size:.2f} MB")

    # Validate the exported model
    print(f"\n🧪 Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed!")
    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")

    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    model_name = "BAAI/bge-reranker-base"
    output_dir = "./models/bge-reranker-onnx"

    export_reranker_to_onnx(model_name, output_dir)

    print("\n📝 Next steps:")
    print("1. Quantize the model:")
    print(f"   python inference_service/quantize_to_int8.py")
    print("\n2. Start the ONNX inference service:")
    print("   export USE_INT8_QUANTIZATION=true")
    print("   python -m uvicorn inference_service.main_onnx:app --port 8001")
