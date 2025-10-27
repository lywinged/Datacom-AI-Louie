"""
Export a reranker model to ONNX with the classification head included.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def export_reranker_with_classifier(model_name: str, output_dir: str):
    """
    Export a complete reranker model (including classifier head) to ONNX.
    """
    print("=" * 60)
    print("🚀 Full Reranker ONNX Export")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Loading model: {model_name}")

    # Load sequence classification model with a single-output head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Reranker outputs a single score
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Force CPU execution
    model = model.cpu()
    model.eval()

    print("✅ Model loaded with classifier head")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Output labels: {model.config.num_labels}")

    # Prepare sample inputs
    print(f"\n🔧 Preparing export...")
    query = "What is machine learning?"
    document = "Machine learning is a subset of artificial intelligence."

    # Reranker input format: [query, document]
    inputs = tokenizer(
        [query, document],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Ensure tensors are on CPU
    inputs = {k: v.cpu() for k, v in inputs.items()}

    output_path = os.path.join(output_dir, "model.onnx")
    print(f"📝 Export destination: {output_path}")

    # Prepare input names and dynamic axes
    input_names = ['input_ids', 'attention_mask']
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch'}
    }

    if 'token_type_ids' in inputs:
        input_names.append('token_type_ids')
        dynamic_axes['token_type_ids'] = {0: 'batch', 1: 'sequence'}

    # Export to ONNX
    print("\n🔄 Starting export...")
    with torch.no_grad():
        # Prepare input tuple
        input_tuple = tuple(inputs[name] for name in input_names)

        torch.onnx.export(
            model,
            input_tuple,
            output_path,
            input_names=input_names,
            output_names=['logits'],  # Classifier head outputs logits
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )

    print("✅ ONNX export complete!")

    # Save tokenizer
    print(f"\n💾 Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # Report model size
    if os.path.exists(output_path):
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n📊 Model details:")
        print(f"   File size: {model_size:.2f} MB")

    # Check external data file
    data_file = output_path + "_data"
    if os.path.exists(data_file):
        data_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"   External data: {data_size:.2f} MB")

    # Validate ONNX graph
    print(f"\n🧪 Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed!")

        # Print model IO signature
        print("\n📋 Model interface:")
        print("   Inputs:")
        for inp in onnx_model.graph.input:
            print(f"      - {inp.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]}")
        print("   Outputs:")
        for out in onnx_model.graph.output:
            print(f"      - {out.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]}")

    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")

    # Test inference
    print(f"\n🔬 Testing ONNX inference...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)

        # Prepare inputs
        ort_inputs = {
            name: inputs[name].numpy() for name in input_names
        }

        # Run inference
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]

        print(f"✅ Inference test succeeded!")
        print(f"   Output shape: {logits.shape}")
        print(f"   Sample value: {logits[0][0]:.4f}")

    except Exception as e:
        print(f"❌ Inference test failed: {e}")

    print("\n" + "=" * 60)
    print("✅ Export finished!")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    model_name = "BAAI/bge-reranker-base"
    output_dir = "./models/bge-reranker-onnx"

    try:
        export_reranker_with_classifier(model_name, output_dir)

        print("\n📝 Next steps:")
        print("1. Quantize the model:")
        print(f"   python inference_service/quantize_to_int8.py")
        print("\n2. Benchmark performance:")
        print("   curl -X POST http://localhost:8001/rerank ...")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
