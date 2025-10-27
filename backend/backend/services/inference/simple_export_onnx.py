"""
Simplified ONNX export script that relies only on PyTorch, not Optimum.
"""
import torch
import os
from transformers import AutoModel, AutoTokenizer


def export_model_to_onnx(model_name: str, output_dir: str):
    """Export an ONNX model directly with PyTorch."""
    print(f"\n📦 Exporting {model_name} to ONNX...")
    print(f"   Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Switch to evaluation mode
    model.eval()

    # Create dummy input
    dummy_text = "This is a sample sentence for export."
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Export to ONNX
    output_path = os.path.join(output_dir, "model.onnx")

    print(f"🔄 Exporting to {output_path}...")

    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Export completed!")
    print(f"   Files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"   - {f} ({size:.1f} MB)")


if __name__ == "__main__":
    import sys

    # Export embedding model
    print("=" * 60)
    print("🚀 Simple ONNX Export Tool")
    print("=" * 60)

    export_model_to_onnx(
        "BAAI/bge-m3",
        "./models/bge-m3-onnx"
    )

    print("\n" + "=" * 60)
    print("✅ All done!")
    print("=" * 60)
