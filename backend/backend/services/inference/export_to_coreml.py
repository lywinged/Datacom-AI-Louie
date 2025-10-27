"""
Convert an ONNX model to CoreML (optimized for macOS Metal acceleration).

Usage:
    # Export the ONNX model first
    python export_to_onnx.py --model BAAI/bge-m3 --output ./models/bge-m3-onnx

    # Then convert to CoreML
    python export_to_coreml.py --input ./models/bge-m3-onnx/model.onnx --output ./models/bge-m3.mlmodel
"""
import argparse
import os


def convert_to_coreml(onnx_path: str, output_path: str, optimize: bool = True):
    """Convert an ONNX model to CoreML."""
    print(f"\n📦 Converting ONNX to CoreML")
    print(f"   Input: {onnx_path}")
    print(f"   Output: {output_path}")

    try:
        import coremltools as ct
        import onnx

        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        print(f"✅ ONNX model loaded")

        # Convert to CoreML
        print(f"🔄 Converting to CoreML (this may take a few minutes)...")

        # Baseline conversion
        coreml_model = ct.convert(
            onnx_model,
            convert_to="mlprogram",  # Leverage the modern ML Program format (Metal compatible)
            compute_units=ct.ComputeUnit.ALL,  # Use all available devices (CPU + GPU)
            minimum_deployment_target=ct.target.macOS13  # macOS 13+
        )

        # Optional optimization
        if optimize:
            print(f"🔧 Optimizing CoreML model...")
            # FP16 quantization to shrink the model and speed up inference
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model,
                nbits=16
            )

        # Save the converted model
        coreml_model.save(output_path)
        print(f"✅ CoreML model saved to {output_path}")

        # Show model metadata
        spec = coreml_model.get_spec()
        print(f"\n📊 Model Info:")
        print(f"   Inputs: {[i.name for i in spec.description.input]}")
        print(f"   Outputs: {[o.name for o in spec.description.output]}")

    except ImportError as e:
        print(f"\n❌ Missing dependencies. Please install:")
        print(f"   pip install coremltools onnx")
        raise e
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to CoreML")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output CoreML model path")
    parser.add_argument("--no-optimize", action="store_true", help="Skip optimization")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ONNX to CoreML Converter")
    print("=" * 60)

    convert_to_coreml(args.input, args.output, optimize=not args.no_optimize)

    print("\n" + "=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Test the model:")
    print(f"   python test_coreml_inference.py --model {args.output}")
    print(f"2. Use in inference service:")
    print(f"   export COREML_MODEL_PATH={args.output}")
    print(f"   python -m inference_service.main_coreml")


if __name__ == "__main__":
    main()
