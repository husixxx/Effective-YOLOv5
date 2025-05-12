import torch
import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType
from models.yolo import Model, Detect, DetectMultiBackend
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8/UINT8")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to onnx model weights')
    parser.add_argument('--onnx', type=str, default='',
                        help='Path to save ONNX model (default: <weights_name>.onnx)')
    parser.add_argument('--quantized', type=str, default='./quantized.onnx',
                        help='Path to save quantized ONNX model (default: <weights_name>_quant.onnx)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for dummy input (default: 640)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default paths if not provided
    if not args.onnx:
        args.onnx = os.path.splitext(args.weights)[0] + '.onnx'
    if not args.quantized:
        args.quantized = os.path.splitext(args.onnx)[0] + '_quant.onnx'
    
    print(f"Loading model: {args.weights}")
    model = DetectMultiBackend(args.weights, device='cpu', dnn=False)
    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    
    # Check if ONNX file already exists
    if not os.path.exists(args.onnx):
        print(f"Exporting to ONNX: {args.onnx}")
        torch.onnx.export(
            model, 
            dummy_input, 
            args.onnx,
            opset_version=12,
            do_constant_folding=True
        )
    else:
        print(f"ONNX file already exists: {args.onnx}")
    
    # Quantization type
    quant_type = QuantType.QUInt8
    quant_type_name = "UINT8"
    
    print(f"Quantizing to {quant_type_name}: {args.quantized}")
    quantize_dynamic(
        args.onnx,
        args.quantized,
        weight_type=quant_type
    )

    # Check size difference
    orig_size = os.path.getsize(args.onnx) / (1024 * 1024)
    quant_size = os.path.getsize(args.quantized) / (1024 * 1024)

    print(f"Base size: {orig_size:.2f} MB")
    print(f"Quantized: {quant_size:.2f} MB")
    print(f"Reduction: {orig_size/quant_size:.2f}x")

if __name__ == "__main__":
    main()