import torch
import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from models.yolo import Model, Detect, DetectMultiBackend
import onnxruntime as ort


# init model
weights_path = '../../models/ft0.8/weights/last.pt'
model = DetectMultiBackend(weights_path, device='cpu', dnn=False)
model.eval()

# dummy input for export
dummy_input = torch.randn(1, 3, 640, 640)


onnx_model_path = "model_original.onnx"
quantized_model_path = "model_quantized_int8.onnx"

# UINT8 quntization (INT8 wasnt working..)
quantize_dynamic(
    onnx_model_path,
    quantized_model_path,
    weight_type=QuantType.QUInt8
)

print(f"Quantized model: {quantized_model_path}")

# Check size difference
orig_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
quant_size = os.path.getsize(quantized_model_path) / (1024 * 1024)

print(f"Base size: {orig_size:.2f} MB")
print(f"Quantized: {quant_size:.2f} MB")
print(f"Difference": {orig_size/quant_size:.2f}x")
