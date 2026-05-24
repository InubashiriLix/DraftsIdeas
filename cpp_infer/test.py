import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

from model import ModelCnn
from train import EXPORT_NAME, device, test_dataset, test_loader
from model import ModelCnn
from to_onnx import EXPORT_ONNX_PATH

model = ModelCnn(28, 28, 10)
model.load_state_dict(torch.load(EXPORT_NAME, map_location=device))

model.eval()

x = torch.randn(1, 1, 28, 28, dtype=torch.float32)

with torch.no_grad():
    torch_out = model(x).detach().cpu().numpy()

session = ort.InferenceSession(EXPORT_ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
onnx_out = session.run(None, {input_name: x.numpy()})[0]


print(np.max(np.abs(torch_out - onnx_out)))
print(np.allclose(torch_out, onnx_out, rtol=1e-4, atol=1e-5))
print(torch_out.argmax(axis=1), onnx_out.argmax(axis=1))
