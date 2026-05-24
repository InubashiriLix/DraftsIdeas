import torch
from model import ModelCnn

MODEL_PATH = "./model.pt"
EXPORT_ONNX_PATH = "./model.onnx"

if __name__ == "__main__":
    model = ModelCnn(28, 28, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,  # pyright: ignore[reportArgumentType]
        EXPORT_ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    print("Model exported to ONNX format successfully.")
