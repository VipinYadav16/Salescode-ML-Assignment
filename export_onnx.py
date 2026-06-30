import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import mobilenet_v2

def build_model():
    model = mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    return model

def main():
    model_pth = Path("artifacts/mobilenet_recapture.pth")
    model_onnx = Path("artifacts/mobilenet_recapture.onnx")
    
    print("Loading PyTorch model weights...")
    model = build_model()
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.eval()
    
    print("Exporting model to ONNX format...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(model_onnx),
        export_params=True,
        opset_version=18,  # Using recommended opset 18
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"ONNX model successfully exported to {model_onnx}")

if __name__ == "__main__":
    main()
