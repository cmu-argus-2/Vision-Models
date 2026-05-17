import torch
import torch.nn as nn
import torchvision
from pathlib import Path


NUM_CLASSES = 16
INPUT_SHAPE = (1, 3, 224, 224)
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parents[1]
DEFAULT_WEIGHTS_PATH = Path("/home/argus/ARGUS/Vision-Models/trained-rc/V4/rc_model_weights.pth")
DEFAULT_ONNX_PATH = MODELS_DIR / "trained-rc" / "V4" / "rc_model_weights.onnx"


class ClassifierEfficient(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights=None)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.efficientnet(x))

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierEfficient().to(device)
    # Load Custom model weights
    model.load_state_dict(torch.load(DEFAULT_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    # Input tensor shape (images)
    input_tensor = torch.randn(INPUT_SHAPE).to(device)

    DEFAULT_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: Changes were made 2/16/26 to the onnx file. Newer versions of onnx (with opset_version >= 18) have conversion issues with tensorRT (8.6.X) and cuda-toolkit (12.2)
    # Keep the legacy exporter behavior and opset 17 for TensorRT compatibility.
    torch.onnx.export(
        model,                  # model to export
        (input_tensor,),        # inputs of the model,
        str(DEFAULT_ONNX_PATH),  # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Saved ONNX model to {DEFAULT_ONNX_PATH}")
