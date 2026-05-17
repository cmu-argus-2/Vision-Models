"""
PyTorch to TensorRT Converter - State Dict Compatible
Handles .pth files that contain only state_dict (OrderedDict)
"""

import argparse
import torch
import torch.nn as nn
import torchvision
import tensorrt as trt
import os
import sys
from collections import OrderedDict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
FALLBACK_V4_WEIGHTS_PATH = "/home/argus/ARGUS/Vision-Models/trained-rc/V4/rc_model_weights.pth"


def check_pth_content(pth_path):
    """Check what's inside the .pth file"""
    print("Analyzing .pth file...")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if isinstance(checkpoint, OrderedDict):
        print("  ✓ File contains: state_dict (OrderedDict)")
        print("  ℹ You need to provide the model architecture")
        print("\n  Model layers found:")
        for i, (key, value) in enumerate(list(checkpoint.items())[:10]):
            print(f"    {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        if len(checkpoint) > 10:
            print(f"    ... and {len(checkpoint) - 10} more layers")
        return 'state_dict', checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("  ✓ File contains: checkpoint dict with 'state_dict' key")
        print("  ℹ You need to provide the model architecture")
        return 'checkpoint', checkpoint['state_dict']
    elif isinstance(checkpoint, nn.Module):
        print("  ✓ File contains: complete model")
        print("  ℹ No model architecture needed")
        return 'model', checkpoint
    else:
        print(f"  ⚠ Unknown format: {type(checkpoint)}")
        return 'unknown', checkpoint

def pth_to_trt(pth_path, trt_path, input_shape, model_architecture, device=None, fp16=True):
    """
    Convert PyTorch .pth model to TensorRT .trt engine
    
    Args:
        pth_path: Path to .pth file
        trt_path: Output path for .trt file
        input_shape: Tuple of input shape (batch, channels, height, width)
        model_architecture: PyTorch model instance or class
        device: 'cuda' or 'cpu' (default: auto-detect)
        fp16: Enable FP16 precision for faster inference (requires CUDA)
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
            fp16 = False
        
        print(f"\n{'='*60}")
        print(f"Converting {pth_path} to {trt_path}")
        print(f"Device: {device}")
        print(f"Input shape: {input_shape}")
        print(f"FP16 mode: {fp16}")
        print(f"{'='*60}\n")
        
        # Step 1: Load PyTorch model
        print("Step 1/3: Loading PyTorch model...")
        
        # Check what's in the file
        content_type, content = check_pth_content(pth_path)
        
        if content_type == 'model':
            # Complete model loaded
            model = content
        elif content_type in ['state_dict', 'checkpoint']:
            # Need to load into architecture
            if model_architecture is None:
                print("\n✗ Error: state_dict found but no model architecture provided!")
                print("\nYou need to define your model architecture and pass it as model_architecture parameter.")
                print("See the example at the bottom of this script.")
                return False
            
            # Initialize model
            if isinstance(model_architecture, type):
                # If it's a class, instantiate it
                model = model_architecture()
            else:
                # Already an instance
                model = model_architecture
            
            # Load state dict
            print("  Loading state_dict into model architecture...")
            model.load_state_dict(content)
        else:
            print(f"\n✗ Error: Unknown .pth file format: {type(content)}")
            return False
        
        model.eval()
        model.to(device)
        print("  ✓ Model loaded successfully")
        
        # Step 2: Export to ONNX
        onnx_path = trt_path.replace('.trt', '.onnx')
        print(f"\nStep 2/3: Exporting to ONNX ({onnx_path})...")
        
        dummy_input = torch.randn(input_shape).to(device)
        
        # Use legacy TorchScript-based exporter for compatibility
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],   # Must match C++ runtimes.cpp
                output_names=['output'], # Must match C++ runtimes.cpp
                dynamic_axes={
                    'input': {0: 'batch'},
                    'output': {0: 'batch'}
                },
                verbose=False,
            )
        print("  ✓ ONNX export successful")
        
        # Step 3: Build TensorRT engine
        print("\nStep 3/3: Building TensorRT engine...")
        print("  This may take a few minutes...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        print("  Parsing ONNX model...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("  ✗ Failed to parse ONNX file:")
                for error in range(parser.num_errors):
                    print(f"    Error {error}: {parser.get_error(error)}")
                return False
        
        print("  ✓ ONNX parsed successfully")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Add optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        # Set min, optimal, and max batch sizes (using input shape from parameter)
        min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        max_shape = (8, input_shape[1], input_shape[2], input_shape[3])
        profile.set_shape("input", min_shape, opt_shape, max_shape)  # Must match input name
        config.add_optimization_profile(profile)
        print("  ✓ Optimization profile added")
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 mode enabled")
        elif fp16:
            print("  ⚠ FP16 requested but not supported on this platform")
        
        # Build engine
        print("  Building engine (this is the slow part)...")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            print("  ✗ Failed to build TensorRT engine")
            return False
        
        print("  ✓ Engine built successfully")
        
        # Save engine (build_serialized_network returns bytes directly)
        print(f"  Saving engine to {trt_path}...")
        with open(trt_path, 'wb') as f:
            f.write(engine)
        
        print("  ✓ Engine saved successfully")
        
        # Clean up ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"  ✓ Cleaned up intermediate ONNX file")
        
        print(f"\n{'='*60}")
        print("✓ CONVERSION SUCCESSFUL!")
        print(f"{'='*60}\n")
        print(f"TensorRT engine saved to: {trt_path}")
        print(f"File size: {os.path.getsize(trt_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {type(e).__name__}")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# DEFINE YOUR MODEL ARCHITECTURE HERE
# ============================================================================

class ClassifierEfficient(nn.Module):
    """
    EfficientNet-B0 based classifier for region classification.
    """
    def __init__(self, num_classes=16):
        super(ClassifierEfficient, self).__init__()

        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = None
        self.efficientnet = torchvision.models.efficientnet_b0(weights=weights)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RC PyTorch .pth weights to a TensorRT .trt engine."
    )
    parser.add_argument(
        "--rc-base-folder",
        "--base-rc-folder",
        default=os.path.join(MODELS_DIR, "trained-rc"),
        help="Base folder containing versioned RC model folders.",
    )
    parser.add_argument(
        "--version",
        default="V4",
        help="RC model version folder under the base RC folder.",
    )
    parser.add_argument(
        "--weights-name",
        default="rc_model_weights",
        help="Weights filename stem used for both .pth input and .trt output.",
    )
    parser.add_argument(
        "--pth-path",
        default=None,
        help="Optional full input .pth path. Overrides --rc-base-folder/--version.",
    )
    parser.add_argument(
        "--trt-path",
        default=None,
        help="Optional full output .trt path. Overrides --rc-base-folder/--version.",
    )
    parser.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        default=(1, 3, 224, 224),
        metavar=("BATCH", "CHANNELS", "HEIGHT", "WIDTH"),
        help="Input tensor shape for ONNX export and TensorRT profiling.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=16,
        help="Number of RC output classes for the EfficientNet classifier.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Device to use for conversion. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        default=True,
        help="Disable FP16 TensorRT conversion.",
    )
    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    rc_folder = os.path.join(args.rc_base_folder, args.version)
    pth_model_path = args.pth_path or os.path.join(
        rc_folder, f"{args.weights_name}.pth"
    )
    if (
        args.pth_path is None
        and args.version == "V4"
        and not os.path.exists(pth_model_path)
        and os.path.exists(FALLBACK_V4_WEIGHTS_PATH)
    ):
        pth_model_path = FALLBACK_V4_WEIGHTS_PATH
    trt_model_path = args.trt_path or os.path.join(
        rc_folder, f"{args.weights_name}.trt"
    )
    input_shape = tuple(args.input_shape)

    # Create model architecture instance
    # If your .pth contains only state_dict, you MUST provide the architecture.
    model_architecture = ClassifierEfficient(num_classes=args.num_classes)
    
    print("PyTorch to TensorRT Converter")
    print("="*60 + "\n")
    
    # Check if model file exists
    print(f"RC base folder: {args.rc_base_folder}")
    print(f"RC version: {args.version}")
    print(f"Input PTH: {pth_model_path}")
    print(f"Output TRT: {trt_model_path}")

    if not os.path.exists(pth_model_path):
        print(f"✗ Model file not found: {pth_model_path}")
        print("\nPlease check --rc-base-folder/--version or pass --pth-path.")
        sys.exit(1)

    trt_dir = os.path.dirname(os.path.abspath(trt_model_path))
    os.makedirs(trt_dir, exist_ok=True)
    
    # Perform conversion
    success = pth_to_trt(
        pth_path=pth_model_path,
        trt_path=trt_model_path,
        input_shape=input_shape,
        model_architecture=model_architecture,  # Your model architecture
        device=args.device,
        fp16=args.fp16,
    )
    
    if not success:
        print("\nConversion failed. Please check the errors above.")
        sys.exit(1)
