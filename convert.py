import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Convert saved PyTorch model to ONNX and QONNX formats.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved PyTorch model (.pt file)')
    parser.add_argument('--save_name', type=str, default='model', help='Base name for the saved ONNX models')
    parser.add_argument('--cifar', action='store_true', help='Use CIFAR model architecture')
    args = parser.parse_args()
    
    # Import or define DiT_Llama
    # Ensure dit.py is in the same directory or installed as a package
    try:
        from dit import DiT_Llama
    except ImportError:
        raise ImportError("The 'dit' module with 'DiT_Llama' class is required. Please ensure it is available.")
    
    # Determine channels and model parameters based on CIFAR or MNIST
    if args.cifar:
        channels = 3
        model = DiT_Llama(
            channels,
            32,
            dim=256,
            n_layers=10,
            n_heads=8,
            num_classes=10
        ).to(device)
    else:
        channels = 1
        model = DiT_Llama(
            channels,
            32,
            dim=64,
            n_layers=3,
            n_heads=2,
            num_classes=10
        ).to(device)
    
    # Load the saved model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    dummy_zt = torch.randn(batch_size, channels, 32, 32).to(device)
    dummy_t = torch.tensor([0.5] * batch_size).to(device)
    dummy_cond = torch.tensor([0] * batch_size).to(device)
    
    # Wrap the model to handle multiple inputs
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model
        
        def forward(self, zt, t, cond):
            return self.model(zt, t, cond)
    
    wrapped_model = WrappedModel(model)
    
    # Export to ONNX
    onnx_path = f"{args.save_name}.onnx"
    print(f"Converting model to ONNX and saving at {onnx_path}")
    torch.onnx.export(
        wrapped_model,
        (dummy_zt, dummy_t, dummy_cond),
        onnx_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['zt', 't', 'cond'],
        output_names=['output'],
        dynamic_axes={
            'zt': {0: 'batch_size'},
            't': {0: 'batch_size'},
            'cond': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model has been converted to ONNX and saved at {onnx_path}")
    
    # Quantize the ONNX model to QONNX
    qonnx_path = f"{args.save_name}_quantized.onnx"
    quantize_dynamic(
        onnx_path,
        qonnx_path,
        weight_type=QuantType.QInt8
    )
    
    print(f"Quantized model has been saved at {qonnx_path}")

if __name__ == "__main__":
    main()
