import torch.onnx
import torch
from train import VisionTransformer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of your model and move it to the same device as the input tensor
model = VisionTransformer(
    img_size=144,
    patch_size=12,
    in_channels=3,
    embed_dim=128,
    num_heads=64,
    mlp_dim=128,
    num_layers=6,
    num_classes=4,
).to(device)

# Load the model state from the .pt file
model_checkpoint = "saved_models/model_epoch1321.pt"
model.load_state_dict(torch.load(model_checkpoint, map_location=device))

# Set the model to evaluation mode
model.eval()

# Dummy input for conversion (replace with appropriate values)
dummy_input = torch.randn(1, 3, 144, 144).to(device)  # Example for an RGB image of size 144x144

# Specify the output path for the ONNX model
output_path = "model.onnx"

# Use torch.onnx.export function for conversion
torch.onnx.export(model, dummy_input, output_path)

print("Conversion completed. ONNX model saved as", output_path)

