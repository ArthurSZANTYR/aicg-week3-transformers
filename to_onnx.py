import torch.onnx
import torch
from train import VisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model_checkpoint = "saved_models/model_epoch1321.pt"  #the one with 1321 epchs
model.load_state_dict(torch.load(model_checkpoint, map_location=device))

model.eval() #evaluation mode

dummy_input = torch.randn(1, 3, 144, 144).to(device)  # Example for an RGB image of size 144x144

output_path = "model.onnx"

torch.onnx.export(model, dummy_input, output_path)

print("Conversion completed. ONNX model saved as", output_path)

