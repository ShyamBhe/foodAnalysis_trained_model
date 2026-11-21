
import torch
from torchvision.datasets import Food101
from torchvision import transforms

# Select Apple GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# Load one actual food image
dataset = Food101(
    root="data",
    split="train",
    download=False
)

image, label = dataset[0]

# Convert image into tensor
image_tensor = transforms.ToTensor()(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Move tensor to GPU
image_tensor = image_tensor.to(device)

# Perform a simple computation
result = image_tensor * 2

print("Food category:", dataset.classes[label])
print("Tensor shape:", image_tensor.shape)
print("Tensor device:", image_tensor.device)
print("Computation successful:", result.shape)
