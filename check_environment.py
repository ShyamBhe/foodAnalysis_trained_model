
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple GPU acceleration is available!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available!")
else:
    device = torch.device("cpu")
    print("Using CPU.")

print("Selected device:", device)

# Test tensor computation
x = torch.rand(3, 3).to(device)

print("\nTest tensor:")
print(x)
