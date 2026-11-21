
from torchvision.datasets import Food101
from pathlib import Path

# Dataset storage location
DATA_DIR = Path("data")

print("Starting Food-101 dataset download...")

# Download training images
train_dataset = Food101(
    root=DATA_DIR,
    split="train",
    download=True
)

print("Training dataset downloaded!")
print("Training images:", len(train_dataset))

# Load test dataset
test_dataset = Food101(
    root=DATA_DIR,
    split="test",
    download=True
)

print("Test dataset downloaded!")
print("Testing images:", len(test_dataset))

# Display food categories
print("\nNumber of food categories:", len(train_dataset.classes))

print("\nFirst 10 categories:")
print(train_dataset.classes[:10])

print("\nDataset download completed!")
