
import random
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import Food101
from torchvision import transforms

# 1. Load the training dataset
dataset = Food101(
    root="data",
    split="train",
    download=False
)

print("Total training images:", len(dataset))
print("Total food categories:", len(dataset.classes))

# 2. Select 8 random images
random_indices = random.sample(range(len(dataset)), 8)

# 3. Create a visualization
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, index in zip(axes.flatten(), random_indices):

    # Retrieve image and its label
    image, label = dataset[index]

    # Display the original image
    ax.imshow(image)
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()

# 4. Inspect one image as numerical data
image, label = dataset[random_indices[0]]

to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)

print("\nImage information")
print("-----------------")
print("Food category:", dataset.classes[label])
print("Original image size:", image.size)
print("Tensor shape:", image_tensor.shape)
print("Tensor data type:", image_tensor.dtype)
print("Minimum pixel value:", image_tensor.min().item())
print("Maximum pixel value:", image_tensor.max().item())
