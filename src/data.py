
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Configuration
DATA_DIR = "data"
BATCH_SIZE = 16
IMAGE_SIZE = 224

# ImageNet normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Transformations for training images
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# Transformations for testing images
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def get_data_loaders():

    # Load training dataset
    train_dataset = datasets.Food101(
        root=DATA_DIR,
        split="train",
        transform=train_transforms,
        download=False
    )

    # Load official test dataset
    test_dataset = datasets.Food101(
        root=DATA_DIR,
        split="test",
        transform=test_transforms,
        download=False
    )

    # Create training batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Create testing batches
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = get_data_loaders()

    images, labels = next(iter(train_loader))

    print("Training images shape:", images.shape)
    print("Training labels shape:", labels.shape)
    print("Number of training images:", len(train_loader.dataset))
    print("Number of testing images:", len(test_loader.dataset))
    print("Number of food categories:", len(train_loader.dataset.classes))
    print("Food categories:", train_loader.dataset.classes)
