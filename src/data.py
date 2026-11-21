
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "data"
BATCH_SIZE = 16
IMAGE_SIZE = 224
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

# ImageNet normalization values used by the pretrained model.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Augmentation is used only during training.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE
])

# Validation and testing use consistent image preprocessing.
evaluation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    NORMALIZE
])


def get_data_loaders():
    # Two views of the same official training split:
    # one with augmentation and one without it.
    train_dataset = datasets.Food101(
        root=DATA_DIR,
        split="train",
        transform=train_transforms,
        download=False
    )

    validation_dataset = datasets.Food101(
        root=DATA_DIR,
        split="train",
        transform=evaluation_transforms,
        download=False
    )

    test_dataset = datasets.Food101(
        root=DATA_DIR,
        split="test",
        transform=evaluation_transforms,
        download=False
    )

    # Create a reproducible 90/10 training-validation split.
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    validation_size = int(len(indices) * VALIDATION_RATIO)

    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    train_subset = Subset(train_dataset, training_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = get_data_loaders()

    print("========== FOODVISION AI DATA ==========")
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(validation_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Number of food categories:", len(train_loader.dataset.dataset.classes))
