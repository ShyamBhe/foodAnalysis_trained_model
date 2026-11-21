
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_data_loaders
from model import build_model, get_device


# ==========================================
# CONFIGURATION
# ==========================================

EPOCHS = 3

FEATURE_LEARNING_RATE = 0.00001
CLASSIFIER_LEARNING_RATE = 0.0001

BASELINE_PATH = "models/foodvision_baseline.pth"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "foodvision_finetuned_best.pth"
)


# ==========================================
# TRAINING
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    # Keep frozen feature blocks in evaluation mode.
    model.features[:-3].eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Fine-tuning")

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{100 * correct / total:.2f}%"
        )

    return running_loss / len(loader), 100 * correct / total


# ==========================================
# VALIDATION
# ==========================================

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Validation"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), 100 * correct / total


# ==========================================
# MAIN
# ==========================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()

    print("\n========== FOODVISION AI FINE-TUNING ==========")
    print("Device:", device)

    train_loader, validation_loader, _ = get_data_loaders()

    # Build model with the final three feature blocks unfrozen.
    model = build_model(unfreeze_last_n=3)
    model = model.to(device)

    # Load the baseline Food-101 weights.
    checkpoint = torch.load(
        BASELINE_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded baseline checkpoint.")
    print("Baseline validation accuracy:",
          f"{checkpoint['validation_accuracy']:.2f}%")

    # Use separate learning rates for feature layers and classifier.
    feature_parameters = [
        p for p in model.features.parameters()
        if p.requires_grad
    ]

    classifier_parameters = list(model.classifier.parameters())

    optimizer = Adam([
        {
            "params": feature_parameters,
            "lr": FEATURE_LEARNING_RATE
        },
        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LEARNING_RATE
        }
    ])

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        print("\nEpoch results:")
        print(f"Training loss:       {train_loss:.4f}")
        print(f"Training accuracy:   {train_accuracy:.2f}%")
        print(f"Validation loss:     {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": val_accuracy
                },
                MODEL_PATH
            )

            print("Fine-tuned checkpoint saved:", MODEL_PATH)

    print("\n========== FINE-TUNING COMPLETE ==========")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("Checkpoint:", MODEL_PATH)


if __name__ == "__main__":
    main()
