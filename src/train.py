
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

EPOCHS = 4
LEARNING_RATE = 0.001

# Initial smoke test: process only a few batches.
# Set both values to None for full training.
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "foodvision_best.pth")


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    # Keep the frozen feature extractor in evaluation mode.
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    for batch_index, (images, labels) in enumerate(progress):

        if MAX_TRAIN_BATCHES is not None:
            if batch_index >= MAX_TRAIN_BATCHES:
                break

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate prediction error.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update trainable weights.
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{100 * correct / total:.2f}%"
        )

    average_loss = running_loss / (batch_index + 1)
    accuracy = 100 * correct / total

    return average_loss, accuracy


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation")

        for batch_index, (images, labels) in enumerate(progress):

            if MAX_VAL_BATCHES is not None:
                if batch_index >= MAX_VAL_BATCHES:
                    break

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = running_loss / (batch_index + 1)
    accuracy = 100 * correct / total

    return average_loss, accuracy


# ==========================================
# MAIN
# ==========================================

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()

    print("\n========== FOODVISION AI TRAINING ==========")
    print("Selected device:", device)

    # Load datasets.
    train_loader, validation_loader, test_loader = get_data_loaders()

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # Build the pretrained model.
    model = build_model()
    model = model.to(device)

    # Classification loss.
    criterion = nn.CrossEntropyLoss()

    # Optimize only trainable parameters.
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Resume from the previous checkpoint.
    start_epoch = 0
    best_accuracy = 0.0

    if os.path.exists(MODEL_PATH):
         checkpoint = torch.load(MODEL_PATH, map_location=device)

         model.load_state_dict(checkpoint["model_state_dict"])
         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

         start_epoch = checkpoint["epoch"]
         best_accuracy = checkpoint["validation_accuracy"]

         print(f"Resuming from epoch {start_epoch}")
         print(f"Previous best validation accuracy: {best_accuracy:.2f}%")

    for epoch in range(start_epoch, start_epoch + EPOCHS):

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
        print(f"Training loss:      {train_loss:.4f}")
        print(f"Training accuracy:  {train_accuracy:.2f}%")
        print(f"Validation loss:    {val_loss:.4f}")
        print(f"Validation accuracy:{val_accuracy:.2f}%")

        # Save the best model.
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

            print("Best model checkpoint saved:", MODEL_PATH)

    print("\n========== TRAINING FINISHED ==========")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("Model location:", MODEL_PATH)


if __name__ == "__main__":
    main()
