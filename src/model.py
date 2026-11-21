
import torch
import torch.nn as nn

from torchvision import models


# ==========================================
# 1. Model configuration
# ==========================================

NUM_CLASSES = 101


# ==========================================
# 2. Build MobileNetV3-Small
# ==========================================

def build_model(num_classes=NUM_CLASSES, freeze_backbone=True):

    # Load pretrained MobileNetV3-Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    # Freeze the pretrained feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Find the input size of the final classification layer
    input_features = model.classifier[-1].in_features

    # Replace the final layer with our 101-class classifier
    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    return model


# ==========================================
# 3. Select computing device
# ==========================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ==========================================
# 4. Test the model
# ==========================================

if __name__ == "__main__":

    device = get_device()

    model = build_model()

    model = model.to(device)

    # Create two example RGB images
    sample_images = torch.randn(
        2, 3, 224, 224
    ).to(device)

    # Run a forward pass
    model.eval()

    with torch.no_grad():
        predictions = model(sample_images)

    print("\n========== FOODVISION AI ==========")

    print("Selected device:", device)
    print("Model architecture: MobileNetV3-Small")
    print("Number of classes:", NUM_CLASSES)

    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nModel initialized successfully!")
