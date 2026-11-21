
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 101


def build_model(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    unfreeze_last_n=0
):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(weights=weights)

    # Freeze all feature extraction layers initially.
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

        # Optionally unfreeze the final feature blocks.
        if unfreeze_last_n > 0:
            feature_blocks = list(model.features.children())

            for block in feature_blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

    # Replace the final classification layer for Food-101.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    return model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()

    model = build_model(unfreeze_last_n=3)
    model = model.to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total = sum(
        p.numel() for p in model.parameters()
    )

    print("========== FOODVISION AI FINE-TUNING ==========")
    print("Device:", device)
    print("Architecture: MobileNetV3-Small")
    print("Food categories:", NUM_CLASSES)
    print("Unfrozen feature blocks: 3")
    print("Trainable parameters:", f"{trainable:,}")
    print("Total parameters:", f"{total:,}")
