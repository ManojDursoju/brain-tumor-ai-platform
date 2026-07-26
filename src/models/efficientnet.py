import torch
from torchvision import models
import torch.nn as nn


def create_model(num_classes=4):

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model