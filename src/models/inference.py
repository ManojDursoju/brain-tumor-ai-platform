from pathlib import Path

import torch
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.data.preprocessing import preprocess_image


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Class names
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def load_model():
    """
    Load trained EfficientNet-B0 model.
    """

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    num_features = model.classifier[1].in_features

    model.classifier[1] = torch.nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )

    model_path = Path("models/best_model.pth")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


# Load once
model = load_model()


def predict_image(image: Image.Image):
    """
    Predict tumor class from PIL image.
    """

    input_tensor = preprocess_image(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = probabilities.argmax(dim=1).item()

        confidence = probabilities.max().item()

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2)
    }

def is_model_loaded():
    return model is not None


def get_device():
    return str(device)