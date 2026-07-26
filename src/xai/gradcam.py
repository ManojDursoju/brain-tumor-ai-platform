from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.preprocessing import preprocess_image
from src.models.inference import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Last convolutional layer of EfficientNet-B0
target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


def generate_gradcam(image: Image.Image):
    """
    Generate Grad-CAM visualization for a brain MRI image.

    Args:
        image (PIL.Image): Input MRI image.

    Returns:
        dict: Prediction, confidence and saved Grad-CAM image path.
    """

    # Preprocess image
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = probabilities.argmax(dim=1).item()

        confidence = probabilities.max().item()

    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Convert original image
    rgb_img = (
        np.array(image.resize((224, 224)))
        .astype(np.float32)
        / 255.0
    )

    # Overlay heatmap
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # Save output
    output_dir = Path("outputs/gradcam")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "gradcam_result.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "gradcam_path": str(output_path)
    }