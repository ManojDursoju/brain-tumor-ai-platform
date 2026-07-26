TUMOR_INFO = {
    "glioma": {
        "description": (
            "Glioma is a tumor that develops from glial cells in the brain. "
            "Its severity varies depending on its grade and location."
        ),
        "recommendation": [
            "Consult a neurologist or neurosurgeon.",
            "Review MRI findings with a radiologist.",
            "Follow up with additional diagnostic tests if required."
        ]
    },

    "meningioma": {
        "description": (
            "Meningioma is usually a slow-growing tumor arising from the meninges "
            "that surround the brain and spinal cord."
        ),
        "recommendation": [
            "Consult a neurosurgeon.",
            "Monitor tumor growth through periodic imaging.",
            "Discuss treatment options based on size and symptoms."
        ]
    },

    "pituitary": {
        "description": (
            "Pituitary tumors occur in the pituitary gland and may affect hormone production."
        ),
        "recommendation": [
            "Consult an endocrinologist.",
            "Evaluate hormone levels.",
            "Discuss medical or surgical treatment options."
        ]
    },

    "notumor": {
        "description": (
            "No evidence of a brain tumor was detected by the AI model."
        ),
        "recommendation": [
            "Continue routine medical follow-up if symptoms persist.",
            "Consult a healthcare professional if clinically indicated."
        ]
    }
}


def generate_report(prediction: str, confidence: float):
    """
    Generate an AI medical report.
    """

    info = TUMOR_INFO.get(
        prediction.lower(),
        {
            "description": "Information unavailable.",
            "recommendation": [
                "Consult a healthcare professional."
            ]
        }
    )

    report = {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "description": info["description"],
        "recommendations": info["recommendation"],
        "disclaimer": (
            "This report is AI-generated for educational purposes only "
            "and is not a substitute for professional medical diagnosis."
        )
    }

    return report