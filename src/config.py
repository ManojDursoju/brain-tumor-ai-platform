from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = BASE_DIR.parent / "models" / "best_efficientnet_b0.pth"
CLASS_MAPPING_PATH = BASE_DIR.parent / "models" / "class_mapping.json"
DATABASE_PATH = BASE_DIR.parent / "brain_tumor.db"

# Image Settings
IMAGE_SIZE = 224

# API
API_TITLE = "Brain Tumor AI API"
API_VERSION = "1.0.0"