from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class GradCAMResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam_path: str


class PredictionHistory(BaseModel):
    id: int
    image_name: str
    prediction: str
    confidence: float
    gradcam_path: Optional[str] = None
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

class ReportResponse(BaseModel):
    prediction: str
    confidence: float
    description: str
    recommendations: list[str]
    disclaimer: str