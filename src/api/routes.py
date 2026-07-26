from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from PIL import Image
import io

from src.db.database import get_db
from src.db.crud import (
    save_prediction,
    get_predictions,
    delete_prediction
)

from src.models.inference import (
    predict_image,
    is_model_loaded,
    get_device
)

from src.xai.gradcam import generate_gradcam
from src.llm.report_generator import generate_report

from src.api.schemas import (
    PredictionResponse,
    PredictionHistory,
    HealthResponse,
    GradCAMResponse,
    ReportResponse
)
from fastapi.responses import FileResponse

from src.llm.pdf_generator import generate_pdf

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse
)
def health():
    """
    Check API and model status.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=is_model_loaded(),
        device=get_device()
    )


@router.get(
    "/predictions",
    response_model=list[PredictionHistory]
)
def prediction_history(
    db: Session = Depends(get_db)
):
    """
    Retrieve all saved prediction records.
    """
    return get_predictions(db)


@router.delete(
    "/predictions/{prediction_id}"
)
def delete_prediction_record(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a prediction record by ID.
    """

    record = delete_prediction(
        db=db,
        prediction_id=prediction_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    return {
        "message": "Prediction deleted successfully.",
        "deleted_id": prediction_id
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Predict brain tumor class from an uploaded MRI image.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(image)

    save_prediction(
        db=db,
        image_name=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"]
    )


@router.post(
    "/gradcam",
    response_model=GradCAMResponse
)
async def gradcam(file: UploadFile = File(...)):
    """
    Generate Grad-CAM visualization from an uploaded MRI image.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = generate_gradcam(image)

    return GradCAMResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        gradcam_path=result["gradcam_path"]
    )


@router.post(
    "/report",
    response_model=ReportResponse
)
async def report(file: UploadFile = File(...)):
    """
    Generate an AI medical report from an uploaded MRI image.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(image)

    report = generate_report(
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    return ReportResponse(**report)
@router.post("/report/pdf")
async def report_pdf(file: UploadFile = File(...)):
    """
    Generate and download an AI medical report as a PDF.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(image)

    report = generate_report(
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    pdf_path = generate_pdf(report)

    return FileResponse(
        path=pdf_path,
        filename="brain_tumor_report.pdf",
        media_type="application/pdf"
    )