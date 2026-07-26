from sqlalchemy.orm import Session

from src.db.models import Prediction


def save_prediction(
    db: Session,
    image_name: str,
    prediction: str,
    confidence: float,
    gradcam_path: str = None
):
    record = Prediction(
        image_name=image_name,
        prediction=prediction,
        confidence=confidence,
        gradcam_path=gradcam_path
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record

def get_predictions(db: Session):
    """
    Retrieve all prediction records.
    """
    return db.query(Prediction).order_by(
        Prediction.created_at.desc()
    ).all()

def delete_prediction(
    db: Session,
    prediction_id: int
):
    """
    Delete a prediction by its ID.
    """

    record = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if record is None:
        return None

    db.delete(record)
    db.commit()

    return record