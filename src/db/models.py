from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime

from datetime import datetime

from src.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    image_name = Column(String, nullable=False)

    prediction = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)

    gradcam_path = Column(String)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )