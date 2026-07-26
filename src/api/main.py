from fastapi import FastAPI
from src.api.routes import router
from src.db.database import engine
from src.db.models import Base

# Create's database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Brain Tumor AI API",
    version="1.0.0"
)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Brain Tumor AI API Running"}

# Register all API routes
app.include_router(router)