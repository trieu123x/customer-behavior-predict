"""FastAPI prediction service for the E-commerce Customer Behavior application.

Loads the exact preprocessing+model pipeline persisted by the training notebook
(notebook/customer_behavior.ipynb -> ../model/model_pipeline.joblib) and exposes it as
POST /predict. No TfidfVectorizer/scaler/encoder/model is ever re-fit here — the same
vocabulary and IDF weights learned on the training reviews are reused for every request.

Run:
    uvicorn app:app --reload --port 8003
"""
import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import TextCleaner, FEATURE_ORDER  # noqa: F401  (TextCleaner needed for unpickling)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_pipeline.joblib")
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="E-commerce Customer Interest Prediction API",
    description="Predicts whether a customer recommends a product from their profile, product category, and review text.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewInput(BaseModel):
    Age: int = Field(..., ge=10, le=100)
    Rating: int = Field(..., ge=1, le=5)
    Positive_Feedback_Count: int = Field(0, ge=0, le=1000, alias="Positive Feedback Count")
    Division_Name: str = Field(..., alias="Division Name")
    Department_Name: str = Field(..., alias="Department Name")
    Class_Name: str = Field(..., alias="Class Name")
    Review_Text: str = Field("", alias="Review Text")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 35, "Rating": 5, "Positive Feedback Count": 2,
                "Division Name": "General", "Department Name": "Dresses", "Class Name": "Dresses",
                "Review Text": "This dress is absolutely beautiful, fits perfectly and the fabric feels great. Highly recommend!",
            }
        }


class PredictionResponse(BaseModel):
    interest: str
    recommended: int
    confidence: float


@app.get("/")
def root():
    return {"status": "ok", "app": "customer_behavior", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(review: ReviewInput):
    row = {
        "Age": review.Age,
        "Rating": review.Rating,
        "Positive Feedback Count": review.Positive_Feedback_Count,
        "Division Name": review.Division_Name,
        "Department Name": review.Department_Name,
        "Class Name": review.Class_Name,
        "Review Text": review.Review_Text,
    }
    df = pd.DataFrame([row])[FEATURE_ORDER]

    pred = int(pipeline.predict(df)[0])
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(df)[0, 1])
    else:
        from scipy.special import expit

        proba = float(expit(pipeline.decision_function(df)[0]))

    return PredictionResponse(
        interest="recommended" if pred == 1 else "not_recommended",
        recommended=pred,
        confidence=round(proba if pred == 1 else 1 - proba, 4),
    )
