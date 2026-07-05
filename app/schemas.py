"""Pydantic request and response schemas for the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: float = Field(..., ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    risk: str
    confidence: float
    model_version: str
