# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Optional

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk",
    version="1.0.0"
)

MODEL_PATH = "models/diabetes_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")

model = joblib.load(MODEL_PATH)


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: int = Field(..., ge=0, description="Plasma glucose concentration 2 hours in oral glucose tolerance test")
    BloodPressure: int = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    BMI: float = Field(..., ge=0, description="Body mass index (weight in kg / height in m²)")
    Age: int = Field(..., ge=0, description="Age in years")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_diabetic: bool
    probability_diabetic: Optional[float] = None
    raw_prediction: int


@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API is running",
        "interactive_docs": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_diabetes(data: DiabetesInput):
    try:
        input_array = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.BMI,
            data.Age
        ]])

        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]

        is_diabetic = bool(prediction)
        confidence = float(np.max(probabilities) * 100)
        prob_diabetic = float(probabilities[1] * 100) if len(probabilities) > 1 else None

        result_text = "Diabetic" if is_diabetic else "Not Diabetic"

        return PredictionResponse(
            prediction=result_text,
            confidence=round(confidence, 2),
            is_diabetic=is_diabetic,
            probability_diabetic=round(prob_diabetic, 2) if prob_diabetic is not None else None,
            raw_prediction=int(prediction)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)