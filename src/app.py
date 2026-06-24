from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List, Dict, Any

app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost model predicting telecom customer churn",
    version="1.0.0"
)

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "model/model.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions:       List[int]
    churn_probability: List[float]
    churn_label:       List[str]

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/score", response_model=PredictResponse)
def score(request: PredictRequest):
    try:
        df = pd.DataFrame(request.data)
        predictions   = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        return PredictResponse(
            predictions=       [int(p)             for p in predictions],
            churn_probability= [round(float(p), 4) for p in probabilities],
            churn_label=       ["Yes" if p == 1 else "No" for p in predictions]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))