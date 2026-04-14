from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from pathlib import Path

from .schemas import BorrowerInput, PredictionOutput
from .utils import engineer_features, calculate_risk_level

app = FastAPI(title="Credit Risk Prediction API",description="Predict loan default risk using Random Forest and SMOTE",version="1.0.0")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

MODEL_PATH = Path(__file__).parent.parent/"models"/"random_forest_model.pkl"

try:
    model=joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Model not found at {MODEL_PATH}")
    print("Train the model first using: python src/train_random_forest.py")
    model = None

THRESHOLD = 0.78

@app.get("/")
def read_root():
    return {
        "message": "Credit Risk Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "threshold": THRESHOLD
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_default_risk(borrower: BorrowerInput):
    if model is None:
        raise HTTPException(status_code=503,detail="Model not loaded. Train the model first.")
    
    try:
        borrower_data = {
            'revolving_utilization': borrower.revolving_utilization,
            'age': borrower.age,
            'times_30_59_late': borrower.times_30_59_late,
            'debt_ratio': borrower.debt_ratio,
            'monthly_income': borrower.monthly_income,
            'open_credit_lines': borrower.open_credit_lines,
            'times_90_late': borrower.times_90_late,
            'real_estate_loans': borrower.real_estate_loans,
            'times_60_89_late': borrower.times_60_89_late,
            'dependents': borrower.dependents
        }
        
        features = engineer_features(borrower_data)
        probabilities = model.predict_proba(features)[0]
        default_prob = float(probabilities[1])
        decision = "reject" if default_prob >= THRESHOLD else "approve"
        
        confidence = float(max(probabilities))
        risk_level = calculate_risk_level(default_prob)
        
        return PredictionOutput(default_probability=default_prob,decision=decision,confidence=confidence,risk_level=risk_level)ím
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
        
@app.get("/model-info")
def get_model_info():
    return {
        "algorithm": "Random Forest with SMOTE",
        "threshold": THRESHOLD,
        "precision": 0.43,
        "recall": 0.39,
        "savings": "$3,161,000",
        "roi": "15.8%"
    }
