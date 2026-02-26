from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ICU Sepsis Risk API")

model = joblib.load("saved_models/triage_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
threshold = joblib.load("saved_models/threshold.pkl")

class PatientInput(BaseModel):
    HR_mean: float
    O2Sat_min: float
    Temp_max: float
    MAP_min: float
    Resp_mean: float
    Lactate_mean: float
    WBC_mean: float
    Age: float
    Gender: float

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: PatientInput):

    input_data = np.array([[
        data.HR_mean,
        data.O2Sat_min,
        data.Temp_max,
        data.MAP_min,
        data.Resp_mean,
        data.Lactate_mean,
        data.WBC_mean,
        data.Age,
        data.Gender
    ]])

    scaled = scaler.transform(input_data)
    proba = model.predict_proba(scaled)[0][1]

    risk = "High" if proba > threshold else "Low"

    return {
        "probability": round(float(proba), 4),
        "risk_level": risk
    }