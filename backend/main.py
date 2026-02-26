import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------
# 1️⃣ App Initialization
# ---------------------------------------------------

app = FastAPI(
    title="ICU Sepsis Risk Prediction API",
    description="Predicts early sepsis risk using ICU patient data",
    version="1.0.0"
)

# ---------------------------------------------------
# 2️⃣ CORS Middleware (Allows Frontend Connection)
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 3️⃣ Load Model Artifacts Safely
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "saved_models", "triage_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "saved_models", "scaler.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "saved_models", "threshold.pkl"))

# ---------------------------------------------------
# 4️⃣ Input Schema
# ---------------------------------------------------

class PatientData(BaseModel):
    Age: float
    HR: float
    O2Sat: float
    Temp: float
    SBP: float
    MAP: float
    DBP: float
    Resp: float
    Lactate: float
    WBC: float

# ---------------------------------------------------
# 5️⃣ Root Endpoint
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "ICU Sepsis Risk Prediction API",
        "status": "running",
        "docs": "/docs"
    }

# ---------------------------------------------------
# 6️⃣ Health Check Endpoint
# ---------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ---------------------------------------------------
# 7️⃣ Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    input_array = np.array([[
        data.Age,
        data.HR,
        data.O2Sat,
        data.Temp,
        data.SBP,
        data.MAP,
        data.DBP,
        data.Resp,
        data.Lactate,
        data.WBC
    ]])

    input_scaled = scaler.transform(input_array)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability > threshold)

    return {
        "sepsis_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_level": (
            "High Risk" if probability > 0.7 else
            "Medium Risk" if probability > 0.4 else
            "Low Risk"
        )
    }