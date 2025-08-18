import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="Telecom Multisimmer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([r.model_dump() for r in req.records])
        pipe = app.state.model
        proba = pipe.predict_proba(df)[:, 1]
        labels = (proba >= 0.5).astype(int).tolist()
        return PredictResponse(probabilities=proba.tolist(), labels=labels)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
