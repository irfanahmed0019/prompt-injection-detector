"""FastAPI service: POST /scan returns whether a prompt looks like a prompt injection."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "detector.joblib"
app = FastAPI(title="Prompt Injection Detector", version="1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "model not trained: run `python -m src.train`")
        _model = joblib.load(MODEL_PATH)
    return _model


class ScanIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
def scan(body: ScanIn):
    m = get_model()
    score = float(m.decision_function([body.text])[0])
    return {"injection": score > 0, "score": round(score, 3)}
