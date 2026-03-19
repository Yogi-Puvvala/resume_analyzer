from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict
from src.predict import predict
import fitz  

class PredictionResponse(BaseModel):
    predicted_role   : str
    target_role      : str
    match_score      : float
    confidence_scores: Dict[str, float]
    present_skills   : List[str]
    missing_skills   : List[str]
    suggested_roles  : List[str]


app = FastAPI(title="Resume Screener API")

# ── Extract text from PDF bytes ─────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ── Health check ────────────────────────────────────
@app.get("/")
def health():
    return {"status": "Resume Screener API is running!"}

# ── Predict from PDF upload ─────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict_resume(
    file       : UploadFile = File(...),
    target_role: str        = Form(...)
):
    # Read PDF bytes
    pdf_bytes = await file.read()

    # Extract text
    resume_text = extract_text_from_pdf(pdf_bytes)

    # Predict
    result = predict(resume_text, target_role)

    return result