import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import pickle
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ── Constants ───────────────────────────────────────
MAX_LENGTH = 300

# ── Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load artifacts ──────────────────────────────────
tokenizer     = pickle.load(open(os.path.join(BASE_DIR, "models/tokenizer.pkl"),     "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb"))
model         = load_model(os.path.join(BASE_DIR, "models/my_model.keras"))

print("Model and artifacts loaded successfully!")

# ── Role Skills ─────────────────────────────────────
ROLE_SKILLS = {
    "Information-Technology": [
        "python", "java", "sql", "git", "docker",
        "linux", "rest api", "agile", "testing",
        "algorithms", "data structures", "javascript"
    ],
    "Engineering": [
        "autocad", "matlab", "solidworks", "mechanical",
        "electrical", "civil", "project management",
        "manufacturing", "quality control", "python"
    ],
    "Accountant": [
        "accounting", "excel", "tally", "gst", "taxation",
        "financial reporting", "audit", "balance sheet",
        "sql", "power bi"
    ],
    "Finance": [
        "financial analysis", "excel", "python", "sql",
        "risk management", "investment", "portfolio",
        "banking", "accounting", "bloomberg"
    ],
    "Healthcare": [
        "patient care", "clinical", "medical",
        "diagnosis", "nursing", "pharmacology",
        "emr", "hipaa", "anatomy", "surgery"
    ],
    "HR": [
        "recruitment", "payroll", "onboarding",
        "performance management", "hris", "excel",
        "labour law", "training", "compensation"
    ],
    "Designer": [
        "photoshop", "illustrator", "figma", "sketch",
        "ui", "ux", "wireframe", "typography",
        "branding", "adobe xd"
    ],
    "Sales": [
        "crm", "salesforce", "lead generation",
        "negotiation", "b2b", "revenue", "cold calling",
        "excel", "target", "pipeline"
    ],
    "Digital-Media": [
        "seo", "social media", "content writing",
        "google analytics", "email marketing",
        "wordpress", "copywriting", "canva", "strategy"
    ],
    "Banking": [
        "banking", "credit analysis", "loans",
        "kyc", "aml", "excel", "financial modelling",
        "risk", "compliance", "sql"
    ],
    "Consultant": [
        "consulting", "strategy", "powerpoint", "excel",
        "business analysis", "project management",
        "stakeholder", "presentation", "research"
    ],
    "Teacher": [
        "teaching", "curriculum", "lesson planning",
        "assessment", "classroom", "education",
        "communication", "mentoring", "training"
    ],
    "Business-Development": [
        "business development", "sales", "partnerships",
        "negotiation", "crm", "market research",
        "strategy", "revenue", "networking", "b2b"
    ],
    "Public-Relations": [
        "pr", "media relations", "press release",
        "communication", "crisis management",
        "social media", "branding", "events"
    ],
    "Advocate": [
        "law", "legal", "litigation", "contracts",
        "compliance", "court", "research",
        "drafting", "negotiation", "client"
    ],
    "Aviation": [
        "aviation", "pilot", "atc", "aircraft",
        "safety", "navigation", "faa", "iata",
        "maintenance", "operations"
    ],
    "Agriculture": [
        "agriculture", "farming", "soil",
        "irrigation", "crop", "fertilizer",
        "pest control", "agronomy", "harvesting"
    ],
    "Automobile": [
        "automobile", "automotive", "engine",
        "mechanical", "autocad", "quality",
        "manufacturing", "maintenance", "diagnostics"
    ],
    "BPO": [
        "bpo", "customer service", "call center",
        "communication", "crm", "english",
        "problem solving", "escalation", "kpi"
    ],
    "Chef": [
        "cooking", "culinary", "food safety",
        "menu planning", "kitchen", "haccp",
        "pastry", "catering", "nutrition"
    ],
    "Fitness": [
        "fitness", "personal training", "nutrition",
        "workout", "gym", "yoga", "physiology",
        "weight loss", "coaching", "anatomy"
    ],
    "Apparel": [
        "fashion", "textile", "merchandising",
        "garment", "pattern making", "retail",
        "fabric", "design", "buying", "trend"
    ],
    "Arts": [
        "painting", "sculpture", "photography",
        "creative", "exhibition", "portfolio",
        "drawing", "animation", "video editing"
    ],
    "Construction": [
        "construction", "civil", "autocad",
        "project management", "site management",
        "estimation", "structural", "safety", "contracts"
    ]
}

# ── Text cleaning ───────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]',   '', text)
    text = re.sub(r'\s+',           ' ', text)
    return text.strip()

# ── Match score + skill gap ─────────────────────────
def compute_match_score(resume_text: str,
                        target_role: str) -> tuple:
    text   = resume_text.lower()
    skills = ROLE_SKILLS.get(target_role, [])

    # ✅ Fixed — return empty lists not skills list
    if not skills:
        return 0.0, [], []

    present = [s for s in skills if s in text]
    missing = [s for s in skills if s not in text]
    score   = round((len(present) / len(skills)) * 100, 2)

    return score, present, missing

# ── Suggested roles ─────────────────────────────────
def get_suggested_roles(predictions: np.ndarray,
                        label_encoder) -> list:
    top2_indices = np.argsort(predictions[0])[::-1][:2]
    return [label_encoder.classes_[i] for i in top2_indices]

# ── Main predict function ───────────────────────────
def predict(resume_text: str, target_role: str) -> dict:

    # Step 1: Clean and tokenize
    cleaned  = clean_text(resume_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(
                    sequence,
                    maxlen     = MAX_LENGTH,
                    padding    = 'post',
                    truncating = 'post'
               )

    # Step 2: Predict
    predictions     = model.predict(padded, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_role  = label_encoder.inverse_transform(
                        [predicted_index]
                      )[0]

    # Step 3: Confidence scores
    confidence_scores = {
        label_encoder.classes_[i]: round(
            float(predictions[0][i]) * 100, 2
        )
        for i in range(len(label_encoder.classes_))
    }

    # Step 4: Match score + skills
    match_score, present_skills, missing_skills = compute_match_score(
        resume_text, target_role
    )

    # Step 5: Suggested roles
    suggested_roles = get_suggested_roles(predictions, label_encoder)

    return {
        "predicted_role"   : predicted_role,
        "target_role"      : target_role,
        "match_score"      : match_score,
        "confidence_scores": confidence_scores,
        "present_skills"   : present_skills,
        "missing_skills"   : missing_skills,
        "suggested_roles"  : suggested_roles
    }