import pickle
import numpy as np
import logging
from fastapi import HTTPException
from pathlib import Path

logger = logging.getLogger("model_utils")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

except Exception as e:
    logger.exception("Failed to load model.")
    model = None


def validate_features(features, expected_len=None):
    if not isinstance(features, (list, tuple)):
        raise HTTPException(status_code=400, detail="Features must be a list.")

    cleaned = []
    for f in features:
        try:
            cleaned.append(float(f))
        except:
            raise HTTPException(status_code=400, detail="All features must be numeric.")

    if expected_len and len(cleaned) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_len} features but got {len(cleaned)}."
        )

    return cleaned


def predict_model(features, expected_len=None):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    clean = validate_features(features, expected_len=expected_len)

    X = np.array(clean).reshape(1, -1)

    try:
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(X)[0][1]
            return float(pred)
        else:
            pred = model.predict(X)[0]
            return float(pred)
    except Exception as e:
        logger.exception("Model prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))
