from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd

from fastapi_app.schemas import PredictRequest, PredictResponse
from fastapi_app.model_utils import predict_model, validate_features, logger

app = FastAPI(title="Loan Defaulter API")

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    logger.info("JSON prediction request received")
    score = predict_model(data.features, expected_len=None)
    return {"prediction": score}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    logger.info(f"CSV file received: {file.filename}")

    try:
        df = pd.read_csv(file.file)
    except:
        raise HTTPException(status_code=400, detail="Invalid CSV file.")

    preds = []
    for _, row in df.iterrows():
        features = row.values.tolist()
        features = validate_features(features)
        pred = predict_model(features)
        preds.append(pred)

    return {"rows": len(preds), "predictions": preds}
