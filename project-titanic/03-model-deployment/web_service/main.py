from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_VERSION,
    PATH_TO_MODEL,
    PATH_TO_PREPROCESSOR,
    PATH_TO_ENCODER,
)

from fastapi import FastAPI
from lib.modelling import run_inference
from lib.models import InputData, SurvivalPredictionOut
from lib.utils import load_model, load_preprocessor

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=SurvivalPredictionOut, status_code=201)

def predict(payload: InputData):
    dv = load_preprocessor(PATH_TO_PREPROCESSOR)
    model = load_model(PATH_TO_MODEL)
    encoder = load_model(PATH_TO_ENCODER)
    y = run_inference([payload], dv, model, encoder)
    return {"survival_prediction": y[0]}
