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
from lib.utils import load_model, load_preprocessor, load_encoder

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=SurvivalPredictionOut, status_code=201)

def predict(payload: InputData):
    """
    Makes a prediction using a trained model, preprocessor, and encoder.
    This function loads the preprocessor, model, and encoder from disk using predefined paths, and then runs inference on the provided input data.
    Args:
        payload (InputData): The input data for prediction. Should be an instance of the `InputData` class, which contains the features necessary for prediction (e.g., Age, Sex, Fare).
    Returns:
        dict: A dictionary containing the survival prediction. The key is "survival_prediction" and the value is the predicted outcome (0 for not survived, 1 for survived).
    """
    dv = load_preprocessor(PATH_TO_PREPROCESSOR)
    model = load_model(PATH_TO_MODEL)
    encoder = load_encoder(PATH_TO_ENCODER)
    y = run_inference([payload], dv, model, encoder)
    return {"survival_prediction": y[0]}
