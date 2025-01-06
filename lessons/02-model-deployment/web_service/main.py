from fastapi import FastAPI
from pydantic import BaseModel, StrictStr, StrictFloat, StrictInt
from lib.models import InputData, PredictionOUT
from lib.modelling import run_inference
from lib.utils import load_model, load_processor
from lib.preproccessing import exract_x_y

from app_config import APP_TITLE, APP_VERSION, PATH_TO_PIPELINE
# Define a data model for the request body
# We're using StrictStr to ensure that the name is a string
# More information here: https://stackoverflow.com/questions/72263682/checking-input-data-types-in-pydantic

# Initiate the FastAPI app
app = FastAPI(title=APP_TITLE, description="Description",version=APP_VERSION)

# Initiate the homepage of the API
@app.get("/")
def index():
    return {"message": "This is the homepage of the API realted to MLO course "}

@app.post("/inference", response_model=PredictionOUT, status_code=201)
def predict(payload: InputData):
    dv = load_processor("./local_models/dv_v0.0.1.pkl")
    #model = load_model('./local_models/model.pkl')
    y = run_inference(payload.dict(),dv,'./local_models/model.pkl')
    return {"trip_duration": y[0]}
    #return {"Result is": f"{l}"}

""""
# Define a POST operation for the path "/greet"
@app.post("/greet")
def greet_user(item: Item):
    return {"message": f"Hello, {item.name}"}
"""