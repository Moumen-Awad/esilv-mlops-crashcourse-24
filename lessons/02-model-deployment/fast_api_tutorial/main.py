from fastapi import FastAPI
from pydantic import BaseModel, StrictStr

# Define a data model for the request body
# We're using StrictStr to ensure that the name is a string
# More information here: https://stackoverflow.com/questions/72263682/checking-input-data-types-in-pydantic
class Item(BaseModel):
    name: StrictStr

# Initiate the FastAPI app
app = FastAPI()

# Initiate the homepage of the API
@app.get("/")
def index():
    return {"message": "This is the homepage of the API "}

# Define a POST operation for the path "/greet"
@app.post("/greet")
def greet_user(item: Item):
    return {"message": f"Hello, {item.name}"}