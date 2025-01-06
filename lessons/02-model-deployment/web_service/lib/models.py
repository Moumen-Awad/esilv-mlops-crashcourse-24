from pydantic import BaseModel

class InputData(BaseModel):
    pul: int = 254
    dul: int = 264
    pass_count: int = 1

class PredictionOUT(BaseModel):
    trip_duration: float