from pydantic import BaseModel

class InputData(BaseModel):
    Age: int = 38
    Fare: float = 71.2833
    Sex: int = 1
    Pclass: int = 1
    Embarked: int = 0
    SibSp: int = 1
    Parch: int = 0


class SurvivalPredictionOut(BaseModel):
    survival_prediction: int  # 0 for not survived, 1 for survived
