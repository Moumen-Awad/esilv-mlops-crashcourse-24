from pydantic import BaseModel


class InputData(BaseModel):
    Age: float = 22
    Sex: str = 0
    Pclass: int = 3
    Embarked: str = 0
    SibSp: int = 1
    Parch: int = 0


class SurvivalPredictionOut(BaseModel):
    survival_prediction: int  # 0 for not survived, 1 for survived
