# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR = f"local_models/dv.pkl"
PATH_TO_MODEL = f"local_models/model.pkl"
PATH_TO_ENCODER = f"local_models/encoders.pkl"
CATEGORICAL_VARS = ['Pclass', 'Sex', 'Embarked']


# DISCRPTION
APP_TITLE = "TitanicSurvivalPredictionApp"
APP_DESCRIPTION = (
    "A machine learning API to predict whether a passenger survived the Titanic disaster, "
    "based on features such as age, sex, passenger class, and embarked location."
)
APP_VERSION = "0.0.1"
