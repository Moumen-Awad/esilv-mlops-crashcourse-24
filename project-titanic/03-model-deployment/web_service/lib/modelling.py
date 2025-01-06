from typing import List, Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer

from lib.models import InputData
from lib.preprocessing import CATEGORICAL_COLS, encode_categorical_cols

def run_inference(input_data: List[InputData], dv: DictVectorizer, model: BaseEstimator, encoder: Dict[str, LabelEncoder]) -> np.ndarray:
    """Run inference on a list of input data.

    Args:
        input_data (List[InputData]): the data points to run inference on.
        dv (DictVectorizer): the fitted DictVectorizer object.
        model (BaseEstimator): the fitted model object.

    Returns:
        np.ndarray: the predicted survival outcomes (0 for not survived, 1 for survived).

    Example payload:
        {'Age': 22, 'Sex': '0', 'Pclass': 3, 'Embarked': '0', 'SibSp': 1, 'Parch': 0}
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    df = encode_categorical_cols(df, encoder)  # Ensure categorical columns are encoded properly
    dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
    X = dv.transform(dicts)  # Transform the data using the DictVectorizer
    y = model.predict(X)  # Predict survival (0 or 1)
    logger.info(f"Predicted survival outcomes:\n{y}")
    return y
