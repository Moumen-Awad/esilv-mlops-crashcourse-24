from typing import List

import pandas as pd
from loguru import logger

from sklearn.feature_extraction import DictVectorizer

CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked']

def encode_categorical_cols(df, encoders):
    for col in CATEGORICAL_COLS:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            logger.warning(f"No encoder found for column {col}")
    return df